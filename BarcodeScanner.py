#python default packages
from dataclasses import dataclass
from itertools import product
from functools import partial
#pandas
import pandas as pd
#numpy and scipy
import numpy as np
from numpy.linalg import norm, inv
from scipy.stats import f

# within dissertation2 
import sample
import estimators
#joblib
from joblib import Parallel, delayed

# itertools
from itertools import combinations

# types
from typing import Union


@dataclass
class cluster_barcode_scanner:
    sample_generator: sample.sample_generator
    cluster_estimator: estimators.ClusteredSegmentation
    multi_processing: bool = True
        
    @property
    def C(self):
        if hasattr(self, '_C'):
            pass
        else:
            def gen_contrast(groupby_series, p):
                if len(groupby_series) == 1:
                    return None
                else:
                    result = []
                    for idx, mean_idx in enumerate(groupby_series.values.tolist()):
                        if idx == 0:
                            array_1 = np.array([1 if x == mean_idx else 0 for x in range(2**p)])
                        else:
                            array_2 = np.array([[1 if x == mean_idx else 0 for x in range(2**p)]])
                            contrast = (array_1 - array_2).reshape(1, -1)
                            result.append(contrast)
                    return np.concatenate(result, axis = 0)
            result = self.cluster_estimator.full_to_reduced_with_counts.copy()
            C = np.concatenate([x for x in result.groupby('reduced')['full'].apply(gen_contrast, p = self.sample_generator.p) if x is not None ], axis = 0)
            self._C = C
        return self._C


    @property
    def L_inv(self):
        if hasattr(self, '_L_inv'):
            pass
        else:
            L = self.sample_generator.L
            self._L_inv = inv(L)
        return self._L_inv
    
    @property
    def full_var(self):
        if hasattr(self, '_full_var'):
            pass
        else:
            self._full_var = self.cluster_estimator.full_var
        return self._full_var
        
    @staticmethod
    def get_single_distance(beta, L_inv, C, orth = False):
        """ given a contrast for beta, it computes the normalized mahalanobis distance between 
        L_inv*beta and I - P_c where P_c is a projection matrix with respect to C"""
        if isinstance(beta, list):
            beta = np.array(beta)
        assert max(beta.shape) == L_inv.shape[1]
        beta = beta.reshape(-1)
        ctc_inv = np.linalg.inv(C.dot(C.T))
        if orth:
            H = C.T.dot(ctc_inv).dot(C)
        else:
            H = (np.diag(np.ones(max(C.shape))) - C.T.dot(ctc_inv).dot(C))
        l_inv_b = L_inv.dot(beta)
        denom = np.array(l_inv_b.dot(l_inv_b))
        num   = np.array(l_inv_b.T.dot(H).dot(l_inv_b))
        try:
            output = num/denom
            return output
        except:
            return 0
    
    @staticmethod
    def get_cdf(beta_contrasts, L_inv, full_var, group_means, N, p):
        c = beta_contrasts #self.gen_beta_contrasts(beta)
        assert c.shape[0] <= c.shape[1]
        LC = L_inv.dot(c.T)
        mu = LC.T.dot(group_means) #self.cluster_estimator.full_to_reduced_with_counts.y
        var = LC.T.dot(full_var).dot(LC)
        f_value = mu.T.dot(inv(var)).dot(mu)/c.shape[0]
        cdf = f.cdf(f_value, c.shape[0], N-p, loc=0, scale=1)
        return cdf    
    
    
    @property
    def cdf_table(self):
        if hasattr(self, '_cdf_table'):
            pass
        else:
            all_beta = self.all_available_beta_for_test
            L_inv = self.L_inv
            full_var = self.full_var
            # fill the missing grouping
            # This part needs to be replaced with alternative methods flling the missing barcode
            # we used the overall mean to impute the missing groups mean
            if self.cluster_estimator.full_to_reduced_with_counts.shape[0] < self.cluster_estimator.full_p:
                max_group_id = self.cluster_estimator.full_to_reduced_with_counts.reduced.max()
                df = pd.DataFrame(range(self.cluster_estimator.full_p), columns = ['full'])
                df = df.merge(self.cluster_estimator.full_to_reduced_with_counts, how = 'left')
                df.counts.fillna(0, inplace = True); df.reduced.fillna(max_group_id + 1, inplace = True)
                df.y.fillna(self.cluster_estimator.overall_mean, inplace = True)
            else:
                df = self.cluster_estimator.full_to_reduced_with_counts
            self.cp = df
            group_means = df.y
            N = self.cluster_estimator.n
            p = self.cluster_estimator.full_p
            
            if self.multi_processing:
                cdf_func = partial(self.get_cdf, L_inv = L_inv, full_var = full_var, group_means = group_means,
                                   N = N, p = p)
                cdf = Parallel(n_jobs = -1, prefer = 'threads')(delayed(cdf_func)(self.gen_beta_contrasts(beta)) for beta in all_beta)
            else:
                cdf = []
                for beta in all_beta:
                    c = self.gen_beta_contrasts(beta)
                    cdf_value = self.get_cdf(c, L_inv, full_var, group_means, N, p)
                    cdf.append(cdf_value)
            df_list = []
            for beta, p_val in zip(all_beta, cdf):
                seq = list(beta) + [p_val]
                df_list.append(seq)
            df = pd.DataFrame(df_list, columns = self.sample_generator.beta_names + ['cdf'])
            columns = df.columns[df.columns.str.contains('\*')].tolist() + ['cdf']
            df = df.loc[:, columns].sort_values(['cdf'], ascending = False).reset_index(drop = True)
            df.cdf.fillna(0, inplace = True)
            self._cdf_table = df
        return self._cdf_table
        
        
    @property
    def cdf_ranking(self):
        if hasattr(self, '_cdf_ranking'):
            pass
        else:
            df = self.cdf_table
            df = df.loc[df.cdf.gt(0), :].reset_index(drop = True)
            result = df.apply(np.average, weights = df.cdf, axis = 0).sort_values(ascending = False)
            result.pop('cdf')
            ranking = pd.DataFrame(zip(result.index.tolist(), result), columns = ['coefficients', 'score'])
            ranking['ranking'] = ranking.score.rank(ascending = False, method = 'min')
            self._cdf_ranking = ranking
            del result, ranking
        return self._cdf_ranking
    
    @property
    def distance_table(self):
        if hasattr(self, '_distance_table'):
            pass
        else:
            all_beta = self.all_available_beta_for_test
#             avg_distance = []
            def return_distance(beta, L_inv, C):
                beta = self.gen_beta_contrasts(beta)
                dist = []
                for c in beta:
                    dist.append(self.get_single_distance(c, L_inv, C))
                return np.mean(dist)
            
            def gen_C_distance(all_beta, orth = False):
                distance = partial(return_distance, L_inv = self.L_inv, C = self.C)
                ###################################################################################################
                if self.multi_processing:
                    avg_distance = Parallel(n_jobs=-1, prefer = 'threads')(delayed(distance)(beta) for beta in all_beta) 
                else:
                    avg_distance = []
                    for beta in all_beta:
                        dist = []
                        beta = self.gen_beta_contrasts(beta)
                        for c in beta:
                            dist.append(self.get_single_distance(c, self.L_inv, self.C, orth = orth))
                        avg_distance.append(np.mean(dist))
                return avg_distance
            avg_distance = gen_C_distance(all_beta)
            df_list = []
            for beta, dist in zip(all_beta, avg_distance):
                seq = list(beta) + [dist]
                df_list.append(seq)
            df = pd.DataFrame(df_list, columns = self.sample_generator.beta_names + ['distance'])
            columns = df.columns[df.columns.str.contains('\*')].tolist() + ['distance']
            df = df.loc[:, columns].sort_values(['distance'], ascending = False).reset_index(drop = True)
            try:
                base_line = df.loc[df.sum(axis = 1)-df.distance == 0,'distance'].values[0]
#             base_line_2 = df.loc[df.sum(axis = 1)-df.distance == 0, 'orth_dist'].values[0]
                if df.distance.gt(base_line).sum():
                    self._distance_table = df.loc[df.distance.gt(base_line),:]
                else:
                    self._distance_table = df #if all the distance is less than the base_line, return first 25%
            except:
                self._distance_table = df
            del df, df_list, avg_distance, seq
        return self._distance_table
    
    
    def get_distance_ranking(self, percentile = .75, normalize = True):
        if (1-percentile)*self.distance_table.shape[0] <2:
            N = 2 # at least two lines
        else:
            N = int((1-percentile)*self.distance_table.shape[0])
        df = self.distance_table
        new_df = df.copy()
        prop = 1-percentile
        weights_seq = new_df.distance[:N]
        if normalize:
            self._weights = (weights_seq - weights_seq.min() + 1e-3)/weights_seq.std()
        else:
            self._weights = weights_seq
        avg = partial(np.average, weights = self._weights)
        if prop == 1:
            result = new_df
        result = new_df.iloc[:N,:].apply(avg, axis = 0).sort_values(ascending = False)
        result.pop('distance')
        ranking = pd.DataFrame(zip(result.index.tolist(), result), columns = ['coefficients', 'proportion'])
        ranking['ranking'] = ranking.proportion.rank(ascending = False, method = 'min')
        return ranking
        
    def set_beta_sum_range(self, low = None, high = None):
        if low:
            assert low > 0
        else:
            low = 1
        if high:
            assert high < 2**self.sample_generator.p - self.sample_generator.p
        else:
            high = 4
        self._beta_sum_range = (low, high)
        return self._beta_sum_range
        
        
    @property
    def beta_sum_range(self):
        if hasattr(self, '_beta_sum_range'):
            pass
        else:
            self._beta_sum_range = self.set_beta_sum_range()
        return self._beta_sum_range

    @property
    def all_available_beta_for_test(self):
        if hasattr(self, '_all_available_beta_for_test'):
            pass
        else:
            non_zero_part = product([0,1], repeat = 2**self.sample_generator.p - self.sample_generator.p - 1)
            zeros = [0] + [0 for _ in range(self.sample_generator.p)]
            if hasattr(self, '_beta_sum_range'):
                beta_range = list(range(self.beta_sum_range[0], self.beta_sum_range[1]+1))
                self._all_available_beta_for_test = np.array([zeros + list(x) for x in non_zero_part if sum(x) in beta_range ]) #if sum(x)
            else:
                self._all_available_beta_for_test = np.array([zeros + list(x) for x in non_zero_part if sum(x)]) #if sum(x)
        return self._all_available_beta_for_test
    
   
    
    @property
    def all_distance(self): # this part needs to be done in multi-processing fashion later...
        if hasattr(self, '_all_distance'):
            pass
        else:
            from scipy.linalg import svd
            all_beta = self.all_available_beta_for_test
            avg_distance = []
            for beta in all_beta:
                dist = []
                k = sum(beta)
                beta = self.gen_beta_contrasts(beta)
                orths = svd(beta.T)[0][:, :k].T
                for c in orths:
                    dist.append(self.get_single_distance(c, scanner.L_inv, scanner.C))
                avg_distance.append(np.mean(dist))
            df_list = []
            for beta, dist in zip(all_beta, avg_distance):
                seq = list(beta) + [dist]
                df_list.append(seq)
            self._all_distance = pd.DataFrame(df_list) # need to add columns
        return self._all_distance
    
    
    
    def gen_beta_contrasts(self, beta):
        if isinstance(beta, list):
            beta = np.array(beta)
        beta = beta.reshape(-1)
        assert beta.shape[0] == 2**self.sample_generator.p
        assert beta.max() == 1
        where_ones= np.where(beta)[0]
        if where_ones.shape[0] == 1:
            return beta.reshape(1, -1)
        else:
            N = 2**self.sample_generator.p
            contrasts = []
            for idx, beta_idx in enumerate(where_ones):
                if idx:
                    c_compare = np.zeros(N, dtype = np.byte).reshape(1, -1); c_compare[0,beta_idx] = 1
                    c = c_base - c_compare
                    contrasts.append(c/norm(c, 2))
                    c_base = c.copy(); c_base[0, c_base_index ] += 1
                else:
                    c_base = np.zeros(N, dtype = np.byte).reshape(1, -1)
                    c_base[0,beta_idx] = 1
                    contrasts.append(c_base)
                    c_base_index = beta_idx
            contrasts = np.concatenate(contrasts, axis = 0)
            return contrasts
        


class base_barcode:
    def __init__(self, X, y):
        self.X = X
        self.clean_X()
        self.y = y
        self.clean_y()

    def clean_X(self):
        if isinstance(self.X, np.ndarray):
            self.X = pd.DataFrame(self.X, columns = [f"x_{i}" for i in range(self.X.shape[1])])
            self.original_columns = self.X.columns.tolist()
        elif isinstance(self.X, pd.DataFrame):
            self.original_columns = self.X.columns.tolist()
        self.p = self.X.shape[1]

    def clean_y(self):
        if isinstance(self.y, np.ndarray):
            if self.y.shape[1]:
                pass
            else:
                self.y = self.y.reshape(-1, 1)
        elif isinstance(self.y, pd.DataFrame):
            self.y = self.y.iloc[:, 0].to_numpy().reshape(-1,1)
        elif isinstance(self.y, pd.Series):
            self.y = self.y.to_numpy().reshape(-1,1)

            
    @property
    def beta_names(self):
        if hasattr(self, '_beta_names'):
            pass
        else:
            beta_raw_names = [f'beta{i}' for i in range(1, self.p + 1)]
            beta_names = [f'beta{i}' for i in range(self.p + 1)]
            for r in range(2, self.p):
                beta_names += ['*'.join(x) for x in combinations(beta_raw_names, r)]
            beta_names += ['*'.join(beta_raw_names)]
            self._beta_names = beta_names
            del beta_names
            del beta_raw_names
        return self._beta_names

    @property
    def barcode(self):
        if hasattr(self, '_barcode'):
            pass
        else:
            self._barcode = self.gen_barcode(self.X)
        return self._barcode
    
    @staticmethod
    def gen_barcode(X):
        pack_bits = np.packbits(np.array(X), axis = -1)
        if pack_bits.shape[1]>1:
            def gen_barcode(a, k, return_float = False):
                m = len(a)-1
                total_sum = 0
                for i, x in enumerate(a):
                    if i < m:
                        if return_float:
                            adjust = (x * 2**(m-i-1) * 2**k)
                            total_sum += float(adjust)
                        else:
                            total_sum += (x * 2**(m-i-1) << k)
                    else:
                        total_sum += (x >> (8-k))
                return total_sum
            if X.shape[1] > 500:
                barcode = partial(gen_barcode, k = X.shape[1]%8, return_float = True)
            else:
                barcode = partial(gen_barcode, k = X.shape[1]%8)
        else:
            def adjust_barcode(a, k):
                a = a >> k
                return a
            barcode = partial(adjust_barcode, k = 8-X.shape[1])
        output = np.apply_along_axis(barcode, 1, pack_bits)
        return output
    
    
    @staticmethod
    def barcode_to_beta(barcode):
        if isinstance(barcode, list):
            output = [1] + barcode
        elif isinstance(barcode, Union[int, float]):
            barcode = int(barcode)
            barcode = bin(barcode)[2:]
            output = [1] + [int(x) for x in list(barcode)]
            barcode = output.copy()
        else:
            output = [1] + list(barcode)
        N = len(output)
        for i in range(2, N):
            output += [np.prod(x) for x in combinations(barcode, i)]
    #     output += [np.prod(output)]
        return output
    
    @property
    def all_combinations(self):
        if hasattr(self, '_all_combinations'):
            pass
        else:
            all_sets = list(set(product([0,1], repeat = self.p))); all_sets.sort()
            all_comb = pd.DataFrame(np.array([list(x) for x in all_sets]), columns = self.original_columns)
            self._all_combinations = all_comb
        return self._all_combinations


    @property
    def L(self):
        if hasattr(self, '_L'):
            pass
        else:
            all_sets = list(set(product([0,1], repeat = self.p))); all_sets.sort()
            self._L = np.array([self.barcode_to_beta(x) for x in all_sets]).astype(np.int8)
        return self._L
    
    @property
    def L_inv(self):
        if hasattr(self, '_L_inv'):
            pass
        else:
            L = self.L
            self._L_inv = np.linalg.inv(self.L).round(0).astype(np.int8)
        return self._L_inv
    








class tree_and_clustering(base_barcode):
    def __init__(self, X, y, tree_model):
        super().__init__(X, y)
        self.clean_data()
        self.estimator = tree_model
        self.fit()

    def clean_data(self):
        full_df = self.X.copy()
        full_df['y'] = self.y.reshape(-1)
        sorted_index = full_df.sort_values(full_df.columns.tolist()[:-1]).index.tolist()
        self.full_df = full_df.loc[sorted_index, :].reset_index(drop = True)
        self.barcode_df = pd.DataFrame(zip(self.barcode.reshape(-1), self.y.reshape(-1)), columns = ['z','y']).loc[sorted_index, :].reset_index(drop = True)
        self.X = full_df.iloc[:, :-1]
        self.y = full_df.y.to_numpy().reshape(-1,1)
        del sorted_index
        del full_df
        
    def fit(self):   
        X_train = self.X.copy()
        y_train = self.y.copy()     
        self.estimator.fit(X_train, y_train.reshape(-1))
        self._fit = True

    def predict_from_training(self):
        if hasattr(self, '_fit'):
            if self._fit:
                pass
            else:
                self.fit()
        else:
            self.fit()
        full_df = self.full_df.copy()
        full_df['y_hat'] = self.estimator.predict(full_df.loc[:, self.original_columns])
        full_df['sq'] = (full_df.y_hat - full_df.y)**2
        summary = full_df.groupby(self.original_columns).agg({"y_hat": np.mean, "sq": [lambda x: x.sum()/x.count(), 'count']}).reset_index()
        summary.columns = self.original_columns + ['group_means','mse','count']
        return summary
    
    @property
    def summary_table(self):
        if hasattr(self, "_summary_table"):
            pass
        else:
            self._summary_table = self.predict_from_training()
        return self._summary_table
    
    @property
    def init_mu_hat(self):
        if hasattr(self, "_init_mu_hat"):
            pass
        else:
            self._init_mu_hat = self.summary_table.group_means.tolist()
            self._init_mu_hat = np.array(self._init_mu_hat).reshape(-1)
        return self._init_mu_hat
    
    @property
    def init_mu_var(self):
        if hasattr(self, "_init_mu_var"):
            pass
        else:
            self._init_mu_var = np.diag(self.summary_table.mse/self.summary_table['count'])
        return self._init_mu_var
    
    @property
    def init_beta_hat(self):
        if hasattr(self, "_init_beta_hat"):
            pass
        else:
            self._init_beta_hat = self.L_inv @ self.init_mu_hat
        return self._init_beta_hat

    @property
    def init_beta_var(self):
        if hasattr(self, "_init_beta_var"):
            pass
        else:
            self._init_beta_var = self.L_inv @ self.init_mu_var @ self.L_inv.T
        return self._init_beta_var

    @property
    def clustering_init_kwargs(self):
        summary = self.summary_table
        kwargs = {"means": summary.group_means.tolist(), "variances": summary.mse.tolist(), "sample_sizes": summary['count'].tolist()}
        return kwargs
    
    @property
    def init_cluster_idx(self):
        if hasattr(self, "_init_cluster_idx"):
            pass
        else:
            sample_sizes = self.clustering_init_kwargs['sample_sizes']
            clusters = []
            last = 0
            for n in sample_sizes:
                clusters.append([x for x in range(last, n+last)])
                last += n
            self._init_cluster_idx = clusters
        return self._init_cluster_idx

    @property
    def init_pdist(self):
        if hasattr(self, "_init_pdist"):
            pass
        else:
            self._init_pdist = self.pairwise_distances_from_means_variances(**self.clustering_init_kwargs)
        return self._init_pdist

    def cluster(self, n_clusters, save = True):
        self.last_n_clusters = n_clusters
        pdist = self.init_pdist.copy()
        init_cluster_idx = self.init_cluster_idx.copy()
        result = self.agglomerative_clustering(pairwise_distances = pdist, n_clusters = n_clusters, clusters = init_cluster_idx)
        cluster_idx = result[0]
        final_pdist = result[1]
        cluster_df = {}
        for cluster_id, cluster_index in enumerate(cluster_idx):
            cluster_id_name = f"cluster_{cluster_id}"
            cluster_df[cluster_id_name] = self.full_df.loc[cluster_index,:].copy()
            cluster_df[cluster_id_name] = cluster_df[cluster_id_name].groupby(self.original_columns).agg(np.mean).reset_index()
            cluster_df[cluster_id_name]['barcode'] = self.gen_barcode((cluster_df[cluster_id_name].loc[:, self.original_columns]))
        if save:
            self._latest_cluster_dfs = cluster_df
        else:
            self._latest_cluster_dfs = None
        
        return {"cluster": cluster_df, "final_pdist": final_pdist}

    def gen_mu_contrast_from_cluster(self, n_clusters, use_latest_cluster_df = True):
        if use_latest_cluster_df:
            if hasattr(self, '_latest_cluster_dfs'):
                if (self._latest_cluster_dfs != None) & (self.last_n_clusters == n_clusters):
                    cluster_df_list = self._latest_cluster_dfs
        try:
            cluster_df_list
        except:
            cluster_result = self.cluster(n_clusters = n_clusters)
            cluster_df_list = cluster_result['cluster']
            
        contrast_matrix = 0
        for cluster_name, cluster_df in cluster_df_list.items():
            if cluster_df.shape[0] > 1:
                contrast = np.zeros((cluster_df.shape[0]-1, 2**len(self.original_columns)))
                for i, row in enumerate(contrast):
                    row[cluster_df.barcode[0]] = 1
                    row[cluster_df.barcode[i+1]] = -1
                    contrast[i] = row
                if isinstance(contrast_matrix, np.ndarray):
                    contrast_matrix = np.concatenate([contrast_matrix, contrast], axis = 0)
                else:
                    contrast_matrix = contrast
            else:
                pass
        return contrast_matrix
    
    def gen_projection_matrix_mu(self, n_clusters):
        C = self.gen_mu_contrast_from_cluster(n_clusters)
        projection_matrix = C.T @ np.linalg.inv(C @ C.T) @ C
        projection_matrix = np.identity(C.shape[1]) - projection_matrix
        return projection_matrix
    
    
    def get_projected_beta_hat(self, n_clusters):
        mu = self.init_mu_hat
        proj_mu = self.gen_projection_matrix_mu(n_clusters = n_clusters) @ mu
        proj_beta = self.L_inv @ proj_mu
        return proj_beta
    
    def get_projected_beta_hat_var(self, n_clusters):
        mu_var = self.init_mu_var
        proj = self.gen_projection_matrix_mu(n_clusters = n_clusters)
        return self.L_inv @ proj @ mu_var @ proj.T @ self.L_inv.T

    
    def ward_linkage(self, pairwise_distances, clusters, merge_indices):
        clusters = clusters.copy()
        i, j = merge_indices
        cluster_i = clusters[i]
        cluster_j = clusters[j]
        n_i = len(cluster_i)
        n_j = len(cluster_j)
        n = pairwise_distances.shape[1]
        new_distances = pairwise_distances.copy()
        new_distances = np.delete(new_distances, merge_indices, axis = 0)
        new_distances = np.delete(new_distances, merge_indices, axis = 1)
        new_distances = np.append(new_distances, np.zeros((1, new_distances.shape[1])), axis = 0)
        new_distances = np.append(new_distances,  np.zeros((new_distances.shape[0], 1)), axis = 1)
        clusters.append(cluster_i + cluster_j)
        clusters.remove(cluster_i)
        clusters.remove(cluster_j)
        col = 0
        
        for k in range(n):  # Subtract 2 because we've already added a row and column
            if k != i and k != j:
                n_k = len(clusters[col])
                n_all = n_i + n_j + n_k
                dist_ik = pairwise_distances[i, k]*(n_k + n_i)/n_all
                dist_jk = pairwise_distances[j, k]*(n_j + n_k)/n_all
                dist_ij = pairwise_distances[i, j]*(n_k)/n_all
                new_dist = dist_ik + dist_jk - dist_ij
                # Add other linkage methods here if needed
                
                new_distances[-1, col] = new_dist
                new_distances[col, -1] = new_dist
                col += 1

        return new_distances, clusters

    def agglomerative_clustering(self, pairwise_distances, n_clusters, clusters = None, estimator = None):
        n = pairwise_distances.shape[0]
        if clusters:
            pass
        else:
            clusters = [[i] for i in range(n)]
        
        assert len(clusters) == n
        
        for k in range(n - n_clusters):
            min_dist = np.inf
            merge_indices = None
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = pairwise_distances[i, j]
                    if dist < min_dist:
                        min_dist = dist
                        merge_indices = (i, j)
            
            if merge_indices:
                pairwise_distances, clusters = self.ward_linkage(pairwise_distances, clusters, merge_indices)
        
        return clusters, pairwise_distances

    
    
    def pairwise_distances_from_means_variances(self, means, variances, sample_sizes):
        num_clusters = len(means)
        pairwise_distances = np.zeros((num_clusters, num_clusters))
        
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                n_i = sample_sizes[i]
                n_j = sample_sizes[j]
                mean_i = means[i]
                mean_j = means[j]
                var_i = variances[i]
                var_j = variances[j]
                
                # Calculate the pairwise distance using Ward linkage formula
                numerator = (n_i * n_j / (n_i + n_j)) * (mean_i - mean_j)**2
                denominator = np.sqrt((n_i * var_i + n_j * var_j) / (n_i + n_j))
                
                pairwise_distances[i, j] = np.sqrt(numerator / denominator)
                pairwise_distances[j, i] = pairwise_distances[i, j]
        
        return pairwise_distances


@dataclass
class summary:
    summary: pd.DataFrame
    mu_hat: np.ndarray
    mu_var: np.ndarray
    beta_hat: np.ndarray
    beta_var: np.ndarray
    pdist_kwargs: dict
    pdist: np.ndarray
    init_cluster_idx: list

class rf_and_clustering(tree_and_clustering):
    def __init__(self, X, y, random_forest_model):
        super().__init__(X, y, tree_model = random_forest_model)
        self.estimators = self.estimator.estimators_

    def _gen_summary(self, estimator):
        full_df = self.full_df.copy()
        full_df['y_hat'] = estimator.predict(full_df.loc[:, self.original_columns].to_numpy())
        full_df['sq'] = (full_df.y_hat - full_df.y)**2
        summary = full_df.groupby(self.original_columns).agg({"y_hat": np.mean, "sq": [lambda x: x.sum()/x.count(), 'count']}).reset_index()
        summary.columns = self.original_columns + ['group_means','mse','count']
        return summary
    
    def _gen_cluster_idx(self, sample_sizes_by_cluster):
        clusters = []
        last = 0
        for n in sample_sizes_by_cluster:
            clusters.append([x for x in range(last, n+last)])
            last += n
        return clusters
    
    def _single_clustering(self, n_clusters, pdist, init_custer_idx):
        result = self.agglomerative_clustering(pairwise_distances = pdist, n_clusters = n_clusters, clusters = init_cluster_idx)
        cluster_idx = result[0]
        final_pdist = result[1]
        cluster_df = {}
        for cluster_id, cluster_index in enumerate(cluster_idx):
            cluster_id_name = f"cluster_{cluster_id}"
            cluster_df[cluster_id_name] = self.full_df.loc[cluster_index,:].copy()
            cluster_df[cluster_id_name] = cluster_df[cluster_id_name].groupby(self.original_columns).agg(np.mean).reset_index()
            cluster_df[cluster_id_name]['barcode'] = self.gen_barcode((cluster_df[cluster_id_name].loc[:, self.original_columns]))
        return {"cluster": cluster_df, "final_pdist": final_pdist}

    def _gen_statistics(self, estimator):
        summary_table = self._gen_summary(estimator)
        mu_hat = summary_table.group_means.to_numpy().reshape(-1)
        mu_var = np.diag(summary_table.mse/summary_table['count'])
        beta_hat = self.L_inv @ mu_hat
        beta_var = self.L_inv @ mu_var @ self.L_inv.T
        pdist_kwargs = {"means": summary_table.group_means.tolist(),
                         "variances": summary_table.mse.tolist(), 
                         "sample_sizes": summary_table['count'].tolist()}
        pdist = self.pairwise_distances_from_means_variances(**pdist_kwargs)
        init_cluster_idx = self._gen_cluster_idx(pdist_kwargs['sample_sizes'])
        estimator_summary = summary(summary_table, mu_hat, mu_var, beta_hat, beta_var, pdist_kwargs, pdist, init_cluster_idx)
        return estimator_summary

    def _cluster_estimator(self, estimator_summary, n_clusters):
        pdist = estimator_summary.pdist
        init_cluster_idx = estimator_summary.init_cluster_idx
        result = self.agglomerative_clustering(pairwise_distances = pdist, n_clusters = n_clusters, clusters = init_cluster_idx)
        cluster_idx = result[0]
        final_pdist = result[1]
        cluster_df = {}
        for cluster_id, cluster_index in enumerate(cluster_idx):
            cluster_id_name = f"cluster_{cluster_id}"
            cluster_df[cluster_id_name] = self.full_df.loc[cluster_index,:].copy()
            cluster_df[cluster_id_name] = cluster_df[cluster_id_name].groupby(self.original_columns).agg(np.mean).reset_index()
            cluster_df[cluster_id_name]['barcode'] = self.gen_barcode((cluster_df[cluster_id_name].loc[:, self.original_columns]))
        return {"cluster": cluster_df, "final_pdist": final_pdist}
    
    def _gen_contrast_matrix(self, cluster_df_list):
        contrast_matrix = 0
        for cluster_name, cluster_df in cluster_df_list.items():
            if cluster_df.shape[0] > 1:
                contrast = np.zeros((cluster_df.shape[0]-1, 2**len(self.original_columns)))
                for i, row in enumerate(contrast):
                    row[cluster_df.barcode[0]] = 1
                    row[cluster_df.barcode[i+1]] = -1
                    contrast[i] = row
                if isinstance(contrast_matrix, np.ndarray):
                    contrast_matrix = np.concatenate([contrast_matrix, contrast], axis = 0)
                else:
                    contrast_matrix = contrast
            else:
                pass
        return contrast_matrix

    def _gen_mu_contrast_from_cluster(self, estimator_summary, n_clusters):
        cluster_result = self._cluster_estimator(estimator_summary = estimator_summary, n_clusters = n_clusters)
        cluster_df_list = cluster_result['cluster']
        contrast_matrix = self._gen_contrast_matrix(cluster_df_list = cluster_df_list)
        return contrast_matrix
    
    def projected_beta_for_single_estimator(self, estimator, n_clusters):
        estimator_summary = self._gen_statistics(estimator = estimator)
        C_mu = self._gen_mu_contrast_from_cluster(estimator_summary = estimator_summary, n_clusters = n_clusters)
        projection_matrix = C_mu.T @ np.linalg.inv(C_mu @ C_mu.T) @ C_mu
        projection_matrix = np.identity(C_mu.shape[1]) - projection_matrix
        init_mu = estimator_summary.mu_hat.reshape(-1)
        mu_hat = projection_matrix @ init_mu
        beta_hat = self.L_inv @ mu_hat
        return beta_hat
    
    
    def importance_mean(self, n_clusters):
        beta_hat_arrays = [self.projected_beta_for_single_estimator(e, n_clusters).reshape(1, -1) for e in self.estimator.estimators_]
        beta_hat_arrays = np.concatenate(beta_hat_arrays, axis = 0)
        # importance_array = np.abs(beta_hat_arrays)
        importance_array = beta_hat_arrays
        importance_pd = pd.DataFrame(importance_array, columns = self.beta_names)
        return importance_pd
