from bitstring import Bits
from dask import dataframe as dd
from typing import List, Union
import numpy as np
import pandas as pd
import scipy
from itertools import product, combinations
import math
import os

#python default packages
from dataclasses import dataclass
from functools import partial
#numpy and scipy
from numpy.linalg import norm, inv
from scipy.stats import f
from scipy.sparse import diags, csr_matrix, csc_matrix
from scipy.sparse.linalg import inv as sparse_inv

from sympy import Symbol


@dataclass
class f_test_result:
    nu_1:int
    nu_2:int
    f_statistic: float
    
    @property
    def p_value(self):
        if hasattr(self, '_p_value'):
            pass
        else:
            sdf = f.sf(self.f_statistic, self.nu_1, self.nu_2, loc=0, scale=1)
            self._p_value = sdf
        return self._p_value
    
    @property
    def cdf(self):
        if hasattr(self, '_cdf'):
            pass
        else:
            self._cdf = 1 - self.p_value
        return self._cdf



class dask_parameter_generator():
    def __init__(self, dask_dataframe, output_feature_name = None):
        self.ddf = self.init_process_dataframe(dask_dataframe, output_feature_name)

    @property
    def beta_names(self):
        if hasattr(self, '_beta_names'):
            pass
        else:
            beta_raw_index = [str(i) for i in range(1, self.num_main_var + 1)]
            beta_names = ['beta_0'] + [f'beta_{i}' for i in beta_raw_index]
            for r in range(2, self.num_main_var):
                interaction_beta_index = (','.join(x) for x in combinations(beta_raw_index, r))
                beta_names += [f'beta_{i}' for i in interaction_beta_index]
            beta_names += [f"beta_{','.join(beta_raw_index)}"]
            self._beta_names = beta_names
            del beta_names
            del beta_raw_index
        return self._beta_names

    @property
    def beta_names_symbol(self):
        if hasattr(self, '_beta_names_symbol'):
            pass
        else:
            self._beta_names_symbol = [Symbol(x) for x in self.beta_names]
        return self._beta_names_symbol



    @staticmethod
    def get_L(barcode_length):
        """
        Since L is a mapping from \beta to \mu, L should be in the sparse.csc format so that each column indicates how each \beta is represented with \mu
        """
        def barcode_to_beta(barcode):
            if isinstance(barcode, list):
                output = [1] + barcode
            else:
                output = [1] + list(barcode)
            N = len(output)
            for i in range(2, N):
                output += [np.prod(x) for x in combinations(barcode, i)]
        #     output += [np.prod(output)]
            return output
        all_sets = list(set(product([0,1], repeat = barcode_length))); all_sets.sort()
        linear_function_L = np.array([barcode_to_beta(x) for x in all_sets]).T.astype(np.int8)
        linear_function_L = csc_matrix(linear_function_L)
        del all_sets
        return linear_function_L
    

    def compute(self, return_type:str = 'dask'):
        if return_type == 'dask':
            return dd.from_pandas(self.ddf.compute(), chunksize = 10)
        elif return_type == 'pandas':
            return self.ddf.compute()
        elif return_type == 'numpy':
            return self.ddf.compute().to_numpy()
        else:
            raise ValueError("return type can be one of the followings: 'dask','pandas','numpy'")

    def init_process_dataframe(self, dask_dataframe, output_feature_name = None):
        ddf = dask_dataframe
        if output_feature_name:
            input_features = ~ddf.columns.str.contains(output_feature_name)
        else:
            input_features = ddf.columns.str.contains('x')
        # Finding basic information of the full linear model
        self.num_main_var = sum(input_features); self.sample_size = ddf.shape[0].compute()
        self.num_full_var = 2**self.num_main_var
        self.L = self.get_L(self.num_main_var)

        dtype = self.find_integer_type(input_features)
        for input_name in ddf.columns[input_features].tolist():
            ddf[input_name] = ddf[input_name].astype(np.int8)
        self.original_ddf = ddf.copy()
        ddf['barcode'] = ddf.loc[:, input_features].apply(self.return_bitstring, axis = 1, meta=(None, dtype))
        ddf = ddf.loc[:, ~np.concatenate([input_features, [False]])]
        ddf = ddf.set_index('barcode')
        return ddf

    @staticmethod
    def find_integer_type(input_features):
        Sum = sum(input_features)
        if Sum <= 8:
            return 'uint8'
        elif Sum <= 16:
            return 'uint16'
        elif Sum <= 32:
            return 'uint32'
        elif Sum <= 64:
            return 'uint64'
        else:
            return int

    @staticmethod
    def return_bitstring(x_seq):
        b = ''.join((str(z) for z in x_seq))
        bit = Bits(bin = b)
        return bit.uint

    # Groupby Y Means
    @property
    def full_segment_means(self):
        if hasattr(self, '_full_segment_means'):
            pass
        else:
            self._full_segment_means = self.find_groupby_means()
        return self._full_segment_means



    def find_groupby_means(self):
        """
        Finding the groupby means by the bits
        """
        y_means = self.ddf.groupby('barcode').y.mean()
        return y_means.reset_index().set_index('barcode')

    # SSTO, SSE, MSE, FIM
    @property
    def ssto(self):
        if hasattr(self, '_ssto'):
            pass
        else:
            self.gen_scale_estimates()
        return self._ssto
    
    @property
    def sse(self):
        if hasattr(self, '_sse'):
            pass
        else:
            self.gen_scale_estimates()
        return self._sse


    @property
    def MLE_SCALE(self):
        if hasattr(self, '_MLE_SCALE'):
            pass
        else:
            self.gen_scale_estimates()
        return self._MLE_SCALE
    
    @property
    def U_SCALE(self):
        if hasattr(self, '_UNBIASED_SCALE'):
            pass
        else:
            self.gen_scale_estimates()
        return self._UNBIASED_SCALE
    
    @property
    def MLE_VAR(self):
        if hasattr(self, '_mle_var'):
            pass
        else:
            self.gen_scale_estimates()
        return self._mle_var

    @property
    def U_VAR(self):
        if hasattr(self, '_unbiased_var'):
            pass
        else:
            self.gen_scale_estimates()
        return self._unbiased_var


    def gen_scale_estimates(self):
        ssto, sse = self.find_error_statistics()
        sse = sse.compute()
        self._sse = sse
        self._ssto = ssto.compute()
        if self.num_full_var >= self.sample_size:
            self._unbiased_var = None
            self._UNBIASED_SCALE = None
        else:
            self._unbiased_var = sse/(self.sample_size- self.num_full_var)
            self._UNBIASED_SCALE = math.sqrt(self._unbiased_var)
        self._mle_var = sse/(self.sample_size)
        self._MLE_SCALE = math.sqrt(self._mle_var)

    @property
    def y_mean(self):
        if hasattr(self, '_y_mean'):
            pass
        else:
            self._y_mean = self.ddf.y.mean().compute()
        return self._y_mean
    
    def find_error_statistics(self):
        result = self.ddf.merge(self.full_segment_means, how = 'left', left_index = True, right_index = True, suffixes = ['','_pred'])
        sse = result.apply(lambda x: (x[0]- x[1])**2, axis = 1, meta=(None, 'float64')).sum()
        y_mean = result.y.mean().compute()
        ssto = result.y.map(lambda y: (y-y_mean)**2, meta=(None, 'float64')).sum()
        return ssto, sse
    
    # Covariance and FIM
    @property
    def cell_means_covariance(self):
        if hasattr(self, '_cell_means_covariance'):
            pass
        else:
            self.find_FIM()
        return self._cell_means_covariance
    
    @property
    def cell_means_fim(self):
        if hasattr(self, '_cell_means_fim'):
            pass
        else:
            self.find_FIM()
        return self._cell_means_fim

    def find_FIM(self) -> scipy.sparse._dia.dia_matrix:
        raw_counts = pd.Series(self.ddf.groupby('barcode').y.count().compute().sort_index())
        sparse_diagonal_matrix= diags(raw_counts.reindex(list(range(self.num_full_var)), fill_value = 0))/self.MLE_VAR
        self._cell_means_fim = sparse_diagonal_matrix
        self._cell_means_covariance = diags(list(map(lambda x: 1/x if x else float('inf'), sparse_diagonal_matrix.diagonal())))


    def contrast_generator(self, max_variables = None):
        from scipy.sparse import vstack
        all_interaction_betas = scipy.sparse.eye(self.num_full_var - self.num_main_var -1, self.num_full_var, k = self.num_main_var + 1)
        betas_in_test = product([False, True], repeat = self.num_full_var - self.num_main_var -1)
        
        def get_new_contrasts(beta):
            vstacks = []
            for i, b in enumerate(beta):
                if b:
                    vstacks.append(all_interaction_betas.getrow(i))
            return vstack(vstacks).astype(np.uint8)
        if max_variables is not None:
            for beta in betas_in_test:
                if sum(beta) and (sum(beta) <= max_variables):
                    yield get_new_contrasts(beta)
        else:
            for beta in betas_in_test:
                if sum(beta):
                    yield get_new_contrasts(beta)


    @property
    def L_inv(self):
        if hasattr(self, '_L_inv'):
            pass
        else:
            L_inv = sparse_inv(self.L)
            L_inv.data = np.round(L_inv.data,0)
            self._L_inv = L_inv.astype(np.int8)
        return self._L_inv

    @staticmethod    
    def partial_f_test(contrast, L_inv, cell_means, cell_means_covariance, sample_size):
        assert contrast.shape[0] < contrast.shape[1]
        assert contrast.shape[1] == L_inv.shape[0]
        assert L_inv.shape[0] == L_inv.shape[1]
        assert cell_means_covariance.shape == L_inv.shape
        nu_1 = contrast.shape[0]
        nu_2 = sample_size - L_inv.shape[0]
        contrast = contrast.toarray()

        LC = contrast @ L_inv.T
        mu = LC @ cell_means
        var = LC @ cell_means_covariance @ LC.T
        f_value = mu.T @ inv(var) @ mu
        f_value /= nu_1
        
        return f_test_result(nu_1, nu_2, f_value)
    


    def gen_importance_score(self, filename = 'importance_score.csv', dir = os.getcwd(), max_variables = None):
        cell_means = self.find_groupby_means().compute().y.to_numpy()
        kwargs = {"L_inv": self.L_inv, "cell_means": cell_means, "cell_means_covariance": self.cell_means_covariance, "sample_size": self.sample_size}

        from csv import writer
        from tqdm import tqdm
        with open(os.path.join(dir, filename), 'w') as f:
            csv_writer = writer(f)
            csv_writer.writerow(self.beta_names + ['score'])
            contrasts =  self.contrast_generator(max_variables = max_variables)
            from math import comb
            if max_variables:
                total_iter = sum([comb(self.num_full_var - self.num_main_var -1, x) for x in range(1, max_variables +1)])
            else:
                total_iter = sum([comb(self.num_full_var - self.num_main_var -1, x) for x in range(1,self.num_full_var - self.num_main_var)])
            if total_iter < os.cpu_count() * 10:
                for contrast in tqdm(contrasts, total = total_iter):
                    result = self.partial_f_test(contrast = contrast, **kwargs)
                    score = result.cdf
                    row = contrast.sum(axis = 0).tolist()[0] + [score]
                    csv_writer.writerow(row)
            else:
                from joblib import Parallel, delayed
                num_cores = os.cpu_count()
                batch_size = 1000 * num_cores
                partition_size = 1000
                contrast_batch = []
                for i, contrast in tqdm(enumerate(contrasts), total = total_iter):
                    contrast_batch.append(contrast)
                    if len(contrast_batch) % batch_size == 0:
                        f_test_results = Parallel(n_jobs=num_cores)(delayed(self.partial_f_test)(c, **kwargs) for c in contrast_batch)
                        f_test_results = [x.cdf for x in f_test_results]
                        rows = [c.sum(axis = 0).tolist()[0] + [f] for c,f in zip(contrast_batch, f_test_results)]
                        csv_writer.writerows(rows)
                        del contrast_batch
                        del rows
                        del f_test_results
                        contrast_batch = []
                # process the remainder
                if len(contrast_batch):
                    f_test_results = Parallel(n_jobs=num_cores)(delayed(self.partial_f_test)(c, **kwargs) for c in contrast_batch)
                    f_test_results = [x.cdf for x in f_test_results]
                    rows = [c.sum(axis = 0).tolist()[0] + [f] for c,f in zip(contrast_batch, f_test_results)]
                    csv_writer.writerows(rows)
                    





        


    
            


