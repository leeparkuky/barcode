from bitstring import Bits
from dask import dataframe as dd
from typing import List, Union
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import diags, csc_matrix
from itertools import product, combinations
import math



class dask_parameter_generator():
    def __init__(self, dask_dataframe, output_feature_name = None):
        self.ddf = self.init_process_dataframe(dask_dataframe, output_feature_name)

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

    def init_process_dataframe(self, dask_dataframe, output_feature_name):
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
