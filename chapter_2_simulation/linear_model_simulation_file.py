from memory_profiler import profile
import sys
import os
import pandas as pd
import numpy as np
# sys.path.append('/root/barcode/')
sys.path.append('../')
from BarcodeScanner import tree_and_clustering, base_barcode
from itertools import product, combinations
from sklearn.linear_model import LinearRegression
import timeit
from datasets import Dataset
from itertools import product
from sklearn.linear_model import LinearRegression





def L(p):
    all_sets = list(set(product([0,1], repeat = p))); all_sets.sort()
    return np.array([base_barcode.barcode_to_beta(x) for x in all_sets]).astype(np.int8)


def gen_X(num_var: int, sample_size : int):
    data_dictionary = {}
    for i in range(num_var):
        var_name = "x" + f"{i + 1}"
        data_dictionary[var_name] = list(np.random.binomial(1, .5, sample_size))
    return pd.DataFrame(data_dictionary)

def gen_full_X(num_var: int, sample_size :int):
    raw_X = gen_X(num_var = num_var, sample_size = sample_size)
    colnames = raw_X.columns
    for k in range(2, len(colnames)+ 1):
        interaction_generator = combinations(colnames, k)
        for interaction_tuple in interaction_generator:
            new_colname = "*".join(interaction_tuple)
            raw_X[new_colname] = raw_X[list(interaction_tuple)].apply(np.prod, axis = 1)
    return raw_X

def gen_y(df):
    y = df.apply(lambda seq: 1 + seq.x1 + seq.x2 + seq.x1*seq.x3 + np.random.normal(), axis = 1).to_numpy().reshape(-1, 1)
    return y

@profile
def pipeline(p, n):
    df = pd.read_csv('sample_dataset_small.csv')
    X = df.loc[:, df.columns.str.contains('x')].to_numpy()
    y = df.y.to_numpy().reshape(-1,1)
    reg = LinearRegression()
    reg.fit(X, y)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="number of binary explanatory variables", type = int)
    parser.add_argument("-n", help= 'sample_size', type = int)
    
    args = parser.parse_args()

    
    pipeline(args.p, args.n)
    


