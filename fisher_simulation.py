# variables to consider within the script
# variance = 1, 2, 5, 10
# sample size for subsamples: 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000
# max_variables for importance score: 2, 5, 10, 20, all

#%% function that adds error to y
from scipy.stats import norm
import pickle 
from math import sqrt
def add_error(y, scale):
    length = len(y)
    return y + norm.rvs(scale = scale, size = length)


#%% import data
from importance_score import dask_parameter_generator
from dask import dataframe as dd

#%% subsample

def gen_subsample( sample_size , error_scale, ddf):
    total_sample_size = ddf.shape[0].compute()
    ddf_subsample = ddf.sample(frac = sample_size/total_sample_size).copy()
    if error_scale > 1:
        error_scale_diff = error_scale - 1
        ddf_subsample['y'] = ddf.y.map_partitions(add_error, error_scale_diff)
    return ddf_subsample

#%% dask parameter generator
import os
def single_simulation(sample_size, iter, error_scale, ddf, num_interactions):
    for i in range(iter):
        subsample = gen_subsample(sample_size, error_scale, ddf)
        gen = dask_parameter_generator(subsample)
        gen.gen_importance_score(max_variables = 7, filename= f"importance_score_numInteractions_{num_interactions}_iter_{i}_n_{sample_size}_error_{error_scale}.csv", dir = os.path.join(os.getcwd(), 'importance_score'))


#%% argparse

if __name__ == '__main__':
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description='Generating importance score tables')

    parser.add_argument('-i', '--num_interactions', required = True, type = int, help = "Whether to differ the sign of beta coefficients with a 50-50 chance")

    # Parse the command-line arguments
    args = parser.parse_args()

    # reading parquet file
    num_interactions = args.num_interactions
    parquet_file_name = f"simulation_with_{num_interactions}_interactions"

    # create parquet dataset
    subprocess.run(['python', 'simple_beta_sample_generator.py', '-f', f'{parquet_file_name}' ,'-p', '5', '-n', '64_000', '-e', '1', '-i', f'{num_interactions}'])
    parquet_file_name += '.parquet'
    # read the dataset
    ddf = dd.read_parquet(parquet_file_name)

    # reading config_file
    with open('sample_generator_config.pickle', 'rb') as f:
        config = pickle.load(f)

    
    with open(f"importance_score/sample_generator_config_i_{num_interactions}.pickle", 'wb') as f:
        pickle.dump(config, f)

    subsample_sizes = [500, 1_000, 5_000, 10_000]
    error_scales = [1, sqrt(2), sqrt(5), sqrt(10)]

    from itertools import product
    from tqdm import tqdm
    subsample_size_x_error_scales = list(product(subsample_sizes, error_scales))
    for sample_size, error_scale in tqdm(subsample_size_x_error_scales):
        single_simulation(sample_size = sample_size, iter = 5, error_scale = error_scale, ddf = ddf, num_interactions = num_interactions)