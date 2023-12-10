import sys
import os
cwd = os.path.dirname(__file__)
os.chdir(cwd)
sys.path.append(cwd)
from scipy.stats import norm
import pickle 
from math import sqrt
from importance_score import dask_parameter_generator
from dask import dataframe as dd
import argparse
import subprocess
from fisher_simulation import *
from itertools import product
from tqdm import tqdm
# interactions = [2, 5, 7]
subsample_sizes = [500, 1_000]
error_scales = [1, sqrt(2)]


def single_simulation(sample_size, iter, error_scale, ddf, num_interactions, start_number = None):
    for i in range(iter):
        subsample = gen_subsample(sample_size, error_scale, ddf)
        gen = dask_parameter_generator(subsample)
        if start_number:
            i += start_number
        gen.gen_importance_score(max_variables = 7, filename= f"importance_score_numInteractions_{num_interactions}_iter_{i}_n_{sample_size}_error_{error_scale}.csv", dir = os.path.join(os.getcwd(), 'importance_score'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating importance score tables')

    parser.add_argument('-i', '--num_interactions', required = True, type = int, help = "Whether to differ the sign of beta coefficients with a 50-50 chance")
    parser.add_argument('-s', '--start_number', required = True, type = int, help = "Whether to differ the sign of beta coefficients with a 50-50 chance")

    # Parse the command-line arguments
    args = parser.parse_args()

    interactions = [args.num_interactions]
    start_number = args.start_number
    subsample_size_x_error_scales = list(product(interactions, subsample_sizes, error_scales))
    for int_size, sample_size, error_scale in tqdm(subsample_size_x_error_scales):
        df = dd.read_parquet(f'simulation_with_{int_size}_interactions.parquet')
        single_simulation(sample_size, iter = 5, error_scale = error_scale, ddf = df, num_interactions = int_size, start_number = start_number)
