from dataclasses import dataclass
from typing import  Union, Tuple
import numpy as np
from numpy.linalg import inv
from scipy.stats import bernoulli, norm
import pandas as pd
from itertools import combinations, product
from math import comb, sqrt
from sympy import Symbol
from tempfile import NamedTemporaryFile
from csv import DictWriter, writer
import os

import dask
from dask.diagnostics import ProgressBar


def symbol_prod(*args):
    output = args[0]*args[1]
    for i, arg in enumerate(args):
        if i <= 1:
            pass
        else:
            output *= arg
    return output

@dataclass
class sample_generator:
    var_num: int
    sample_size: int
    num_interactions: int = None
    rng: np.random._generator.Generator = np.random.default_rng()
    beta_range: Tuple[Union[float]] = (3, 8)
    pi_range : Tuple[Union[float]] = (.2, .8)
    error_scale : float = sqrt(5)

    def __iter__(self):
        return self.generator()
    



    def gen_main_effects(self):
        self._main_effects_variabls = [Symbol(f'x_{i+1}') for i in range(self.var_num)]
        self._main_beta = [Symbol(f'beta_{i+1}') for i in range(self.var_num)]
    
    @property
    def main_effect_coefficients(self):
        if hasattr(self, '_main_beta'):
            pass
        else:
            self.gen_main_effects()
        return self._main_beta
    
    @property
    def main_effect_variables(self):
        if hasattr(self,'_main_effects_variables'):
            pass
        else:
            self.gen_main_effects()
        return self._main_effects_variabls
    
    @property
    def main_effect_variables_str(self):
        if hasattr(self,'_main_effects_variables'):
            pass
        else:
            self.gen_main_effects()
            self._main_effects_variables_str = [str(x) for x in self.main_effect_variables]
        return self._main_effects_variables_str
    
    
    def interaction_variables_generator(self):
        i = 1
        for k in range(2, self.var_num + 1):
            for idx, var_name in zip(range(i, i+comb(self.var_num, k)), combinations(self.main_effect_variables, k)):
                yield idx, symbol_prod(*var_name)
            i += comb(self.var_num, k)

    def gen_interactions(self):
        main_variables = self.main_effect_variables
        num_interactions_total = 2**self.var_num - 1 - self.var_num
        # setting the number of interactions
        if self.num_interactions:
            if (self.num_interactions >= 0) and (self.num_interactions <= num_interactions_total):
                pass
            else:
                self.num_interactions = int(self.rng.uniform(1, num_interactions_total))
        else:
            self.num_interactions = int(self.rng.uniform(1, num_interactions_total))

        chosen_index = np.sort(self.rng.choice(range(1, num_interactions_total+1), self.num_interactions, replace = False))
        chosen_interactions = [var_name for idx, var_name in self.interaction_variables_generator() if idx in chosen_index.tolist()]
        chosen_index = [1 for _ in range(self.var_num + 1)] + \
        [1 if x in chosen_index.tolist() else 0 for x in range(1, num_interactions_total+ 1)]
        self.interactions_coef = [x if self.rng.random() < .5 else -x for x \
                                    in self.rng.uniform(low=self.beta_range[0], high=self.beta_range[1], size=self.num_interactions)]
        self.beta_effective = chosen_index
        self._interactions = dict(zip(chosen_interactions, self.interactions_coef))
    
    @property
    def interaction_effect_coefficients(self):
        if hasattr(self, '_interactions'):
            pass
        else:
            self.gen_interactions()
        return self._interactions
    
    @property
    def main_coefficients(self):
        if hasattr(self, '_main_coefficients'):
            pass
        else:
            self._main_coefficients = [x if self.rng.random() < .5 else -x for x in self.rng.uniform(low=self.beta_range[0], high=self.beta_range[1], size=self.var_num+1)]
        return {Symbol(f'beta_{i}'): v for i,v in enumerate(self._main_coefficients)}
    

    ## generating X and y
    @property
    def pi(self):
        if hasattr(self, '_pi'):
            pass
        else:
            self._pi = self.rng.uniform(low = self.pi_range[0], high = self.pi_range[1], size = self.var_num)
        return {f'pi_{i+1}': v for i,v in enumerate(self._pi)}
    
    
    def X_generator(self):
        for row in zip(*[bernoulli.rvs(pi, size = self.sample_size) for pi in self.pi.values()]):
            yield row

    def generator(self):
        X_generator = self.X_generator()
        main_effect_variables = self.main_effect_variables_str

        def find_interaction_effect(row, interactions = self.interaction_effect_coefficients):
            output = 0
            for key, val in interactions.items():
                key = str(key)
                output += row[key.split('*')].prod() * val
            return output


        for x, error in zip(X_generator, norm.rvs(scale = self.error_scale, size = self.sample_size)):
            main_variables = np.array([1]+list(x)); main_coefficients = np.array(list(self.main_coefficients.values()))
            main_effect = main_variables.dot(main_coefficients)
            x_row = pd.Series(x, index = main_effect_variables)
            interaction_effect = find_interaction_effect(x_row)
            y = main_effect + interaction_effect + error
            yield list(x) + [y]



    @property
    def fieldnames(self):
        if hasattr(self, '_fieldnames'):
            pass
        else:
            self._fieldnames = self.main_effect_variables + [Symbol('y')]
        return self._fieldnames
    

    def save_config(self, filename = 'sample_generator_config.pickle', dir = os.getcwd()):
        import pickle
        with open(os.path.join(dir, filename), 'wb') as f:
            pickle.dump(self.config, f)
    
    @property
    def config(self):
        if hasattr(self,'_config'):
            pass
        else:
            self.interaction_effect_coefficients
            self.main_coefficients
            config_variables = ['var_num','sample_size','num_interactions','error_scale','_main_ffects_variables','_main_beta','interactions_coef',
                                'beta_effective','_interactions','_main_coefficients']
            self._config = {k:v for k,v in self.__dict__.items() if k in config_variables}
        return self._config
    
    @classmethod
    def from_config(cls, config):
        original_variables = ['var_num','sample_size','num_interactions','error_scale']
        original_kwargs = {k:v for k,v in config.items() if k in original_variables}
        sg = cls(**original_kwargs)
        for key, val in config.items():
            if key not in original_variables:
                sg.__dict__[key] = val
        return sg

    
def gen_small_file(**kwargs):
    if 'filename' in kwargs.keys():
        with open( kwargs['filename'], 'w') as csv_file:
            del kwargs['filename']
            generator = sample_generator(**kwargs)
            csv_writer = writer(csv_file)
            fieldnames = [str(x) for x in generator.fieldnames]
            csv_writer.writerow(fieldnames)
            for row in generator:
                csv_writer.writerow(row)
        return filename
    
    else:
        with NamedTemporaryFile('w',prefix = 'temp_', suffix = '.csv', dir = os.getcwd(), delete = False) as csv_file:
            filename = csv_file.name
            generator = sample_generator(**kwargs)
            csv_writer = writer(csv_file)
            fieldnames = [str(x) for x in generator.fieldnames]
            csv_writer.writerow(fieldnames)
            for row in generator:
                csv_writer.writerow(row)
        return filename
    

def gen_small_file_with_config(config, **kwargs):
    if 'filename' in kwargs.keys():
        with open( kwargs['filename'], 'w') as csv_file:
            del kwargs['filename']
            generator = sample_generator.from_config(config)
            csv_writer = writer(csv_file)
            fieldnames = [str(x) for x in generator.fieldnames]
            csv_writer.writerow(fieldnames)
            for row in generator:
                csv_writer.writerow(row)
        return filename
    
    else:
        with NamedTemporaryFile('w',prefix = 'temp_', suffix = '.csv', dir = os.getcwd(), delete = False) as csv_file:
            filename = csv_file.name
            generator = sample_generator.from_config(config)
            csv_writer = writer(csv_file)
            fieldnames = [str(x) for x in generator.fieldnames]
            csv_writer.writerow(fieldnames)
            for row in generator:
                csv_writer.writerow(row)
        return filename


    
def gen_large_file(**kwargs):
    kwargs_copy = kwargs.copy()
    if 'filename' in kwargs.keys():
        del kwargs_copy['filename']
    main_sampler = sample_generator(**kwargs_copy)
    config = main_sampler.config # same coefficients and interaction effects will be used across sample generator
    main_sampler.save_config() # save config file
    num_cores = os.cpu_count()
    batch_size = kwargs['sample_size'] // num_cores
    remainder = kwargs['sample_size'] % num_cores
    from joblib import Parallel, delayed
    config['sample_size'] = batch_size

    filenames = Parallel(n_jobs=num_cores)(delayed(gen_small_file_with_config)(config, **kwargs) for _ in range(num_cores))
    if remainder:
        config['sample_size'] = remainder
        filenames.append(gen_small_file_with_config(config, **kwargs))

    from dask import dataframe as dd
    df = dd.read_csv("temp_*.csv")
    df.to_parquet('sample_data.parquet', engine='pyarrow', write_index = False)

    from glob import glob
    for fname in glob('temp_*.csv'):
        os.remove(fname)

    return os.path.join(os.getcwd(), 'sample_data.parquet')



def main(**kwargs):
    if kwargs['sample_size'] > 100:
        gen_large_file(**kwargs)
    else:
        gen_small_file(**kwargs)
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Creating a csv file for barcode study')

    parser.add_argument('-f', '--file', type = str, required = False, default = None, help='Path to the output file')
    parser.add_argument('-p', '--num_var', type=int, required = False, default = 3, help='A number of binary explanatory variables')
    parser.add_argument('-n','--sample_size', type = int, required = False, default = 100, help = 'Total number of rows of the dataset')
    parser.add_argument('-e','--scale', required = False, default = None, type = float, help = "The scale parameter of the linear model" )
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    kwargs = {}
    if args.file:
        kwargs['filename'] = args.file
    kwargs['var_num'] = args.num_var
    kwargs['sample_size'] = args.sample_size
    if args.scale:
        kwargs['error_scale'] = args.scale

    # run main function
    filename = main(**kwargs)
    print(filename)