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
from csv import DictWriter
import os

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
        for _ in range(self.sample_size):
            yield bernoulli.rvs(list(self.pi.values()), size = self.var_num)
        

    def get_y(self, x):
        main_effect_variables = (str(x) for x in self.main_effect_variables)
        main_effect_coefficients = self.main_coefficients
        interaction_effect_coefficients = self.interaction_effect_coefficients
        main_effect = sum([coef*variable for coef, variable in zip(main_effect_coefficients.values(), np.concatenate([[1], x]))])
        
        x_row = pd.Series(x, index = main_effect_variables)
        interaction_effect = sum([x_row[str(key).split('*')].prod() * val for key, val in self.interaction_effect_coefficients.items()])
        error = norm.rvs(scale = self.error_scale)
        y = main_effect + interaction_effect + error
        return y
    
    def generator(self):
        x_generator = self.X_generator()
        variables = self.main_effect_variables + [Symbol('y')]
        for array_x in x_generator:
            y = self.get_y(array_x)
            yield dict(zip(variables, array_x.tolist() + [y]))


    @property
    def fieldnames(self):
        if hasattr(self, '_fieldnames'):
            pass
        else:
            self._fieldnames = self.main_effect_variables + [Symbol('y')]
        return self._fieldnames
    
    
    @property
    def L(self):
        if hasattr(self, '_L'):
            pass
        else:
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
            all_sets = list(set(product([0,1], repeat = self.var_num))); all_sets.sort()
            self._L = np.array([barcode_to_beta(x) for x in all_sets]).T
        return self._L.T
    
    @property
    def L_inv(self):
        if hasattr(self, '_L_inv'):
            pass
        else:
            self._L_inv = inv(self.L).astype(np.int8)
        return self._L_inv
    
    def save_file(self):
        with NamedTemporaryFile('w', suffix = '.csv', dir = os.getcwd(), delete = False) as csv_file:
            filename = csv_file.name
            generator = self.generator()
            writer = DictWriter(csv_file, fieldnames = [str(x) for x in self.fieldnames])
            writer.writeheader()
            for row in generator:
                row = {str(key):val for key, val in row.items()}
                writer.writerow(row)
        return filename
    
    def __call__(self):
        self.save_file()




def main(**kwargs):
    if 'filename' in kwargs.keys():
        with open( kwargs['filename'], 'w') as csv.file:
            del kwargs['filename']
            generator = sample_generator(**kwargs)
            writer = DictWriter(csv_file, fieldnames = [str(x) for x in generator.fieldnames])
            writer.writeheader()
            for row in generator:
                row = {str(key):val for key, val in row.items()}
                writer.writerow(row)
        return filename
    
    else:
        with NamedTemporaryFile('w', suffix = '.csv', dir = os.getcwd(), delete = False) as csv_file:
            filename = csv_file.name
            generator = sample_generator(**kwargs)
            writer = DictWriter(csv_file, fieldnames = [str(x) for x in generator.fieldnames])
            writer.writeheader()
            for row in generator:
                row = {str(key):val for key, val in row.items()}
                writer.writerow(row)
        return filename


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Creating a csv file for barcode study')

    parser.add_argument('-f', '--file', type = str, required = False, default = None, help='Path to the input file')
    parser.add_argument('-p', '--num_var', type=int, required = False, default = 3, help='A number of binary explanatory variables')
    parser.add_argument('-n','--sample_size', type = int, required = False, default = 100, help = 'Total number of rows of the dataset')
    parser.add_argument('-e','--scale', required = False, default = None, help = "The scale parameter of the linear model" )
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    kwargs = {}
    if args.file:
        kwargs['filename'] = args.file
    kwargs['var_num'] = args.num_var
    kwargs['sample_size'] = args.sample_size
    if args.scale:
        kwargs['scale'] = args.scale

    # run main function
    filename = main(**kwargs)
    print(filename)