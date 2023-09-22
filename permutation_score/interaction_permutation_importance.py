import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from typing import Union
from itertools import combinations
from collections import defaultdict

def permutation_importance_together_new(X_test:pd.DataFrame, y_test: Union[pd.DataFrame, pd.Series, np.array, list], model, n_repeats = 30, n_jobs = -1, p = None, mae = False):
    if mae:
        benchmark = mean_absolute_error(y_test, model.predict(X_test))
    else:
        benchmark = mean_squared_error(y_test, model.predict(X_test))
    if p:
        assert p <= X_test.shape[1]
        assert p >= 2
    else:
        p = X_test.shape[1]
    
    mse_diff_dict = defaultdict(list)
    mse = defaultdict(list)
    variable_names = []
    for idx in range(n_repeats):
        X_replace = X_test.sample(X_test.shape[0]).reset_index(drop = True)
        for k in range(p):
            k += 1
            if k == 1:
                for col in X_test.columns:
                    X_perm = X_test.copy()
                    X_perm[col] = X_replace[col]
                    y_pred = model.predict(X_perm)
                    if mae:
                        perm_score = mean_absolute_error(y_test, y_pred)
                    else:
                        perm_score = mean_squared_error(y_test, y_pred)
                    mse_diff_dict[col].append(perm_score- benchmark)
                    mse[col].append(perm_score)
                    if idx == 0:
                        variable_names.append(col)
            else:
                for var in combinations(X_test.columns, k):
                    var = list(var)
                    colname = "*".join(var)
                    if idx == 0:
                        variable_names.append(colname)
                    X_perm = X_test.copy()
                    for v in var:
                        X_perm[v] = X_replace[v]
                    y_pred = model.predict(X_perm)
                    if mae:
                        perm_score = mean_absolute_error(y_test, y_pred)
                    else:
                        perm_score = mean_squared_error(y_test, y_pred)
                    mse_diff_dict[colname].append(perm_score- benchmark)
                    mse[colname].append(perm_score)
    
    importances = np.array([x for x in mse_diff_dict.values()])
    mean_squared_errors =  np.array([x for x in mse.values()])
    importances_mean = np.array([np.mean(x) for x in mse_diff_dict.values()])
    importances_std = np.array([np.std(x) for x in mse_diff_dict.values()])
    importances_se = np.sqrt(importances_std**2/n_repeats)
    importances_me = 1.96*importances_se
    variable_names = np.array(variable_names)
    result = {"importances_mean": importances_mean,
              "importances_std":  importances_std,
              "importances":      importances,
              'mean_squared_errors': mean_squared_errors,
              "variable_names":   variable_names,
              "importances_se":   importances_se,
              "importances_me":   importances_me}
    return result





def permutation_importance_together(X_test:pd.DataFrame, y_test: Union[pd.DataFrame, pd.Series, np.array, list], model, n_repeats = 30, n_jobs = -1, p = None):
    benchmark = mean_squared_error(y_test, model.predict(X_test))
    permutation_score_result = permutation_importance(model, X_test, y_test, n_repeats = n_repeats, n_jobs = n_jobs, scoring = 'neg_mean_squared_error')
    permutation_score_result['variable_names'] = X_test.columns.to_numpy()
    if p:
        assert p <= X_test.shape[1]
        assert p >= 2
    else:
        p = X_test.shape[1]
    for k in range(2, p+1):
        for var in combinations(X_test.columns, k):
            var = list(var)
            mse_diff = []
            for _ in range(n_repeats):
                X_perm = X_test.copy()
                X_replace = X_perm[var].sample(X_perm.shape[0]).reset_index(drop = True)
                for j in range(k):
                    X_perm[var[j]] = X_replace[var[j]]
                y_pred = model.predict(X_perm)
                mse_diff.append(mean_squared_error(y_test, y_pred)- benchmark)    # it's neg_mean_squared_error
            permutation_score_result['variable_names'] = np.append(permutation_score_result['variable_names'], "*".join(var))
            permutation_score_result['importances_mean'] = np.append(permutation_score_result['importances_mean'], [np.mean(mse_diff)])
            permutation_score_result['importances_std'] = np.append(permutation_score_result['importances_std'], [np.std(mse_diff)])
            permutation_score_result['importances'] = np.append(permutation_score_result['importances'], [mse_diff], axis = 0)
    permutation_score_result['importances_ste'] = permutation_score_result['importances_std']/np.sqrt(n_repeats)
    permutation_score_result['importances_me'] = 1.96 * permutation_score_result['importances_ste']
    return permutation_score_result


def permutation_importance_separate(X_test:pd.DataFrame, y_test: Union[pd.DataFrame, pd.Series, np.array, list], model, n_repeats = 30, n_jobs = -1, p = None):
    benchmark = mean_squared_error(y_test, model.predict(X_test))
    permutation_score_result = permutation_importance(model, X_test, y_test, n_repeats = n_repeats, n_jobs = n_jobs, scoring = 'neg_mean_squared_error')
    permutation_score_result['variable_names'] = X_test.columns.to_numpy()
    if p:
        assert p <= X_test.shape[1]
        assert p >= 2
    else:
        p = X_test.shape[1]
    for k in range(2, p+1):
        for var in combinations(X_test.columns, k):
            var = list(var)
            mse_diff = []
            for _ in range(n_repeats):
                X_perm = X_test.copy()
                for j in range(k):
                    X_perm[var[j]] = X_perm[var[j]].sample(X_perm.shape[0]).reset_index(drop = True)
                y_pred = model.predict(X_perm)
                mse_diff.append(mean_squared_error(y_test, y_pred)- benchmark)    # it's neg_mean_squared_error
            permutation_score_result['variable_names'] = np.append(permutation_score_result['variable_names'], "*".join(var))
            permutation_score_result['importances_mean'] = np.append(permutation_score_result['importances_mean'], [np.mean(mse_diff)])
            permutation_score_result['importances_std'] = np.append(permutation_score_result['importances_std'], [np.std(mse_diff)])
            permutation_score_result['importances'] = np.append(permutation_score_result['importances'], [mse_diff], axis = 0)
    permutation_score_result['importances_ste'] = permutation_score_result['importances_std']/np.sqrt(n_repeats)
    permutation_score_result['importances_me'] = 1.96 * permutation_score_result['importances_ste']
    return permutation_score_result

    
    
    
    
    