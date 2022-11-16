#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

import optuna


""""objective for outlier detection detection"""


def clf_space(trial, novelty, random_seed=42):
    
    classifier_name = trial.suggest_categorical("classifier", ["one_svm", "lof", 'iof'])
    if classifier_name == "one_svm":
               
        kernel = trial.suggest_categorical('kernel', ['rbf', 'linear','sigmoid', 'poly'])
        
        tol = trial.suggest_float('tol', 0.00001, 0.05, log=True)
        nu = trial.suggest_float('nu', 0.002, 0.55, log=True)
        gamma = trial.suggest_float('gamma', 0.0001, 1, log=True)
        if kernel == 'poly':
            degree = 2
            
            clf = OneClassSVM(kernel=kernel, tol=tol, nu=nu, gamma=gamma, degree=degree)
        else:
            clf = OneClassSVM(kernel=kernel, tol=tol, nu=nu, gamma=gamma)
    
    elif classifier_name == "lof":
        
        n_neighbors = trial.suggest_int('n_neighbors', 2, 30)
        leaf_size = trial.suggest_int('leaf_size', 15, 60)
        metric = trial.suggest_categorical('metric', ['minkowski', 'euclidean', 'manhattan', 'chebyshev'])
        
        
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, leaf_size=leaf_size, p=2, 
                                              contamination='auto', novelty=novelty, metric=metric, n_jobs=-1)
        
    else:
        
        n_estimators = trial.suggest_int('n_estimators', 50, 400)
        max_samples = trial.suggest_float('max_samples', 0.01, 0.6, log=True)
        max_features = trial.suggest_float('max_features', 0.7, 1)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        
        clf = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, 
                              contamination='auto', max_features=max_features, n_jobs=-1, bootstrap=bootstrap, random_state=random_seed)
     
    return clf





def choose_sampler(sampler_coice, random_seed=42):
    
    if sampler_coice == 'nsga2':
        
        sampler = optuna.samplers.NSGAIISampler(population_size=80, seed=random_seed)
    
    elif sampler_coice == 'motpe':
        
        sampler = optuna.samplers.MOTPESampler()
        
    elif sampler_coice == 'cma_es':
        
        sampler = optuna.samplers.CmaEsSampler()
        
    elif sampler_coice == 'tpe':
                
        sampler = optuna.samplers.TPESampler(seed=random_seed)
        
    elif sampler_coice == 'skopt':
        
        sampler = optuna.integration.SkoptSampler()
        
    else:
        # default one
        sampler = optuna.samplers.TPESampler()
        
    return sampler


