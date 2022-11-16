#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score


from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score

from sklearn.metrics import make_scorer

def scoring_novel(clf, X_train, y_train, X_test, y_test, scoring, novelty=True):
    
    if novelty==True:
        # in case novelty we need to consider only inliner for trainig
       
        X_train_in = X_train[(y_train==1)]
        y_train_in = y_train[(y_train==1)]
        
                
        clf.fit(X_train_in, y_train_in)
        y_hat = clf.predict(X_test)
                    
    else:
               
        y_hat = clf.fit_predict(X_test)
        
    if scoring == 'accuracy':
        
        score = accuracy_score(y_test, y_hat)
        
    elif scoring == 'f1':
        
        score = f1_score(y_test, y_hat, pos_label=-1)
    
    return score



def my_cv(clf, X, y, scoring='accuracy', clf_type=None, novelty=True, cv=5, test_size=None, random_state=42):
    
    sss = StratifiedShuffleSplit(n_splits=cv, test_size=test_size, random_state=random_state)
    score_list = []
    for train_index, test_index in sss.split(X, y):
         
        if novelty==True:
            # in case novelty we need to consider only inliner for trainig
            X_tr_sp = X[train_index]
            y_tr_sp = y[train_index]
            X_train_in_sp = X_tr_sp[(y_tr_sp==1)]
            y_train_in_sp = y_tr_sp[(y_tr_sp==1)]
            
            X_test_sp = X[test_index]
            y_test_sp = y[test_index]
            
            clf.fit(X_train_in_sp, y_train_in_sp)
            y_hat = clf.predict(X_test_sp)
                        
        else:
            
            X_test_sp = X[test_index]
            y_test_sp = y[test_index]
            
            y_hat = clf.fit_predict(X_test_sp)
        if scoring == 'accuracy':
            
            score_list.append(accuracy_score(y_test_sp, y_hat))
            
        elif scoring == 'f1':
            
            score_list.append(f1_score(y_test_sp, y_hat, pos_label=-1))
        
                    
    return np.array(score_list).mean()



def split_gen(X, y, spliter, novelty=True):
    
    spliter_gen = spliter.split(X, y)
    for idx_train, idx_test in spliter_gen:
        
        if novelty==True:
            
            idx_train_in = idx_train[y[idx_train]==1]
                        
            yield idx_train_in, idx_test
            
        else:
            
            yield idx_train, idx_test



def scorer_sfs(scoring):
    
    if scoring=='f1':
        f1_scorer = make_scorer(f1_score, pos_label=-1)
        
        return f1_scorer
    else:
        
        return scoring








