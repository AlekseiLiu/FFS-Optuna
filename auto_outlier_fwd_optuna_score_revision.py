#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

file_path = os.path.dirname(os.path.realpath(__file__)) + '/'

print(file_path)


import numpy as np

from argparse import ArgumentParser

from utils.bfe import my_cv, scoring_novel
from utils.bfe import scorer_sfs, split_gen
from hpo.functions import clf_space, choose_sampler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import StratifiedShuffleSplit

import optuna

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ColumnSelector

from sklearn.feature_selection import mutual_info_classif

from datetime import datetime
from pathlib import Path



parser = ArgumentParser()

parser.add_argument('-d', '--data_name', type=str,
                    help='specify the name of data, it will be used to download data and to store the data in specific results folder', default='satimage-2')

parser.add_argument('-t', '--time', nargs='?', const=1, type=int, default=1800)

parser.add_argument('-n', '--n_trials', nargs='?', const=1, type=int, default=10)

parser.add_argument('-cv', '--cross_validation', nargs='?', const=1, type=int, default=3)

parser.add_argument('-fx', '--features_explore', help='# of featires to explore (during FFS)', nargs='?', const=1, type=int, default=8)

parser.add_argument('-s', '--scoring_function', type=str,
                    help='scoring function. could be [accuracy, f1]', default='f1')

parser.add_argument('-sm', '--sampler', type=str,
                    help='sampler choice. could be [nsga2, motpe, cma_es, tpe, skopt]', default='nsga2')

parser.add_argument('-nv', '--novelty', type=bool,
                    help='novelty setting [True, False]', default=True)




args, _ = parser.parse_known_args()


data_name = args.data_name

data_path = file_path + 'data/' + data_name + '/'


NUM_TRIALS =args.n_trials

MAX_WALL_TIME =args.time

CV = args.cross_validation

FEAT_X = args.features_explore

SCORE = args.scoring_function

NOVELTY = args.novelty

SAMPLER = args.sampler

RS = 1985


args_show = parser.parse_args()
print('arguments are the following:')
for key, value in vars(args_show).items():
    print(f'{key} = {value}')
    
arg_list = [(key, value) for key, value in vars(args_show).items()]

arg_list.append(('random_seed', RS))

# load data
X_train = np.loadtxt(data_path +'X_train.txt')
y_train = np.loadtxt(data_path + 'y_train.txt')
X_test = np.loadtxt(data_path + 'X_test.txt')
y_test = np.loadtxt(data_path + 'y_test.txt')

X_temp_tr = np.copy(X_train)
X_temp_ts = np.copy(X_test)




""""objective for outlier detection detection(novelty=False)"""
def objective(trial):
    
    clf = clf_space(trial, novelty=NOVELTY, random_seed=RS)
    
    print(X_temp_tr.shape)
    loss = my_cv(clf, X_temp_tr, y_train, cv=CV, scoring=SCORE, novelty=NOVELTY, test_size=None, random_state=RS)
    
    return loss



# init scorer
scorer = scorer_sfs(SCORE)

# init tracker for results
track = [[] for _ in range(14)]

# init first feature by means of MI
mi = mutual_info_classif(X_train, y_train, discrete_features='auto', n_neighbors=3, copy=True, random_state=RS)

bst_init_feat = int(mi.argmax())


# select best feature
col_selector = ColumnSelector(cols=[bst_init_feat])
X_temp_tr = col_selector.transform(X_train)
X_temp_ts = col_selector.transform(X_test)

# evaluate autoML
sampler = choose_sampler(SAMPLER, random_seed=RS)

study = optuna.create_study(direction="maximize", sampler=sampler)


study.optimize(objective, timeout=MAX_WALL_TIME, n_trials=NUM_TRIALS, gc_after_trial=True)


# best obj
best_obj = study.best_trial.values[0]
# best clf
clf_best = clf_space(study.best_trial, novelty=NOVELTY, random_seed=RS)


score = scoring_novel(clf_best, X_temp_tr, y_train, X_temp_ts, y_test, scoring=SCORE, novelty=NOVELTY)

feature_track = tuple([bst_init_feat])
feat_track = {1 : feature_track}

# update tracker
track[4].append(feature_track)
track[3].append(str(clf_best))
track[2].append(score)
track[0].append(X_temp_tr.shape[1])
track[1].append('nan')
track[13].append(best_obj)


print(track)

spliter = StratifiedShuffleSplit(n_splits=CV, test_size=None, random_state=RS)

for i in range(2, min(FEAT_X, X_train.shape[1]+1)):

    scor_ch_sfs = scorer_sfs(SCORE)
    split_gen_sfs = split_gen(X_train, y_train, spliter, novelty=NOVELTY)

    iter_sfs = []
    for tr, ts in split_gen_sfs:
        
        iter_sfs.append([tr, ts])




    sfs = SFS(clf_best, 
              k_features=i, 
              forward=True, 
              floating=False, 
              verbose=2,
              scoring=scor_ch_sfs,
              cv=iter_sfs,
              fixed_features=feature_track,
              n_jobs=-1)


    sfs.fit(X_train, y_train)
    feature_track = sfs.k_feature_idx_ 
    
    # transform subsets with new feature
    X_temp_tr = sfs.transform(X_train)
    X_temp_ts = sfs.transform(X_test)
    
    # AutoML 
    
    sampler = choose_sampler(SAMPLER, random_seed=RS)

    study = optuna.create_study(direction="maximize", sampler=sampler)#, error_score='raise')

    study.optimize(objective, timeout=MAX_WALL_TIME, n_trials=NUM_TRIALS, gc_after_trial=True)# n_jobs=-1,
    # print(' ')
    # print(study.best_trial)

    # best obj
    best_obj = study.best_trial.values[0]
    # best clf
    clf_best = clf_space(study.best_trial, novelty=NOVELTY, random_seed=RS)
    
    score = scoring_novel(clf_best, X_temp_tr, y_train, X_temp_ts, y_test, scoring=SCORE, novelty=NOVELTY)
    
    d_feat = {i : feature_track}
    feat_track.update(d_feat)
    track[0].append(X_temp_tr.shape[1])
    track[1].append(i)
    track[2].append(score)
    track[3].append(str(clf_best))
    track[4].append(feature_track)
    track[13].append(best_obj)
    
    print(track)
    
    

""" perform strong baseline """

print('*************** Start evaluating strong baseline ***********************')

# X_temp for objective avaluation as full train data
X_temp_tr = np.copy(X_train)

# AutoML
sampler = choose_sampler(SAMPLER, random_seed=RS)

study = optuna.create_study(direction="maximize", sampler=sampler)

study.optimize(objective, timeout=MAX_WALL_TIME, n_trials=NUM_TRIALS, gc_after_trial=True)


# best clf
clf_best = clf_space(study.best_trial, novelty=NOVELTY, random_seed=RS)


# init by mutual information
mi = mutual_info_classif(X_train, y_train, discrete_features='auto', n_neighbors=3, copy=True, random_state=RS)

bst_init_feat = int(mi.argmax())

col_selector = ColumnSelector(cols=[bst_init_feat])
X_temp_tr = col_selector.transform(X_train)
X_temp_ts = col_selector.transform(X_test)

score = scoring_novel(clf_best, X_temp_tr, y_train, X_temp_ts, y_test, scoring=SCORE, novelty=NOVELTY)

feature_track = tuple([bst_init_feat])
feat_track = {1 : feature_track}

# update tracker
track[8].append(feature_track)
track[7].append(score)
track[5].append(X_temp_tr.shape[1])
track[6].append('nan')

print(track)

spliter = StratifiedShuffleSplit(n_splits=CV, test_size=None, random_state=RS)

for i in range(2, min(FEAT_X, X_train.shape[1]+1)):

    scor_ch_sfs = scorer_sfs(SCORE)
    split_gen_sfs = split_gen(X_train, y_train, spliter, novelty=NOVELTY)

    iter_sfs = []
    for tr, ts in split_gen_sfs:
        
        iter_sfs.append([tr, ts])

    sfs = SFS(clf_best, 
              k_features=i, 
              forward=True, 
              floating=False, 
              verbose=2,
              scoring=scor_ch_sfs,
              cv=iter_sfs,
              fixed_features=feature_track,
              n_jobs=-1)


    sfs.fit(X_train, y_train)
    feature_track = sfs.k_feature_idx_ 
    
    # transform subsets with new feature
    X_temp_tr = sfs.transform(X_train)
    X_temp_ts = sfs.transform(X_test)
    
    score = scoring_novel(clf_best, X_temp_tr, y_train, X_temp_ts, y_test, scoring=SCORE, novelty=NOVELTY)
    
    d_feat = {i : feature_track}
    feat_track.update(d_feat)
    track[5].append(0)
    track[6].append(i)
    track[7].append(score)
    track[8].append(feature_track)
    
    print(track)
    
    






""" weak baseline """


print('*************** Start evaluating weak baseline ***********************')


X_train = np.loadtxt(data_path +'X_train.txt')
y_train = np.loadtxt(data_path + 'y_train.txt')
X_test = np.loadtxt(data_path + 'X_test.txt')
y_test = np.loadtxt(data_path + 'y_test.txt')


mi = mutual_info_classif(X_train, y_train, discrete_features='auto', n_neighbors=3, copy=True, random_state=RS)

bst_init_feat = int(mi.argmax())

col_selector = ColumnSelector(cols=[bst_init_feat])
X_temp_tr = col_selector.transform(X_train)
X_temp_ts = col_selector.transform(X_test)

clf_best = LocalOutlierFactor(novelty=NOVELTY)

score = scoring_novel(clf_best, X_temp_tr, y_train, X_temp_ts, y_test, scoring=SCORE, novelty=NOVELTY)

feature_track = tuple([bst_init_feat])
feat_track = {1 : feature_track}

# update tracker
track[12].append(feature_track)
track[11].append(score)
track[9].append(X_temp_tr.shape[1])
track[10].append('nan')

print(track)

spliter = StratifiedShuffleSplit(n_splits=CV, test_size=None, random_state=RS)

for i in range(2, min(FEAT_X, X_train.shape[1]+1)):

    scor_ch_sfs = scorer_sfs(SCORE)
    split_gen_sfs = split_gen(X_train, y_train, spliter, novelty=NOVELTY)

    iter_sfs = []
    for tr, ts in split_gen_sfs:
        
        iter_sfs.append([tr, ts])

    sfs = SFS(clf_best, 
              k_features=i, 
              forward=True, 
              floating=False, 
              verbose=2,
              scoring=scor_ch_sfs,
              cv=iter_sfs,
              fixed_features=feature_track,
              n_jobs=-1)


    sfs.fit(X_train, y_train)
    feature_track = sfs.k_feature_idx_ 
    
    # transform subsets with new feature
    X_temp_tr = sfs.transform(X_train)
    X_temp_ts = sfs.transform(X_test)
    
    score = scoring_novel(clf_best, X_temp_tr, y_train, X_temp_ts, y_test, scoring=SCORE, novelty=NOVELTY)
    
    d_feat = {i : feature_track}
    feat_track.update(d_feat)
    track[9].append(0)
    track[10].append(i)
    track[11].append(score)
    track[12].append(feature_track)
        
    print(track)
    





# from datetime import datetime
# from pathlib import Path
# fix the time for maling folder
curr_time = datetime.today().strftime('%Y_%m_%d_%H%M%S')

# create the folder for results if no exists
save_path = file_path +'results/'
Path(save_path).mkdir(parents=True, exist_ok=True)
# create the folder for specific data results if not exists
save_path_data = save_path + '/' + data_name
Path(save_path_data).mkdir(parents=True, exist_ok=True)
# create the folder for specific results using time to distinguish
time_path = save_path_data + "/" + curr_time
Path(time_path).mkdir(parents=True, exist_ok=True)

# save the file in the folder
with open(time_path + '/' + 'FFS_track_auto.txt', 'w') as f:
    f.write(repr(track))
with open(time_path + '/' + 'setup.txt', 'w') as f:
    f.write(repr(arg_list))




    


########################## plot results #################

#Ceck if last features really able to descriminate
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator


# features to display on horizontal axis
n_features = FEAT_X-1
    
f, ax1 = plt.subplots(1, 1, figsize=(10, 5.5))#, sharey=True)


x_left = [i+1 for i in range(n_features)]

ax1.plot(x_left, track[2][:n_features], 'r.-', label='outlier det auto')
ax1.plot(x_left, track[7][:n_features], c='k', label='outlier det str base')
ax1.plot(x_left, track[11][:n_features], 'b:', label='outlier det weak base')
ax1.xaxis.set_major_locator(MaxNLocator(nbins = 20, integer=True))
ax1.set_title('pareto front for '+ data_name + ' data, with FFS')
ax1.set_xlabel('sparsity [# features]')
ax1.set_ylabel(SCORE + ' score') #'f_1(outliers considered as positive)')
ax1.legend(fontsize=8)
# right plot


f.tight_layout()
path_to_save = time_path +'/' + 'result_plot'
# f.savefig(fname=path_to_save + '.eps', format='eps', dpi=1200, 
#             facecolor='w', edgecolor='w', orientation='portrait', 
#             papertype=None, transparent=False, bbox_inches=None, 
#             pad_inches=0.1, metadata=None)

f.savefig(fname=path_to_save + '.png')    


