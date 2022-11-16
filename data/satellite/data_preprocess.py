#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 20:30:00 2022

@author: aleksei
"""



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import scipy.io



############ Data ###############




# path = '/home/aleksei/Work/Publications/IDEAL_2021/computations/Syntethic data/data_uniform_5_1/'

# X_train = np.loadtxt(path + 'X_train.txt')
# X_test = np.loadtxt(path + 'X_test.txt')

# y_train = np.loadtxt(path + 'y_train.txt')
# y_test = np.loadtxt(path + 'y_test.txt')

path = '/home/aleksei/Work/Publications/IDEAL_journal/Computations/syne-tune/data/satellite/'

data = scipy.io.loadmat(path + 'satellite.mat')

X = np.array(data['X'].tolist())
y = np.array(data['y'].squeeze().tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ################ end data ###################




# ####### prepare data for model test #####

# need to change lable from 0/1 to 1/-1

# y_tt = np.array(y_train.tolist())

y_train[y_train==1] = -1
y_train[y_train==0] = 1

y_test[y_test==1] = -1
y_test[y_test==0] = 1





# split train -> rain, validation
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
# need to change train data for only inliner(0) and test(and probably validation data) data should be mixed 


X_train_in = X_train[(y_train==1)]
y_train_in = y_train[(y_train==1)]
# X_train_in = X_train.loc[(y_train==1).squeeze()]
# y_train_in = y_train.loc[(y_train==1).squeeze()].squeeze()


save_path = path #'/home/aleksei/Work/Publications/IDEAL_journal/Computations/syne-tune/data/data_uniform_5_1/'

np.savetxt(save_path + 'X_val.txt', X_val)
np.savetxt(save_path + 'y_val.txt', y_val)
np.savetxt(save_path + 'X_train_in.txt', X_train_in)
np.savetxt(save_path + 'y_train_in.txt', y_train_in)

np.savetxt(save_path + 'X_test.txt',X_test)
np.savetxt(save_path + 'y_test.txt',y_test)



# X_val.np.savetxt('X_val.txt')
# y_val.np.savetxt('y_val.txt')
# X_train_in.np.savetxt('X_train_in.txt')
# y_train_in.np.savetxt('y_train_in.txt')

# X_test.np.savetxt('X_test.txt')
# y_test.np.savetxt('y_test.txt')







