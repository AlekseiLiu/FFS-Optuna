#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import scipy.io



path = '/data/optdigits/'

data = scipy.io.loadmat(path + 'optdigits.mat')

X = np.array(data['X'].tolist())
y = np.array(data['y'].squeeze().tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


y_train[y_train==1] = -1
y_train[y_train==0] = 1

y_test[y_test==1] = -1
y_test[y_test==0] = 1





# split train -> rain, validation
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)



X_train_in = X_train[(y_train==1)]
y_train_in = y_train[(y_train==1)]



save_path = path 
np.savetxt(save_path + 'X_val.txt', X_val)
np.savetxt(save_path + 'y_val.txt', y_val)
np.savetxt(save_path + 'X_train_in.txt', X_train_in)
np.savetxt(save_path + 'y_train_in.txt', y_train_in)

np.savetxt(save_path + 'X_test.txt',X_test)
np.savetxt(save_path + 'y_test.txt',y_test)







