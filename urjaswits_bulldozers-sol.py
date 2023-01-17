# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#########imports

from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display

from sklearn import metrics

import sys

import feather
import os
print(os.listdir("../input"))

########
############################ read data

df_raw = pd.read_csv('../input/bluebook-for-bulldozers/TrainAndValid.csv', low_memory = False, parse_dates = ["saledate"])

######
########################### FUNCTIONS ##############################


####### display_all function

def display_all(df):
    with pd.option_context("display.max_rows", 1000,"display.max_columns", 1000):
        display(df)

        
        
####### split_vals function

def split_vals(a, n):
    return a[:n].copy(), a[n:].copy()



##### rmse function

def rmse(x, y):
    return math.sqrt(((x-y)**2).mean())



##### print_score function

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid), m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'):
        res.append(m.oob_score_)
    print(res)
    
#####
######### data pre-processing

df_raw.SalePrice = np.log(df_raw.SalePrice)

train_cats(df_raw)

add_datepart(df_raw, 'saledate')

df, y, nas = proc_df(df_raw, 'SalePrice')

#########
####### making cross-validtion set

n_valid = 12000
n_trn = len(df) - n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)

X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)


# X_train.shape, y_train.shape, X_valid.shape

raw_train.shape, raw_valid.shape, X_train.shape, X_valid.shape, y_train.shape, y_valid.shape

#######
###### model 3

m=RandomForestRegressor(n_jobs = -1)
%time m.fit(X_train, y_train)
print_score(m)

######