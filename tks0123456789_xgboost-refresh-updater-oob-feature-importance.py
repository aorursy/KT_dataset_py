# Import all the necessary packages 

import kagglegym

import numpy as np

import pandas as pd

import time

import xgboost as xgb

import matplotlib.pyplot as plt



%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Read the full data set stored as HDF5 file

df = pd.read_hdf('../input/train.h5')
t0 = time.time()

excl = ['id', 'sample', 'y', 'timestamp']

col = [c for c in df.columns if c not in excl]



df_train = df[df.timestamp <= 905][col]

d_mean= df_train.median(axis=0)



df_all = df[col]



X_train = df_all[df.timestamp <= 905].values

y_train = df.y[df.timestamp <= 905].values

X_valid = df_all[df.timestamp > 905].values

y_valid = df.y[df.timestamp > 905].values

feature_names = df_all.columns

del df_all, df_train, df

print("Done: %.1fs" % (time.time() - t0))
X_train.shape
xgmat_train = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)

xgmat_valid = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_names)
params_xgb = {'objective':'reg:linear',

              'eta'             : 0.1,

              'max_depth'       : 4,

              'subsample'       : 0.9,

              #'colsample_bytree':0.8,

              'min_child_weight': 1000,

              'base_score':0

              }
print ("Training")

t0 = time.time()

bst = xgb.train(params_xgb, xgmat_train, 10)

print("Done: %.1fs" % (time.time() - t0))
params_xgb.update({'process_type': 'update',

                   'updater'     : 'refresh',

                   'refresh_leaf': False})
t0 = time.time()

print("Refreshing")

bst_after = xgb.train(params_xgb, xgmat_valid, 10, xgb_model=bst)

print("Done: %.1fs" % (time.time() - t0))
# Before refresh

for line in bst.get_dump(with_stats=True)[0].splitlines()[:10]:

    print(line)
# After refresh

for line in bst_after.get_dump(with_stats=True)[0].splitlines()[:10]:

    print(line)
imp = pd.DataFrame(index=feature_names)

imp['train'] = pd.Series(bst.get_score(importance_type='gain'), index=feature_names)



# OOB feature importance

imp['OOB'] = pd.Series(bst_after.get_score(importance_type='gain'), index=feature_names)

imp = imp.fillna(0)
ax = imp.sort_values('train').tail(10).plot.barh(title='Feature importances sorted by train', figsize=(7,4))
ax = imp.sort_values('OOB').tail(10).plot.barh(title='Feature importances sorted by OOB', xlim=(0,0.07), figsize=(7,4))