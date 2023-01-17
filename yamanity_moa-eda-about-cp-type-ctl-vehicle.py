# import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df_X_train = pd.read_csv('../input/lish-moa/train_features.csv')

df_y_train = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

# I also used nonscored data

df_y_train_ext = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')



# 'sig_id' is not neccessary

del df_X_train['sig_id']

del df_y_train['sig_id']

del df_y_train_ext['sig_id']



df_train_all = pd.concat([df_X_train, df_y_train, df_y_train_ext], axis=1)
N_FEATURES_X = len(df_X_train.columns)

N_FEATURES_Y = len(df_y_train.columns)

N_FEATURES_Y_EXT = len(df_y_train_ext.columns)



N_FEATURES_X, N_FEATURES_Y, N_FEATURES_Y_EXT
# Extract the records where cp_type=="ctl_vehicle".

df_train_all_ctl_vehicle = df_train_all.query('cp_type=="ctl_vehicle"')



# Get row wise sum of train_targets_scored + train_targets_nonscored

df_train_all_ctl_vehicle.iloc[:, N_FEATURES_X:].apply(lambda row: row.sum(), axis=1).sum()