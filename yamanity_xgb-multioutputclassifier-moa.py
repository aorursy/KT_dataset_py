import numpy as np

import pandas as pd



# import the other libraries

import optuna # for hyper parameter tuning

from xgboost import XGBClassifier as cls



#from sklearn.ensemble import ExtraTreesClassifier as cls

#from sklearn.ensemble import RandomForestClassifier as cls

from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split

from sklearn import datasets

from sklearn.metrics import make_scorer, log_loss



from functools import partial
# load data

df_X_train = pd.read_csv('../input/lish-moa/train_features.csv')

df_y_train = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

df_X_test = pd.read_csv('../input/lish-moa/test_features.csv')



df_id_train = df_X_train['sig_id']

df_id_test = df_X_test['sig_id']



del df_X_train['sig_id']

del df_y_train['sig_id']

del df_X_test['sig_id']
df_X_train = pd.get_dummies(df_X_train)

df_X_test = pd.get_dummies(df_X_test)
model =  MultiOutputClassifier(cls(tree_method='gpu_hist'))

model.fit(df_X_train, df_y_train)
y_pred = model.predict(df_X_test)
df_y_pred = pd.DataFrame(y_pred)

df_y_pred.columns = df_y_train.columns
df_y_pred.describe()
df_submit = pd.concat([df_id_test, df_y_pred], axis=1)
df_submit.to_csv('submission.csv', index=False)