import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

from datetime import datetime

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn import metrics

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier



pd.set_option('display.max_columns', 200)
def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

train_df = pd.read_csv('../input/train.csv', engine='python')

test_df = pd.read_csv('../input/test.csv', engine='python')



#Experimenting with a small subset

train_df = train_df[1:10000]
import subprocess

print((subprocess.check_output("lscpu", shell=True).strip()).decode())
# A parameter grid for XGBoost

params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 5, 7, 10],

        'learning_rate': [0.01, 0.02, 0.05]    

        }
%%time 

folds = 3

param_comb = 1



target = 'target'

predictors = train_df.columns.values.tolist()[2:]



X = train_df[predictors]

Y = train_df[target]



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic', nthread=1)



random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001)



# Here we go

start_time = timer(None) # timing starts from this point for "start_time" variable

random_search.fit(X, Y)

timer(start_time) # timing ends here for "start_time" variable

!nvidia-smi
%%time 

folds = 3

param_comb = 1



target = 'target'

predictors = train_df.columns.values.tolist()[2:]



X = train_df[predictors]

Y = train_df[target]



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



xgb = XGBClassifier(learning_rate=0.02, n_estimators=1000, objective='binary:logistic',

                    silent=True, nthread=6, tree_method='gpu_hist', eval_metric='auc')



random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001 )



# Here we go

start_time = timer(None) # timing starts from this point for "start_time" variable

random_search.fit(X, Y)

timer(start_time) # timing ends here for "start_time" variable
%%time 



folds = 3

param_comb = 20



target = 'target'

predictors = train_df.columns.values.tolist()[2:]



X = train_df[predictors]

Y = train_df[target]



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



xgb = XGBClassifier(learning_rate=0.02, n_estimators=1000, objective='binary:logistic',

                    silent=True, nthread=6, tree_method='gpu_hist', eval_metric='auc')



random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001 )



# Here we go

start_time = timer(None) # timing starts from this point for "start_time" variable

random_search.fit(X, Y)

timer(start_time) # timing ends here for "start_time" variable
y_test = random_search.predict_proba(test_df[predictors])

y_test.shape

sub_df = pd.DataFrame({"ID_code": test_df.ID_code.values, "target": y_test[:,1]})

sub_df[:10]
sub_df.to_csv("xgboost_gpu_randomsearch.csv", index=False)