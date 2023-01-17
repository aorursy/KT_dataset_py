# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
# from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
#                               GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, 
                              GradientBoostingRegressor, ExtraTreesRegressor)
# from sklearn.svm import SVC
from sklearn.cross_validation import KFold

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# Some useful parameters which will come in handy later on
# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        rst = list(self.clf.fit(x,y).feature_importances_)
        return rst
# Class to extend XGboost classifer
# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 1000,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 10,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':1000,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 1000,
    'learning_rate' : 0.9
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 2000,
     #'max_features': 0.2,
    'max_depth': 10,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }
def great_train(x_train, y_train, x_test):
    rf = SklearnHelper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
    et = SklearnHelper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
    ada = SklearnHelper(clf=AdaBoostRegressor, seed=SEED, params=ada_params)
    gb = SklearnHelper(clf=GradientBoostingRegressor, seed=SEED, params=gb_params)
    print("Helper generated")
    rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test) # Random Forest
    et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
    ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
    gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
    print("Training complete")
    x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train), axis=1)
    x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test), axis=1)
    # x_train = np.concatenate(( rf_oof_train, gb_oof_train), axis=1)
    # x_test = np.concatenate(( rf_oof_test, gb_oof_test), axis=1)
    gbm = xgb.XGBRegressor().fit(x_train, y_train)
    return gbm.predict(x_test)
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((N_TRAIN,))
    oof_test = np.zeros((N_TEST,))
    oof_test_skf = np.empty((N_FOLDS, N_TEST))
    
    for i, (train_index, test_index) in enumerate(KF):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)
        
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
full = pd.read_csv('../input/train.csv')
full = full.drop('ID', axis='columns')
giba_features = ["f190486d6","58e2e02e6","eeb9cd3aa","9fd594eec","6eef030c1","15ace8c9f","fb0f5dbfe","58e056e12","20aa07010","024c577b9","d6bb78916","b43a7cfd5","58232a6fb"]
full = full[['target'] + giba_features]
y = np.array(full.target)
x = np.array(full.drop('target', axis='columns'))
# x_train, x_test, y_train, y_test = train_test_split(x, y)
# N_TRAIN = len(x_train)
# N_TEST = len(x_test)
s_test = pd.read_csv('../input/test.csv')
s_test_id = s_test.ID
s_test = s_test.drop('ID', axis='columns')
s_test = np.array(s_test[giba_features])
N_TRAIN = len(x)
N_TEST = len(s_test)
SEED = 0 # for reproducibility
N_FOLDS = 3 # set folds for out-of-fold prediction
KF = KFold(N_TRAIN, n_folds= N_FOLDS, random_state=SEED)
result = great_train(x, y, s_test)
submit = pd.DataFrame({'ID': s_test_id, 'target': result})
submit.target = submit.target.map(lambda x: abs(x))
submit.to_csv('result4399.csv', index=False)

