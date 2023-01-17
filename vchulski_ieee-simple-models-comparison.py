import pandas as pd

import numpy as np

import lightgbm as lgb

import xgboost as xgb

import catboost as cb

from catboost import CatBoostClassifier, Pool

import random 



import os

from os import listdir

from tqdm import tqdm

from os.path import isfile



import sklearn

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler 

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn import preprocessing

from sklearn.externals import joblib

from sklearn.decomposition import TruncatedSVD



from bayes_opt import BayesianOptimization

from bayes_opt.event import Events

from bayes_opt.util import load_logs



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set()



import time

import datetime



#import shap

# load JS visualization code to notebook

#shap.initjs()



import warnings

warnings.filterwarnings("ignore")



print(os.listdir("../input"))

print()



print("pandas:", pd.__version__)

print("numpy:", np.__version__)

print("sklearn:", sklearn.__version__)

print()

print("lightgbm:", lgb.__version__)

print("xgboost:", xgb.__version__)

print("catboost:", cb.__version__)
train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')



train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')



sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



print(f'Shape of train set: {train.shape}')

print(f'Shape of test set: {test.shape}')
sns.countplot(train['isFraud']) #Imbalanced Dataset

plt.title('Target distribution');
print(f'Number of fraud samples in train: {len(np.where(train["isFraud"]==1)[0])}')

print(f'Percent of fraud samples in train: {round(100.0*len(np.where(train["isFraud"]==1)[0])/len(train["isFraud"]),2)}')
train = train.sample(frac=0.1, random_state=42) # comment if you want to run on entire set (takes longer time)

train.reset_index(drop=True, inplace=True)
y = train.isFraud.values



train = train.drop('isFraud', axis=1)

test = test.copy()

train = train.fillna(-1) #nan substitution could be done in a better way

test = test.fillna(-1) 

del train_transaction, train_identity, test_transaction, test_identity
# Label Encoding

for f in train.columns:

    if train[f].dtype=='object' or test[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train[f].values) + list(test[f].values))

        train[f] = lbl.transform(list(train[f].values))

        test[f] = lbl.transform(list(test[f].values))  
cols = list(train.columns)

len(cols)
scaler = StandardScaler() #MinMaxScaler StandardScaler RobustScaler



train[cols] = scaler.fit_transform(train[cols])

test[cols] = scaler.transform(test[cols])
N = 50



svd = TruncatedSVD(n_components=N, random_state=42)

X = svd.fit_transform(train[cols], y)  

svd.explained_variance_ratio_.sum()
df = pd.DataFrame()

df["target"] = y



for i in range(50):

    df[i] = X[:,i]

    

df.tail()
%%time



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

arch = "reg"



train[arch] = 0



for i, (train_index, valid_index) in enumerate(skf.split(X, y)):

    

    X_train = X[train_index]

    X_valid = X[valid_index]



    y_train = y[train_index]

    y_valid = y[valid_index]

    

    reg = LogisticRegression(C=1,

                             solver="newton-cg", 

                             penalty="l2", 

                             n_jobs=-1, 

                             max_iter=100).fit(X_train, y_train) 

    

    y_pred = reg.predict_proba(X_valid)[:,1]

    train.loc[valid_index, arch] = y_pred

    print(i, "ROC AUC:", round(roc_auc_score(y_valid, y_pred), 5))



print()

print("OOF ROC AUC:", round(roc_auc_score(y, train[arch]), 5))

print()
%%time



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

arch = "rfc"



train[arch] = 0

test[arch] = 0



for i, (train_index, valid_index) in enumerate(skf.split(X, y)):

    

    X_train = X[train_index]

    X_valid = X[valid_index]



    y_train = y[train_index]

    y_valid = y[valid_index]

    

    rfc = RandomForestClassifier(n_estimators=100,

                                 criterion='gini',

                                 n_jobs=-1).fit(X_train, y_train) 

    

    y_pred = rfc.predict_proba(X_valid)[:,1]

    train.loc[valid_index, arch] = y_pred

    print(i, "ROC AUC:", round(roc_auc_score(y_valid, y_pred), 5))



print()

print("OOF ROC AUC:", round(roc_auc_score(y, train[arch]), 5))

print()
%%time



arch = "lgb"



train[arch] = 0



rounds = 10000

early_stop_rounds = 300



params = {'objective': 'binary',

          'boosting_type': 'gbrt',

          'metric': 'auc',

          'seed': 42,

          'max_depth': -1,

          'verbose': -1,

          'n_jobs': -1}



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



for i, (train_index, valid_index) in enumerate(skf.split(X, y)):

    

    X_train = X[train_index]

    X_valid = X[valid_index]



    y_train = y[train_index]

    y_valid = y[valid_index]



    d_train = lgb.Dataset(X_train, y_train)

    d_valid = lgb.Dataset(X_valid, y_valid)    



    model = lgb.train(params,

                      d_train,

                      num_boost_round=rounds,

                      valid_sets=[d_train, d_valid],

                      valid_names=['train','valid'],

                      early_stopping_rounds=early_stop_rounds,

                      verbose_eval=0) 





    y_pred = model.predict(X_valid)

    train.loc[valid_index, arch] = y_pred

    auc = roc_auc_score(y_valid, y_pred)

    print(i, "ROC AUC:", round(auc, 5))



print()

print("OOF ROC AUC:", round(roc_auc_score(y, train[arch]), 5))

print()
%%time



arch = "cat"



train[arch] = 0



rounds = 10000

early_stop_rounds = 100



params = {'task_type': 'CPU', #GPU

          'iterations': rounds,

          'loss_function': 'Logloss',

          'eval_metric':'AUC',

          'random_seed': 42,

          'learning_rate': 0.5,

          'depth': 2}



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



for i, (train_index, valid_index) in enumerate(skf.split(X, y)):

    

    X_train = X[train_index]

    X_valid = X[valid_index]



    y_train = y[train_index]

    y_valid = y[valid_index]

    

    trn_data = Pool(X_train, y_train)

    val_data = Pool(X_valid, y_valid)

    

    clf = CatBoostClassifier(**params)

    clf.fit(trn_data,

            eval_set=val_data,

            use_best_model=True,

            early_stopping_rounds=early_stop_rounds,

            verbose=0)

    

    y_pred = clf.predict_proba(X_valid)[:, 1]

    train.loc[valid_index, arch] = y_pred

    auc = roc_auc_score(y_valid, y_pred)

    print(i, "ROC AUC:", round(auc, 5))



print()

print("OOF ROC AUC:", round(roc_auc_score(y, train[arch]), 5))

print()
models = ["cat", "lgb", "rfc", "reg"] #"nn"



for model in models:

    train[model] = train[model].rank()/len(train)



train[models].corr(method="spearman")
for arch in models:

    print(arch, round(roc_auc_score(y, train[arch]), 5))
train["avg"] = train[models].mean(axis=1)

print("avg", round(roc_auc_score(y, train["avg"]), 5))
from scipy.stats.mstats import gmean



def power_mean(x, p=1):

    if p==0:

        return gmean(x, axis=1)

    return np.power(np.mean(np.power(x,p), axis=1), 1/p)
for power in [0,1,2,4,8]:

    train["avg"] = power_mean(train[models].values, power)

    print(power, round(roc_auc_score(y, train["avg"]), 5))
%%time



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

arch = "stack"



train[arch] = 0



for i, (train_index, valid_index) in enumerate(skf.split(X, y)):

    

    X_train = train.loc[train_index, models]

    X_valid = train.loc[valid_index, models]



    y_train = y[train_index]

    y_valid = y[valid_index]

    

    reg = LogisticRegression(C=1,

                             solver="newton-cg", 

                             penalty="l2", 

                             n_jobs=-1, 

                             max_iter=100).fit(X_train, y_train) 

    

    y_pred = reg.predict_proba(X_valid)[:,1]

    train.loc[valid_index, arch] = y_pred

    print(i, "ROC AUC:", round(roc_auc_score(y_valid, y_pred), 5))



print()

print("OOF ROC AUC:", round(roc_auc_score(y, train[arch]), 5))

print()