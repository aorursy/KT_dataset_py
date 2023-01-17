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
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

sample_data = pd.read_csv("../input/sample_submission.csv") 
print(train_data.shape) 

train_data.head()
print(test_data.shape) 

test_data.head()
ID_code = sample_data.ID_code.values

print(sample_data.shape)

sample_data.head()
oof = train_data[["ID_code","target"]]

oof['predict'] = 0

prediction = test_data['ID_code']

label_df = train_data['target']
# def data_handle(data):#处理

#     x,y = data.shape

#     a = 0

#     features = [c for c in data.columns]

#     for i in range(1,int(y/10)):

#         i = i*10

#         data["mean%d"%i] = data[features[a:i]].mean(axis=1)

#         data["std%d"%i] = data[features[a:i]].std(axis=1)

#         data["skew%d"%i] = data[features[a:i]].skew(axis=1)

#         data["kurtosis%d"%i] = data[features[a:i]].kurtosis(axis=1)

#         a = i

#     return data
train = train_data

test = test_data

len_train = len(train_data)

print("处理后：",train.shape,test.shape,len_train)
from sklearn.model_selection import StratifiedKFold ,KFold

import lightgbm as lgb

import xgboost as xgb

from sklearn.metrics import roc_auc_score, mean_absolute_error

skf_three= StratifiedKFold(n_splits=15, shuffle=True, random_state=2319)
random_state = 42

np.random.seed(random_state)

params = {

    "objective" : "binary", "metric" : "auc", "boosting": 'gbdt', "max_depth" : -1, "num_leaves" : 13,

    "learning_rate" : 0.01, "bagging_freq": 0.5, "bagging_fraction" : 0.4, "feature_fraction" : 0.05,

    "min_data_in_leaf": 80, "min_sum_heassian_in_leaf": 10, 'num_leaves': 13,

    'num_threads': 8,"tree_learner": "serial", "boost_from_average": "false",

    "bagging_seed" : random_state, "verbosity" : 1, "seed": random_state

}
random_state = 42

np.random.seed(random_state)

lgb_params = {

    "objective" : "binary",

    "metric" : "auc",

    "boosting": 'gbdt',

    "max_depth" : -1,

    "num_leaves" : 13,

    "learning_rate" : 0.01,

    "bagging_freq": 5,

    "bagging_fraction" : 0.4,

    "feature_fraction" : 0.05,

    "min_data_in_leaf": 80,

    "min_sum_heassian_in_leaf": 10,

    "tree_learner": "serial",

    "boost_from_average": "false",

    #"lambda_l1" : 5,

    #"lambda_l2" : 5,

    "bagging_seed" : random_state,

    "verbosity" : 1,

    "seed": random_state

}
random_state = 42

np.random.seed(random_state)

def augment(x,y,t=2):

    xs,xn = [],[]

    for i in range(t):

        mask = y>0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xs.append(x1)



    for i in range(t//2):

        mask = y==0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xn.append(x1)



    xs = np.vstack(xs)

    xn = np.vstack(xn)

    ys = np.ones(xs.shape[0])

    yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn])

    y = np.concatenate([y,ys,yn])

    return x,y
def augmen(x,y,t=2):

    xs,xn = [],[]

    for i in range(t):

        mask = y>0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xs.append(x1)



    for i in range(t//2):

        mask = y==0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xn.append(x1)



    xs = np.vstack(xs); xn = np.vstack(xn)

    ys = np.ones(xs.shape[0]);yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn]); y = np.concatenate([y,ys,yn])

    return x,y
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

oof = train[['ID_code', 'target']]

oof['predict'] = 0

predictions = test[['ID_code']]

val_aucs = []

feature_importance_df = pd.DataFrame()

features = [col for col in train.columns if col not in ['target', 'ID_code']]

X_test = test[features].values

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train['target'])):

    X_train, y_train = train.iloc[trn_idx][features], train.iloc[trn_idx]['target']

    X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx]['target']

    

    N = 5

    p_valid,yp = 0,0

    for i in range(N):

        X_t, y_t = augment(X_train.values, y_train.values)

        X_t = pd.DataFrame(X_t)

        X_t = X_t.add_prefix('var_')

    

        trn_data = lgb.Dataset(X_t, label=y_t)

        val_data = lgb.Dataset(X_valid, label=y_valid)

        evals_result = {}

        lgb_clf = lgb.train(params,

                        trn_data,

                        100000,

                        valid_sets = [trn_data, val_data],

                        early_stopping_rounds=3000,

                        verbose_eval = 1000,

                        evals_result=evals_result

                       )

        p_valid += lgb_clf.predict(X_valid)

        yp += lgb_clf.predict(X_test)

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = features

    fold_importance_df["importance"] = lgb_clf.feature_importance()

    fold_importance_df["fold"] = fold + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    oof['predict'][val_idx] = p_valid/N

    val_score = roc_auc_score(y_valid, p_valid)

    val_aucs.append(val_score)

    

    predictions['fold{}'.format(fold+1)] = yp/N
# submission

predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)

predictions.to_csv('lgb_all_predictions.csv', index=None)

sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})

sub_df["target"] = predictions['target']

sub_df.to_csv("lgb_submission.csv", index=False)

#oof.to_csv('lgb_oof.csv', index=False)

submission3 = pd.DataFrame({"ID_code":ID_code,"target":predictions['fold4']*0.5+predictions['fold5']*0.5})

submission3.to_csv("lgb_submission3.csv", index=False)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

# oof = train_data.iloc[:,:2]

# print(oof)

# oof['predict'] = 0

# predictions = test_data['ID_code']

val_aucs = []



features = [col for col in train_data.columns if col not in ['target', 'ID_code']]

X_test = test_data[features].values



for fold, (trn_idx, val_idx) in enumerate(skf.split(train_data, label_df)):

    X_train, y_train = train_data.iloc[trn_idx][features], label_df.iloc[trn_idx]

    X_valid, y_valid = train_data.iloc[val_idx][features], label_df.iloc[val_idx]

    

    N = 3

    p_valid,yp = 0,0

    for i in range(N):

        X_t, y_t = augmen(X_train.values, y_train.values)

        X_t = pd.DataFrame(X_t)

        X_t = X_t.add_prefix('var_')

    

        trn_data = lgb.Dataset(X_t, label=y_t)

        val_data = lgb.Dataset(X_valid, label=y_valid)

        evals_result = {}

        #lgb_clf = lgb.train(params,trn_data,100000,valid_sets = [trn_data, val_data],early_stopping_rounds=1000,verbose_eval = 5000,evals_result=evals_result)

        lgb_clf = lgb.train(lgb_params,trn_data,100000,valid_sets = [trn_data, val_data],early_stopping_rounds=1000,verbose_eval = 5000,evals_result=evals_result)

        p_valid += lgb_clf.predict(X_valid)

        yp += lgb_clf.predict(X_test)

    

    oof['predict'][val_idx] = p_valid/N

    val_score = roc_auc_score(y_valid, p_valid)

    val_aucs.append(val_score)

    prediction['fold{}'.format(fold+1)] = yp/N
submission = pd.DataFrame({"ID_code":ID_code,"target":yp/N})

submission.to_csv("lgb_submission.csv", index=False)

submission1 = pd.DataFrame({"ID_code":ID_code,"target":yp/N*0.4+predictions['target']*0.6})

submission1.to_csv("lgb_submission1.csv", index=False)

submission2 = pd.DataFrame({"ID_code":ID_code,"target":yp/N*0.25+predictions['target']*0.25+predictions['fold4']*0.25+predictions['fold5']*0.25})

submission2.to_csv("lgb_submission2.csv", index=False)