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
from sklearn.model_selection import KFold
import lightgbm as lgb
from collections import Counter
import gc
# lgb_params =  {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'regression',
#     'metric': 'rmse',
#     'num_leaves': 50,
#     'feature_fraction': 0.7,
#     'bagging_fraction': 0.7,
#     'bagging_freq': 4,
#     'learning_rate': 0.015,
#     'zero_as_missing':True,
#     'verbose': 0
#     }

lgb_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    "num_leaves": 180,
    "feature_fraction": 0.50,
    "bagging_fraction": 0.50,
    'bagging_freq': 4,
    "max_depth": -1,
    "reg_alpha": 0.3,
    "reg_lambda": 0.1,
    #"min_split_gain":0.2,
    "min_child_weight":10,
    'zero_as_missing':True,
    'verbose': 0
}
def setup_dataset():
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    subs = pd.read_csv("../input/sample_submission.csv")

    # preprocess
    train["target"] = np.log1p(train["target"])
    subs["target"] = 0.0
    # Replace 0 with NaN to ignore them.
    train = prepare(train.replace(0, np.nan),"train")
    test = prepare(test.replace(0, np.nan),"test")
    return train,test,subs

def prepare(data,case):
    if case=="train":
        del_col=["ID","target"]
    elif case=="test":
        del_col=["ID"]
    data_c = data.copy()
    data['mean'] = data_c.drop(del_col,axis=1).mean(axis=1)
    data['std'] = data_c.drop(del_col,axis=1).std(axis=1)
    data['min'] = data_c.drop(del_col,axis=1).min(axis=1)
    data['max'] = data_c.drop(del_col,axis=1).max(axis=1)
    data['number_of_different'] = data_c.drop(del_col,axis=1).nunique(axis=1)               # Number of diferent values in a row.
    data['non_zero_count'] = data_c.drop(del_col,axis=1).fillna(0).astype(bool).sum(axis=1) # Number of non zero values (e.g. transaction count)
    
    #extra
    d_matrix = data_c.drop(del_col,axis=1).fillna(0).apply(pair_check,axis=1)
    tmp_result = pd.DataFrame(list(d_matrix),columns=["not_pair_cnt", "not_pair_sum", "not_pair_max", "not_pair_min", "not_pair_mean","even_pair_cnt","even_pair_sum", "even_pair_max","even_pair_min","even_pair_mean"])
    for col in tmp_result.columns.tolist():
        data[col]=tmp_result[col]
    return data
def pair_check(column_values):
    v = column_values.values.tolist()
    not_pair_cnt=0
    not_pair_sum=0
    not_pair_max=0
    not_pair_min=9999999999
    even_pair_cnt=0
    even_pair_sum=0
    even_pair_max=0
    even_pair_min=0
    for key,cnt in Counter(v).items():
        if key ==0:
            continue
        if cnt %2==1:
            not_pair_cnt+=cnt
            not_pair_sum+=key
            if not_pair_max<key:
                not_pair_max=key
            if not_pair_min>key:
                not_pair_min=key
        else:
            even_pair_cnt+=cnt
            even_pair_sum+=key
            if even_pair_max<key:
                even_pair_max=key
            if even_pair_min>key:
                even_pair_min=key
    if not_pair_cnt==0:
        not_pair_mean=0
    else:
        not_pair_mean = not_pair_sum / not_pair_cnt
    if even_pair_cnt==0:
        even_pair_mean=0
    else:
        even_pair_mean = even_pair_sum / even_pair_cnt
    return (not_pair_cnt, not_pair_sum, not_pair_max, not_pair_min, not_pair_mean, even_pair_cnt,even_pair_sum, even_pair_max,even_pair_min, even_pair_mean)
def train_lgb(d_train, d_test, d_subs, epoch, kfold_num=3,
                delete_cols=[], lgb_params=lgb_params):
    best_score=0
    print("epoch:{}".format(epoch))
    d_train = d_train.drop(delete_cols,axis=1)
    d_test = d_test.drop(delete_cols,axis=1)
    X_test  = d_test.drop(["ID"],axis=1)
    
    # substract ID,target
    fti_list=np.zeros(d_train.shape[1] - 2)
    low_fti=[]
    
    del d_test
    gc.collect()

    for i,(train_idx, valid_idx) in enumerate(KFold(n_splits=kfold_num).split(d_train)):
        # setup dataset
        X_train = d_train.iloc[train_idx,:].drop(["ID","target"],axis=1)
        X_valid = d_train.iloc[valid_idx,:].drop(["ID","target"],axis=1)
        Y_train = d_train.iloc[train_idx,:]["target"]
        Y_valid = d_train.iloc[valid_idx,:]["target"]
        lgb_train = lgb.Dataset(X_train, label=Y_train)
        lgb_valid = lgb.Dataset(X_valid, label=Y_valid)
        # training!!
        lgb_clf = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=10000,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train','valid'],
            early_stopping_rounds=50,
            verbose_eval=50
        )
        # get feature importances
        fti_list += lgb_clf.feature_importance()
        # set prediction
        d_subs["target"] += np.expm1(lgb_clf.predict(X_test))
        best_score += lgb_clf.best_score["valid"]['rmse']
    d_subs["target"] /= kfold_num # 3
    best_score /= kfold_num # 3
    # detect low important columns
    tee=[]
    for i, feat in enumerate(X_train.columns.tolist()):
        ##### delete 0.0 importance columns ######
        if fti_list[i]==0.0:
            low_fti.append(feat)
        else:
            tee.append([feat,fti_list[i]])
    print("best_score:{:.4f}".format(best_score))
    return d_subs, low_fti, best_score,tee
def execute_santander():
    print("start")
    train,d_test,subs = setup_dataset()
    delete_cols=[]
    fti_results=[]
    score=99999.9
    for epoch in range(10):
        d_train = train.copy()
        d_subs = subs.copy()
        d_subs, low_fti, best_score,tee = train_lgb(d_train, d_test, d_subs, epoch=epoch,
                                       kfold_num=3, delete_cols=delete_cols,
                                       lgb_params=lgb_params)
        if score < best_score:
            print("score is not improved")
            break
        else:
            score = best_score
            delete_cols.extend(low_fti)
            print("delete_cols numbers:",len(delete_cols))
            d_subs.to_csv("subs_lgb.csv",index=False)
            fti_results.append(tee)
    print("finish")
    return fti_results
fti_results = execute_santander()
# for t in fti_results[0]:
#     print(t)
df = pd.DataFrame(fti_results[0],columns=["name","score"])
df.sort_values(by="score",ascending=False)


