# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

from sklearn.preprocessing import StandardScaler

import string

import category_encoders as ce

import time
#load data

rawtrain=pd.read_csv('../input/cat-in-the-dat-ii/train.csv')

rawtest=pd.read_csv('../input/cat-in-the-dat-ii/test.csv')

sub=pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv')

target=rawtrain['target']

train=rawtrain.drop(['id','target'],axis=1)

test=rawtest.drop('id',axis=1)

#======encode ordinal

cate_ord=['ord_1','ord_2']

for c in cate_ord:

    print(rawtrain[c].unique())

levelmap={c:i for i,c in enumerate(['Novice','Contributor', 'Expert', 'Master','Grandmaster'])}

train['ord_1']=train['ord_1'].replace(levelmap)

test['ord_1']=test['ord_1'].replace(levelmap)

tempratmap={c:i for i,c in enumerate(['Freezing','Cold', 'Warm','Hot' , 'Boiling Hot' ,'Lava Hot' ])}

train['ord_2']=train['ord_2'].replace(tempratmap)

test['ord_2']=test['ord_2'].replace(tempratmap)

lowermap={c:i for i,c in enumerate(string.ascii_lowercase)}

train['ord_3']=train['ord_3'].replace(lowermap)

test['ord_3']=test['ord_3'].replace(lowermap)

upperletter=rawtrain['ord_4'].unique().tolist()

upperletter.remove(np.nan)

upperletter.sort()

uppermap={c:i for i,c in enumerate(string.ascii_uppercase)}

train['ord_4']=train['ord_4'].replace(uppermap)

test['ord_4']=test['ord_4'].replace(uppermap)

#/ord_5

alletter=string.ascii_letters

allmap={c:i for i,c in enumerate(alletter)}

def getP(x,p):

    if pd.isnull(x):

        return x

    else:

        if p==0:

            return x[0]

        else:

            return x[1]

        

train['ord_5_0']=rawtrain['ord_5'].apply(lambda x: getP(x,0)).replace(allmap)

train['ord_5_1']=rawtrain['ord_5'].apply(lambda x: getP(x,1)).replace(allmap)

test['ord_5_0']=rawtest['ord_5'].apply(lambda x: getP(x,0)).replace(allmap)

test['ord_5_1']=rawtest['ord_5'].apply(lambda x: getP(x,1)).replace(allmap)

train=train.drop('ord_5',axis=1)

test=test.drop('ord_5',axis=1)

#======encode binary and nominal+label to num for k mode clustering

normcol59=['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

train_cluster=train.drop(normcol59,axis=1)

test_cluster=test.drop(normcol59,axis=1)

for c in train_cluster.columns:

    test_cluster[c].fillna(train_cluster[c].mode()[0], inplace = True)

    train_cluster[c].fillna(train_cluster[c].mode()[0], inplace = True)



bincol_labeled=['bin_3', 'bin_4']

binOE=OrdinalEncoder()

train_cluster[bincol_labeled]=binOE.fit_transform(train_cluster[bincol_labeled])

test_cluster[bincol_labeled]=binOE.transform(test_cluster[bincol_labeled])



normcol_labeled=['nom_0','nom_1','nom_2', 'nom_3', 'nom_4']

binOE=OrdinalEncoder()

train_cluster[normcol_labeled]=binOE.fit_transform(train_cluster[normcol_labeled])

test_cluster[normcol_labeled]=binOE.transform(test_cluster[normcol_labeled])

#==================k mode clustering========

from kmodes.kmodes import KModes

km = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1,random_state=1990)

train['cluster'] = km.fit_predict(train_cluster)

test['cluster'] = km.predict(test_cluster)
#======target encode binary and norminal

bincol=['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

normcol=['nom_0','nom_1','nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

#modified from category_encoders

def TargetEncode(trainc,testc,targetc, smooth):

    print('Target encoding...')

    smoothing=smooth

    oof = np.zeros(len(trainc))

    for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=2020, shuffle=True).split(trainc, targetc):

        train_x=trainc.iloc[tr_idx].reset_index(drop=True)

        valid_x=trainc.iloc[oof_idx].reset_index(drop=True)

        target_train=targetc.iloc[tr_idx].reset_index(drop=True)

        prior = target_train.mean()

        tmp = target_train.groupby(train_x).agg(['sum', 'count'])

        tmp['mean'] = tmp['sum'] / tmp['count']

        smoothing = 1 / (1 + np.exp(-(tmp["count"] - 1) / smoothing))

        cust_smoothing = prior * (1 - smoothing) + tmp['mean'] * smoothing 

        tmp['smoothing'] = cust_smoothing

        tmp = tmp['smoothing'].to_dict()

        oof[oof_idx]=valid_x.map(tmp).values

    prior = targetc.mean()

    tmp = targetc.groupby(trainc).agg(['sum', 'count'])

    tmp['mean'] = tmp['sum'] / tmp['count']

    smoothing = 1 / (1 + np.exp(-(tmp["count"] - 1) / smoothing))

    cust_smoothing = prior * (1 - smoothing) + tmp['mean'] * smoothing 

    tmp['smoothing'] = cust_smoothing

    tmp = tmp['smoothing'].to_dict()

    testc=testc.map(tmp)

    return oof, testc

for n in normcol+bincol:

    train[n],test[n]=TargetEncode(train[n],test[n],target,0.3)



floatlist=['ord_1','ord_2','ord_3','ord_4','ord_5_0','ord_5_1']

train[floatlist]=train[floatlist].astype(float)

test[floatlist]=test[floatlist].astype(float)

usedfeatures=test.columns.tolist()

cat_cols=['cluster']

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1990)



params = {#1

        'learning_rate': 0.05,

        'feature_fraction': 0.15,

        'min_data_in_leaf' : 80,

        'max_depth': 6,

        'objective': 'binary',

        'num_leaves':25,

        'metric': 'auc',

        'n_jobs': -1,

        'feature_fraction_seed': 42,

        'bagging_seed': 42,

        'boosting_type': 'gbdt',

        'verbose': 1,

        'is_unbalance': True,

        'boost_from_average': False}



t1=time.clock()

traintion = np.zeros(len(train))

validation = np.zeros(len(train))

predictions = np.zeros(len(test))

feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,target)):

    print("fold n°{}".format(fold_))

    train_x=train.iloc[trn_idx][usedfeatures].reset_index(drop=True)

    valid_x=train.iloc[val_idx][usedfeatures].reset_index(drop=True)

    target_train=target.iloc[trn_idx].reset_index(drop=True)

    target_valid=target.iloc[val_idx].reset_index(drop=True)

    trn_data = lgb.Dataset(train_x,

                           label=target_train,

                           categorical_feature=cat_cols

                          )

    val_data = lgb.Dataset(valid_x,

                           label=target_valid,

                           categorical_feature=cat_cols

                          )



    num_round = 1000000

    clf = lgb.train(params,

                    trn_data,

                    num_round,

                    valid_sets = [trn_data, val_data],

                    verbose_eval=250,

                    early_stopping_rounds = 250)

    traintion[trn_idx] += clf.predict(train_x, num_iteration=clf.best_iteration)/(folds.n_splits-1)

    validation[val_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = usedfeatures

    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(test[usedfeatures], num_iteration=clf.best_iteration) / folds.n_splits

t2=time.clock()-t1

print("Time used: {:<8.5f}".format(t2))

print("Train AUC score: {:<8.5f}".format(roc_auc_score(target,traintion)))

print("Valid AUC score: {:<8.5f}".format(roc_auc_score(target,validation)))



sub['target']=predictions

sub.to_csv('submission.csv',index=False)

f_noimp_avg = (feature_importance_df[["feature", "importance"]]

        .groupby("feature")

        .mean()

        .sort_values(by="importance", ascending=False))

plt.figure(figsize=(12,20))

sns.barplot(x="importance",

            y="feature",data=f_noimp_avg.reset_index())