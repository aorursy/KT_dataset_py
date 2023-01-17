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
import lightgbm as lgb

from sklearn.model_selection import GroupKFold
df = pd.read_csv('../input/Training_data.csv')

df_test = pd.read_csv('../input/Testing_data.csv')

df_sub = pd.read_csv('../input/Submission.csv')
features = df.columns[1:-1]

target = df.target
df.head()
def get_folds(df=None, n_splits=5, index = 'index'):

    """Returns dataframe indices corresponding to Visitors Group KFold"""

    # Get sorted unique visitors

    unique_vis = np.array(sorted(df[index].unique()))



    # Get folds

    folds = GroupKFold(n_splits=n_splits)

    fold_ids = []

    ids = np.arange(df.shape[0])

    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):

        fold_ids.append(

            [

                ids[df[index].isin(unique_vis[trn_vis])],

                ids[df[index].isin(unique_vis[val_vis])]

            ]

        )

    return fold_ids
X = df[features]

y = target
params = {

        'boosting_type': 'gbdt',

        'objective': 'binary',

        'metric': 'auc', #root mean square error

        'num_leaves': 63, #max number of leaves in one tree usually 2^(max_depth)-1

        'max_depth': 6, #limit the max depth for tree model

        'learning_rate': 0.03,

        #"min_child_samples" : 100, #minimal number of data in one leaf. Can be used to deal with over-fitting

        "bagging_fraction" : 0.9, #randomly select part of data without resampling

        "feature_fraction" : 0.9, #randomly select part of features on each iteration

        'random_state':1

        #"bagging_frequency" : 5,

        #"bagging_fraction" : 0.75,

        #'subsample':.9,

        #'colsample_bytree':.9

        #'zero_as_missing':True

    }



sub_reg_preds = np.zeros(df_test.shape[0])

folds = get_folds(df = df.reset_index() ,index = 'index', n_splits=5)

# num_fold, train_index, valid_index

for fold_, (trn_, val_) in enumerate(folds):

    print("-"* 20 + "Fold :"+str(fold_+1) + "-"* 20)

    

    #建立 training data & validation data

    trn_x, trn_y = pd.DataFrame(X).iloc[trn_], pd.DataFrame(y).iloc[trn_]

    val_x, val_y = pd.DataFrame(X).iloc[val_], pd.DataFrame(y).iloc[val_]

    

    lgb_train = lgb.Dataset(trn_x,label=trn_y)

    lgb_eval = lgb.Dataset(val_x,label=val_y)

    

    #num_boost_round=1000

    #early stopping =100

    #verbose : 輸出validation data結果的週期

    model = lgb.train(params, lgb_train, 10000, valid_sets=[lgb_eval], early_stopping_rounds=50, verbose_eval=100)

    

    #拿validation表現最好的那棵樹預測，其中預測成負值的將它變成0

    _preds = model.predict(df_test[features], num_iteration=model.best_iteration)

    _preds[_preds < 0] = 0

    #預測的結果取平均

    sub_reg_preds += _preds / len(folds)

print("\nmodeling finished!! \n")
df_sub['Probability'] = sub_reg_preds

df_sub.to_csv('sub.csv',index=False)