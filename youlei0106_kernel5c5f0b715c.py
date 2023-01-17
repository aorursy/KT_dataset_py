import numpy as np

import pandas as pd

import lightgbm as lgb

import matplotlib

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold,RepeatedKFold

import warnings



from six.moves import urllib

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

warnings.filterwarnings('ignore')

%matplotlib inline

plt.style.use('seaborn')

from scipy.stats import norm, skew
#加载数据

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

features = [c for c in train.columns if c not in ['ID_code', 'target']]
#数据探查

train.describe()
train.info()

print(train.shape)

train.head(5)
#检查缺少的值

#检查合并后的值是否丢失

obs = train.isnull().sum().sort_values(ascending = False)

percent = round(train.isnull().sum().sort_values(ascending = False)/len(train)*100, 2)

pd.concat([obs, percent], axis = 1,keys= ['Number of Observations', 'Percent'])
#数据集中没有缺失的值

#可视化Satendar客户事务数据

#检查阶级不平衡

target = train['target']

train = train.drop(["ID_code", "target"], axis=1)

sns.set_style('whitegrid')

sns.countplot(target)
# params基于以下内核https://www.kaggle.com/brandenkmurray/nothing-works

params = {'objective' : "binary", 

               'boost':"gbdt",

               'metric':"auc",

               'boost_from_average':"false",

               'num_threads':8,

               'learning_rate' : 0.01,

               'num_leaves' : 13,

               'max_depth':-1,

               'tree_learner' : "serial",

               'feature_fraction' : 0.05,

               'bagging_freq' : 5,

               'bagging_fraction' : 0.4,

               'min_data_in_leaf' : 80,

               'min_sum_hessian_in_leaf' : 10.0,

               'verbosity' : 1}
train.shape
from sklearn.model_selection import KFold

folds = KFold(n_splits=5, shuffle=True, random_state=1)

features = [c for c in train.columns if c not in ['ID_code', 'target']]

X = train.values

y = target.values

X_test = test[features]

print(X.shape,y.shape,X_test.shape)
#构建Light GBM模型

import lightgbm as lgb



y_pred_lgb = np.zeros(len(X_test))

num_round = 1000000

for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):

    #print('褶皱Fold:', fold_n, '开始started at:', time.ctime())

    X_train, X_valid = X[train_index], X[valid_index]

    y_train, y_valid = y[train_index], y[valid_index]

    

    train_data = lgb.Dataset(X_train, label=y_train)

    valid_data = lgb.Dataset(X_valid, label=y_valid)

        

    lgb_model = lgb.train(params,train_data,num_round,#change 20 to 2000

                          valid_sets = [train_data, valid_data],

                          verbose_eval=1000,early_stopping_rounds = 3500)##change 10 to 200

            

    y_pred_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)/5

# lgb_params = {

#     'objective': 'regression',

#     'num_leaves': 100,

#     'subsample': 0.8,

#     'colsample_bytree': 0.75,

#     'verbose': -1,

#     'seed': 2018,

#     'boosting_type': 'gbdt',

#     'max_depth': 10,

#     'learning_rate': 0.04,

#     'metric': 'l2',

# }

# lgb_param = {

#    'objective' : "binary", 

#    'boost':"gbdt",

#    'metric':"auc",

#    'boost_from_average':"false",

#    'num_threads':8,

#    'learning_rate' : 0.01,

#    'num_leaves' : 13,

#    'max_depth':-1,

#    'tree_learner' : "serial",

#    'feature_fraction' : 0.05,

#    'bagging_freq' : 5,

#    'bagging_fraction' : 0.4,

#    'min_data_in_leaf' : 80,

#    'min_sum_hessian_in_leaf' : 10.0,

#    'verbosity' : 1

# }
# Run KFold运行KFold



# dtrain = lgb.Dataset(data=X, label=np.log1p(y), free_raw_data=False)

# dtrain.construct()

# for trn_idx, val_idx in folds.split(X):

#     # Train lightgbm火车lightgbm

#     clf = lgb.train(

#         params=lgb_param,

#         train_set=dtrain.subset(trn_idx),

#         num_boost_round=10000, 

#         verbose_eval=50

#     )

    # Predict Out Of Fold and Test targets

    # Using lgb.train, predict will automatically select the best round for prediction

    #oof_preds[val_idx] = clf.predict(dtrain.data.iloc[val_idx])

  # sub_preds += clf.predict(X_test[features]) / folds.n_splits

    # Display current fold score显示当前折痕

#     print('Current fold score : %9.6f' % mean_squared_error(np.log1p(train_df['target'].iloc[val_idx]), 

#                              oof_preds[val_idx]) ** .5)

    

# Display Full OOF score (square root of a sum is not the sum of square roots)

# print('Full Out-Of-Fold score : %9.6f' 

#       % (mean_squared_error(np.log1p(train_df['target']), oof_preds) ** .5))
submission_lgb = pd.DataFrame({

        "ID_code": test.ID_code.values,

        "target":y_pred_lgb

    })

submission_lgb.to_csv('submission_lgb.csv', index=False)

print("Handle a successful")