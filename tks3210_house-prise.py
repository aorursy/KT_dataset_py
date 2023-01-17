# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



pd.set_option('display.max_columns', 80)

pd.set_option('display.max_rows', 50)

pd.options.display.max_rows = 50
#!cat ../input/house-prices-advanced-regression-techniques/data_description.txt
train_row = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_row = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv") 

train_row.shape

train_row.index

train_row.columns

(train_row.dtypes == object).sum()
train_row.head()
test_row.head()
# dataのマージ

data_row = pd.concat([train_row, test_row], sort=False)
data_row["MSZoning"].unique()
from sklearn.preprocessing import LabelEncoder

# 複数のLabelを数字に変換してくれる。



lbl = LabelEncoder()

lbl.fit(list(train_row.loc[:, "MSZoning"].values)) #list構造を食わせると重複を取り除いてくれる

print(lbl.classes_)

lbl.transform(list(train_row.loc[:, "MSZoning"].values))
# object形式のものを数字に変換してくれる。



from sklearn.preprocessing import LabelEncoder

#=の場合、参照渡しになるので注意

categorical_index = []



data = data_row.copy(deep=False)

for i in range(data.shape[1]):

    if data.iloc[:, i].dtypes == object:

        lbl = LabelEncoder()

        lbl.fit(list(data.iloc[:, i].values))

        data.iloc[:, i] = lbl.transform(list(data.iloc[:,i].values))

        #print(data.columns[i])

        categorical_index += [data.columns[i]]

categorical_index
data.columns
#欠損値を含むカラム

pd.set_option('display.max_rows', 500)

#print(data_row.isnull().sum()) # null値のみ抽出

#index = (train_row.dtypes == "int64").values.tolist()

#type(data_row.isnull().sum())

data_row.isnull().sum()[data_row.isnull().sum()!=0]
#欠損値の視覚化

import missingno as msno



msno.matrix(df=data, figsize=(20,14), color=(0.5,0,0))
#欠損データの廃棄(解析対象から外す)

data_ = data.drop(['LotFrontage', 'GarageYrBlt'], axis=1)

nullinfo = data_.isnull().sum()

nullinfo[nullinfo!=0]
#欠損データの補間

#中央値補間

data__ = data_.copy(deep=False)

#data__ = data__.drop(['SalePrice'], axis=1)

data__ = data__.fillna(data__.median()) #SalePriceも中央値補間されてしまうが、いったんよし

nullinfo = data__.isnull().sum()

nullinfo[nullinfo!=0]

data__



msno.matrix(df=data__, figsize=(20,14), color=(0.5,0,0))
from sklearn.model_selection import train_test_split

data_in = data__.copy(deep=False)

train = data_in[:len(train_row)]

test = data_in[len(train_row):]





y_train = train['SalePrice']

X_train = train.drop('SalePrice', axis=1)



X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3,random_state=0)

X_test = test.drop('SalePrice', axis=1)

#均等に分割させたいデータはstratifyをどう指定すれば良いか？
pd.set_option('display.max_columns', 35)

X_train.head()
categorical_index += ['MSSubClass']

categorical_index
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_index)

lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_index)



params =  {'metric': 'rmse', 'max_depth' : 9}



gbm = lgb.train(params, lgb_train, valid_sets=lgb_eval, num_boost_round=10000, early_stopping_rounds=100, verbose_eval=50)
import lightgbm as lgb

import optuna

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



def objective(trial):

    params = {

        'metric': 'rmse',

        'max_depth' : trial.suggest_int('max_depth', 5, 30),

        'learning_rate': trial.suggest_loguniform('lambda_l1', 0.005, 0.03),

        'num_leaves': trial.suggest_int('num_leaves', 32, 128),

    }

#    params =  {'metric': 'rmse', 'max_depth' : 9}

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_index)

    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_index)

    gbm = lgb.train(params, lgb_train, valid_sets=lgb_eval, num_boost_round=10000, early_stopping_rounds=100, verbose_eval=50)

    y_pred_valid = gbm.predict(X_valid, num_iteration=gbm.best_iteration)

    RMSE = np.sqrt(mean_squared_error(y_valid, y_pred_valid))

    return RMSE


study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

study.optimize(objective, n_trials=100)

study.best_params


params = {

    'metric': 'rmse',

    'max_depth' : study.best_params['max_depth'],

    'learning_rate': study.best_params['lambda_l1'],

    'num_leaves': study.best_params['num_leaves'],

}

#    params =  {'metric': 'rmse', 'max_depth' : 9}

lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_index)

lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_index)

gbm = lgb.train(params, lgb_train, valid_sets=lgb_eval, num_boost_round=10000, early_stopping_rounds=100, verbose_eval=50)

y_pred_valid = gbm.predict(X_valid, num_iteration=gbm.best_iteration)





y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred_int = y_pred.astype(np.int32)

y_submit = (y_pred_int/500).astype(np.int32)*500 
#y_submit.savetxt('submisstion.csv', index=False)

np.savetxt('submission.csv', y_submit, fmt='%d')

X_test["SalePrice"] = y_submit
X_test[["Id", "SalePrice"]].to_csv('submission.csv', index=False)