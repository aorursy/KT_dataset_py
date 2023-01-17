import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import skew

from sklearn.preprocessing import PowerTransformer

import lightgbm as lgb
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission.head()
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

train.head()
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test.head()
print('=========== train infomation ===========')

train.info()

print('\n\n=========== test infomation ===========')

test.info()
data = pd.concat([train, test])

data.shape
# current categorical data

cat_cols = data.loc[:,data.dtypes == 'object'].columns

data.loc[:,cat_cols].head()
# categorical data after conversion to one-hot vector

cat_data = pd.get_dummies(data.loc[:, cat_cols], drop_first=True)

cat_data.head()
# current numeric data

numerics = data.loc[:,data.dtypes != 'object'].drop('Id', axis=1)

numerics.head()
# numeric data after conversion to logarithm

log_numerics = np.log1p(numerics)

log_numerics.head()
# compare skewnesses before with after of logarithmization

skewness = pd.concat([numerics.apply(lambda x: skew(x.dropna())),

                      log_numerics.apply(lambda x: skew(x.dropna()))],

                     axis=1).rename(columns={0:'original', 1:'logarithmization'}).sort_values('original')

skewness.plot.barh(figsize=(12,10), title='Comparison of skewness of original and logarithmized', width=0.8);
# merge categorical and numeric columns

optimized_data = pd.concat([data['Id'], cat_data, log_numerics], axis=1)

optimized_data.head()
# split data into X_train, y_train and test

train = optimized_data[:train.shape[0]]

test = optimized_data[train.shape[0]:].drop(['Id', 'SalePrice'], axis=1)

X_train = train.drop(['Id', 'SalePrice'], axis=1)

y_train = train['SalePrice']
# train

lgb_train = lgb.Dataset(X_train, y_train)

params = {

        'task' : 'train',

        'boosting_type' : 'gbdt',

        'objective' : 'regression',

        'metric' : {'l2'},

        'num_leaves' : 40,

        'learning_rate' : 0.1,

        'feature_fraction' : 0.9,

        'bagging_fraction' : 0.8,

        'bagging_freq': 5,

        'verbose' : 0

}

gbm = lgb.train(params, lgb_train)

# predict

pred = gbm.predict(test)
# convert logarithms into exponent

pred = np.expm1(pred)

# create submission file

results = pd.Series(pred, name='SalePrice')

submission = pd.concat([submission['Id'], results], axis=1)

submission.to_csv('submission.csv', index=False)

submission.head()