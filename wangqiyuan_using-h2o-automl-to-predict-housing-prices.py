# You can easily install the library using pip

!pip install h2o
# And then load the libraries you'll use in this notebook

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline





import h2o

from h2o.automl import H2OAutoML
# Initialize your cluster

h2o.init()
train = h2o.import_file('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = h2o.import_file('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train.head()
train = train[train['GrLivArea'] < 4500]

train['SalePrice'] = train['SalePrice'].log1p()
# Identify predictors and response

x = [col for col in train.columns if col not in ['Id','SalePrice']]

y = 'SalePrice'

test_id = test['Id']
aml = H2OAutoML( max_runtime_secs=30000,nfolds=10, stopping_metric = 'RMSLE')

aml.train(x = x, y = y, training_frame = train)
#                max_runtime_secs=36000,最大运行时间

#                max_runtime_secs_per_model=3600,最大单模型运行时间

#seed，随便选，随机种子

#                nfolds=10, 交叉验证，越大交叉验证度越高，结果越好，越耗时，默认为5

#sort_metric: Specifies the metric used to sort the Leaderboard by at the end of an AutoML run. Available options include:

#AUTO: This defaults to AUC for binary classification, mean_per_class_error for multinomial classification, and deviance for regression.

#For binomial classification choose between AUC, "logloss", "mean_per_class_error", "RMSE", "MSE". For multinomial classification choose between "mean_per_class_error", "logloss", "RMSE", "MSE". For regression choose between "deviance", "RMSE", "MSE", "MAE", "RMLSE".

#max_models=333,最大模型数目；默认为无穷大

#                include_algos=['XGBoost'], 此选项允许您指定在模型构建阶段要包括在AutoML运行中的算法列表。该选项默认为None / Null，这意味着将包括所有算法，除非在该exclude_algos选项中指定了任何算法。

#                verbosity='info'传递何种信息，info为全部
lb = aml.leaderboard; lb
aml.leader
aml.leader.varimp_plot()
preds = aml.leader.predict(test)
# Convert results back(they had been transformed using log, remember?) and save them in a csv format.

result = preds.expm1()

sub = test_id.cbind(result)

sub.columns = ['Id','SalePrice']

sub = sub.as_data_frame()

sub.to_csv('submission.csv', index = False)