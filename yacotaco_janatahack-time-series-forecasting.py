# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/electrical-consumption/train_6BJx641.csv')
test = pd.read_csv('/kaggle/input/electrical-consumption/test_pavJagI.csv')
sns.distplot(train.electricity_consumption)
train.skew()
sns.distplot(train.var1)
train.var2.unique()
g = sns.FacetGrid(train, hue="var2", palette="Set1", height=5, hue_kws={"marker": ["o", "^", "*"]})
g.map(plt.scatter, "electricity_consumption", "windspeed", s=100, linewidth=.5, edgecolor="white")
g.add_legend();
test.head()
plt.plot(train.query('var2 == "A"').electricity_consumption)
plt.plot(train.query('var2 == "B"').electricity_consumption)
plt.plot(train.query('var2 == "C"').electricity_consumption)
def convert_date(dataset):
        dataset['datetime'] = pd.to_datetime(dataset['datetime'])
        dataset['day'] = dataset.apply(lambda row: row.datetime.day, axis=1)
        dataset['month'] = dataset.apply(lambda row: row.datetime.month, axis=1)
        dataset['year'] = dataset.apply(lambda row: row.datetime.year, axis=1)
        dataset['hour'] = dataset.apply(lambda row: row.datetime.hour, axis=1)
            
convert_date(train)
convert_date(test)
def day(row):
    p = pd.Period(row.datetime.date(), freq='H')
    return p.dayofweek

def day_of_week(dataset):
    dataset['day_of_week'] = dataset.apply(lambda row: day(row), axis=1)

day_of_week(train)
day_of_week(test)
def day(row):
    p = pd.Period(row.datetime.date(), freq='H')
    return p.week

def week(dataset):
    dataset['week'] = dataset.apply(lambda row: day(row), axis=1)

week(train)
week(test)
def multilabel_var2(dataset):
    mlb = preprocessing.MultiLabelBinarizer()
    var2_df = pd.DataFrame(mlb.fit_transform(dataset['var2']),columns=mlb.classes_)
    return var2_df

train_var2_df = multilabel_var2(train)
test_var2_df = multilabel_var2(test)
train = pd.concat([train, train_var2_df], axis=1)
test = pd.concat([test, test_var2_df], axis=1)
train = train.drop(['datetime', 'var2'], axis=1)
test = test.drop(['datetime', 'var2'], axis=1)
plt.figure(figsize=(10, 10))
sns.heatmap(train.corr(), vmin=-1, center=0, vmax=1, square=True)
plt.show()
# def box_cox_transform(dataset, var):
#     v = dataset[var]
#     pt = PowerTransformer(method='box-cox') 
#     pt.fit(v.to_frame())
#     name = "t_" + var
#     dataset[name] = pt.transform(v.to_frame())
#     return pt
    
# y_pt = box_cox_transform(train, 'electricity_consumption') 
# ws_train_pt = box_cox_transform(train, 'windspeed')
# ws_test_pt = box_cox_transform(test, 'windspeed') 
train_groups = train.groupby(['month','year', 'ID']).size().reset_index().sort_values(by='ID')
train_groups = train_groups[['month', 'year']].drop_duplicates()

test_groups = test.groupby(['month','year', 'ID']).size().reset_index().sort_values(by='ID')
test_groups = test_groups[['month', 'year']].drop_duplicates()
all_predictions = []
frames = []

for i in range(len(train_groups)):
        month = train_groups.iloc[i].month
        year = train_groups.iloc[i].year
        
        _train_chunk = train.query('year == {0}'.format(year)).query('month == {0}'.format(month))
        frames.append(_train_chunk)
        concat_train = pd.concat(frames)
        
        _test_chunk =  test.query('year == {0}'.format(year)).query('month == {0}'.format(month))
        
        print("query_parameters: month: {0} year: {1}".format(month, year))
        print("train_shape: ", concat_train.shape, "test_shape: ", _test_chunk.shape)
        
        X_train = concat_train.drop(['ID', 'electricity_consumption'], axis=1)
        y_train = concat_train['electricity_consumption'].values
        X_test = _test_chunk.drop(['ID'], axis=1)
        
        X_train = X_train[['temperature', 'var1', 'pressure', 'windspeed', 'day', 'month', 'year','hour', 'day_of_week', 'week', 'A', 'B', 'C']]
        X_test = X_test[['temperature', 'var1', 'pressure', 'windspeed', 'day', 'month', 'year','hour', 'day_of_week', 'week', 'A', 'B', 'C']]
      
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 890)

        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_val = lgb.Dataset(X_val, y_val)
        
        evals_result = {} 

        params = {
                'task': 'train',
                'objective': 'gamma',
                'metric' : 'rmse',
                'boosting': 'gbdt',
                'learning_rate': 0.001,
                'num_leaves': 100,
                'bagging_fraction': 0.85,
                'bagging_freq': 1,
                'bagging_seed': 1,
                'feature_fraction': 0.9,
                'feature_fraction_seed': 1,
                'max_bin': 256,
                'n_estimators': 10000,
            }
        
        cv_results = lgb.cv(params, lgb_train, num_boost_round = 10000, nfold = 5, early_stopping_rounds = 100, verbose_eval = 10000, seed = 50)
        lgbm_model = lgb.train(params, train_set = lgb_train, valid_sets = lgb_val, evals_result = evals_result, verbose_eval = 10000)
        
        print("train_chunk: ", len(frames))
        print("Train best score: " + str(min(evals_result['valid_0']['rmse'])))
        
        predictions = lgbm_model.predict(X_test)
        all_predictions.append(predictions)
        
        # todo: feed X_test + prediction to train to get full month

concat_predictions = np.concatenate(all_predictions, axis=None)

ax = sns.distplot(concat_predictions)
plt.show()

# Writing output to file
subm = pd.DataFrame()
subm['ID'] = test['ID']
subm['electricity_consumption'] = concat_predictions

subm.to_csv("/kaggle/working/" + 'submission.csv', index=False)
subm