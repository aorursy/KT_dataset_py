# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight') 
%matplotlib inline


import warnings
import datetime
import calendar
from datetime import time
from dateutil.relativedelta import relativedelta

# to ignore future warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Any results you write to the current directory are saved as output.
merchants = pd.read_csv('../input/merchants.csv')
historical_trans = pd.read_csv('../input/historical_transactions.csv')
new_merchants = pd.read_csv('../input/new_merchant_transactions.csv')
data_dict = pd.read_excel('../input/Data_Dictionary.xlsx')
train.head()
test.head()
merchants.head()
new_merchants.head()
data_dict.head(10)
historical_trans.head()
print('train Data shape: {}'.format(train.shape))
print('test Data shape: {}'.format(test.shape))
print('merchants data shape: {}'.format(merchants.shape))
print('New merchants data shape: {}'.format(new_merchants.shape))
print('historical data shape: {}'.format(historical_trans.shape))
print('Data dictionary data shape: {}'.format(data_dict.shape))
sns.distplot(train.target.values, kde = False)
sns.violinplot(x = "target", data=train)
# Target value outliers
print(f'There are: {train[train.target <- 10].target.shape[0]} outliers')
train.describe()
train.info()
# Lets change the First_active_month column from string to date.

train['first_active_month'] = pd.to_datetime(train['first_active_month'])

# Also do this to the test dataset

test['first_active_month'] = pd.to_datetime(test['first_active_month'])
train_cat_feats = train.dtypes[train.dtypes == 'object'].index
train_num_feats = train.dtypes[train.dtypes != 'object'].index
print('train Categorical features {}'.format(len(train_cat_feats)))
print('train Numerical features {}'.format(len(train_num_feats)))
sns.set()
sns.pairplot(train, kind='scatter', size = 2, diag_kind='kde')
plt.show()
plt.figure(figsize = (7,5))
plt.scatter(x= train.feature_1, y = train.target)
plt.title('Feature_1', size = 15)

plt.figure(figsize = (7,5))
plt.scatter(x = train.feature_2, y = train.target)
plt.title('Feature_2')

plt.figure(figsize = (7,5))
plt.scatter(x = train.feature_3, y = train.target)
plt.title('Feature_3')
train.drop(train[train['target'] < -10].index, inplace = True)
# train.drop(train[train['feature_2'] < -10].index, inplace = True)
# train.drop(train[train['feature_3'] < -10].index, inplace = True)
plt.figure(figsize = (7,5))
plt.scatter(x= train.feature_1, y = train.target)
plt.title('Feature_1', size = 15)

plt.figure(figsize = (7,5))
plt.scatter(x = train.feature_2, y = train.target)
plt.title('Feature_2')

plt.figure(figsize = (7,5))
plt.scatter(x = train.feature_3, y = train.target)
plt.title('Feature_3')
correlation = train.corr()
correlation
def missing_values(df):
    total_miss = df.isnull().sum()
    perc_miss = 100 * total_miss / len(df)
    table_miss = pd.concat([total_miss, perc_miss], axis = 1)
    ren_table = table_miss.rename(columns = {0:'Total Miss Values', 1: '% of miss values'})
    ren_table = ren_table[ren_table.iloc[:,1]!=0].sort_values('% of miss values', ascending = False).round(2)
    
    print('You data contains {}'.format(df.shape[1]) + ' columns and has {}'.format(ren_table.shape[0]) + ' colums with missing values' )
    
    return ren_table
missing_values(train)
train = train[train.target > -10]
sns.pairplot(train[['feature_1', 'feature_2', 'feature_3', 'target']])
f, axes = plt.subplots(1,figsize = (12,6))
sns.distplot(train.feature_1, ax = axes, kde = False, color = 'green', bins = 10).set_title('Train data')
axes.set(ylabel = 'Card Counts')
axes.set_xticks(np.arange(1,6,1))
plt.show()
f, axes = plt.subplots(1,figsize = (12,6))
sns.distplot(train.feature_2, ax = axes, kde = False, color = 'green', bins = 10).set_title('Train data')
axes.set(ylabel = 'Card Counts')
axes.set_xticks(np.arange(1,6,1))
plt.show()
f, axes = plt.subplots(1,figsize = (12,6))
sns.distplot(train.feature_3, ax = axes, kde = False, color = 'green', bins = 10).set_title('Train data')
axes.set(ylabel = 'Card Counts')
axes.set_xticks(np.arange(1,6,1))
plt.show()
# f, axes = plt.subplots(1,figsize = (12,6))
# sns.distplot(train.first_active_month, ax = axes, color = 'green', kde = False, bins = 10).set_title('Train data')
# axes.set(ylabel = 'Card Counts')
# axes.set_xticks(np.arange(1,6,1))
# plt.show()
train.groupby('first_active_month').count()['card_id'].plot(figsize = (15,5), title = 'Count of First Active Month in Train set', color = 'r')
plt.show()
train_cat_feats = train.dtypes[train.dtypes == 'object'].index
train_num_feats = train.dtypes[train.dtypes != 'object'].index

print('Train categories: {}'.format(len(train_cat_feats)))
print('Train Numerical values: {}'.format(len(train_num_feats)))
for col in train_cat_feats:
    cols = train[col].value_counts().index.shape[0]
    print(f'{cols} \t unique values in \t {col}')
for col in train_num_feats:
    cols = train[col].value_counts().index.shape[0]
    print(f'{cols} \t unique values in \t {col}')
target = train.target
target.head()
merchants.describe()
missing_values(merchants)
merchants.info()
merchant_cat_feats = merchants.dtypes[merchants.dtypes == 'object'].index
merchant_cat_feats
merchant_num_feats = merchants.dtypes[merchants.dtypes != 'object'].index
merchant_num_feats
print('Merchant categorical features {}'.format(len(merchant_cat_feats)))
print('Merchant Numerical features {}'.format(len(merchant_num_feats)))
for col in merchant_cat_feats:
    cols = merchants[col].value_counts().index.shape[0]
    print(f'{cols} \t unique values in \t {col}')
for col in merchant_num_feats:
    cols = merchants[col].value_counts().index.shape[0]
    print(f'{cols} \t unique values in \t {col}')

historical_trans.head()
historical_trans.describe()
historical_trans.info()
missing_values(historical_trans)
hist_cat_feats = historical_trans.dtypes[historical_trans.dtypes == 'object'].index
hist_num_feats = historical_trans.dtypes[historical_trans.dtypes != 'object'].index

print('Historical Transactions categorical features {}'.format(len(hist_cat_feats)))
print('Historical Transactions Numerical features {}'.format(len(hist_num_feats)))
for col in hist_cat_feats:
    cols = historical_trans[col].value_counts().index.shape[0]
    print(f'{cols} \t unique values in \t {col}')
for col in hist_num_feats:
    cols = historical_trans[col].value_counts().index.shape[0]
    print(f'{cols} \t unique values in \t {col}')

new_merchants.head()
new_merchants.describe()
new_merchants.info()
missing_values(new_merchants)
new_merch_cat_feats = new_merchants.dtypes[new_merchants.dtypes == 'object'].index
new_merch_num_feats = new_merchants.dtypes[new_merchants.dtypes != 'object'].index

print('New Merchant categorical features {}'.format(len(new_merch_cat_feats)))
print('New Merchant numerical features {}'.format(len(new_merch_num_feats)))
for col in new_merch_cat_feats:
    cols = new_merchants[col].value_counts().index.shape[0]
    print(f'{cols} \t unique values in \t {col}')
for col in new_merch_num_feats:
    cols = new_merchants[col].value_counts().index.shape[0]
    print(f'{cols} \t unique values in \t {col}')


print(f'Merchants data shape: {merchants.shape}, new merchants data shape: {new_merchants.shape}')
print(merchants.columns)
print('=='*18)
print(new_merchants.columns)
# Lets merge our datasets
# But first we need to delete duplicate columns
todrop = ['merchant_category_id', 'subsector_id', 'city_id', 'state_id', 'category_2', 'category_1']

for col in todrop:
    merchants = merchants.drop(col, axis = 1)
    
merchants_data = pd.merge(merchants, new_merchants, on = 'merchant_id')
merchants_data[:5]
# Lets merge our merchant data with the training data.

data = pd.merge(merchants_data, train, on = 'card_id')
data.shape
data[:5]
data['purchase_time'] = data['purchase_date'].str.split(' ')
data['purchase_date'] = data['purchase_time'].str[0]
data['purchase_time'] = data['purchase_time'].str[1]
# Let's also do what we have done to the testing dataset
test_data = pd.merge(merchants_data, test, on = 'card_id')
test_data['purchase_time'] = test_data['purchase_date'].str.split(' ')
test_data['purchase_date'] = test_data['purchase_time'].str[0]
test_data['purchase_time'] = test_data['purchase_time'].str[1]
print(f'Train data shape: {data.shape} Test data shape: {test_data.shape}')
test_data[:5]
#We were not suppose to have converted these to date at first But now we fix that so that we can easily manipulate them
data['first_active_month'] = data['first_active_month'].astype(str)
test_data['first_active_month'] = test_data['first_active_month'].astype(str)
data.info()
# The main business with the purchase_date and purchase_time. Lets see if we can't generate more columns out of them
def get_weekday(date_string):
    date = datetime.datetime.strptime(date_string, '%Y-%m-%d')
    return calendar.day_name[date.weekday()]

# get weekday for date variable
data['purchase_weekday'] = data['purchase_date'].apply(lambda x: get_weekday(x))

# for plotting recode to ordered categorical
day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
data['purchase_weekday'] = pd.Categorical(data['purchase_weekday'], categories = day_labels, 
                                          ordered = True)

def get_month(date_string, kind = 'month'):
    if kind == 'month':
        date = datetime.datetime.strptime(date_string, '%Y-%m-%d')
    elif kind == 'day':
        date = datetime.datetime.strptime(date_string, '%Y-%m-%d')
    return date.strftime("%B")

# Creating new columns and their values

data['purchase_month'] = data['purchase_date'].apply(lambda x: get_month(x, kind = 'day'))
data['first_active_month2'] = data['first_active_month'].apply(lambda x: get_month(x))
data['first_active_year'] = data['first_active_month'].str[:4]

month_labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                'September', 'October', 'November', 'December']
data['purchase_month'] = pd.Categorical(data['purchase_month'], categories = month_labels, 
                                          ordered = True)
data['first_active_month2'] = pd.Categorical(data['first_active_month2'], categories = month_labels, 
                                          ordered = True)

year_labels = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']
data['first_active_year'] = pd.Categorical(data['first_active_year'], categories = year_labels, 
                                          ordered = True)

# get time of the day
data['temp'] = data['purchase_time'].str.split(':')

def get_session(time_list):
    time_list[0] = int(time_list[0])
    if time_list[0] > 4 and time_list[0] < 12:
        return 'Morning'
    elif time_list[0] >= 12 and time_list[0] < 17:
        return 'Afternoon'
    elif time_list[0] >= 17 and time_list[0] < 21:
        return 'Evening'
    else:
        return 'Night'
    
data['purchase_session'] = data['temp'].apply(lambda x: get_session(x))

session_labels = ['Morning', 'Afternoon', 'Evening', 'Night']
data['purchase_session'] = pd.Categorical(data['purchase_session'], categories = session_labels, 
                                          ordered = True)
## time of month
# as categorical variable, thressholds are arbitrary and could be different
def get_time_of_month_cat(date):
    date_temp = date.split('-')
    if int(date_temp[2]) < 10:
        time_of_month = 'Beginning'
    elif int(date_temp[2]) >= 10 and int(date_temp[2]) < 20:
        time_of_month = 'Middle'
    else:
        time_of_month = 'End'
    return time_of_month

data['time_of_month_cat'] = data['purchase_date'].apply(lambda x: get_time_of_month_cat(x))

tof_labels = ['Beginning', 'Middle', 'End']
data['time_of_month_cat'] = pd.Categorical(data['time_of_month_cat'], categories = tof_labels, 
                                           ordered = True)

data['time_of_month_num'] = data['purchase_date'].str[8:].astype(int)
ax = sns.lineplot(x = "purchase_month", y = "target", 
                  markers = True, dashes = False, data = data)
plt.xticks(rotation = 45)
ax.set_title('Target Variable Changes over Purchase Month')
ax.set_xlabel('Purchase Month')
ax = sns.lineplot(x = "first_active_month2", y = "target", 
                  markers = True, dashes = False, data = data)
plt.xticks(rotation = 45)
ax.set_title('Target Variable Changes over the First Active Month')
ax.set_xlabel('First Active Month')
print(f'Train data shape: {data.shape} , Test data shape: {test_data.shape}')
# The main business with the purchase_date and purchase_time. Lets see if we can't generate more columns out of them
def get_weekday(date_string):
    date = datetime.datetime.strptime(date_string, '%Y-%m-%d')
    return calendar.day_name[date.weekday()]

# get weekday for date variable
test_data['purchase_weekday'] = test_data['purchase_date'].apply(lambda x: get_weekday(x))

# for plotting recode to ordered categorical
day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
test_data['purchase_weekday'] = pd.Categorical(test_data['purchase_weekday'], categories = day_labels, ordered = True)

def get_month(date_string, kind = 'month'):
    if kind == 'month':
        date = datetime.datetime.strptime(date_string, '%Y-%m-%d')
    elif kind == 'day':
        date = datetime.datetime.strptime(date_string, '%Y-%m-%d')
    return date.strftime("%B")

test_data['purchase_month'] = test_data['purchase_date'].apply(lambda x: get_month(x, kind = 'day'))
test_data['first_active_month2'] = test_data['first_active_month'].apply(lambda x: get_month(x))
test_data['first_active_year'] = test_data['first_active_month'].str[:4]

month_labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                'September', 'October', 'November', 'December']
test_data['purchase_month'] = pd.Categorical(test_data['purchase_month'], categories = month_labels, ordered = True)
test_data['first_active_month2'] = pd.Categorical(test_data['first_active_month2'], categories = month_labels, ordered = True)

year_labels = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']
test_data['first_active_year'] = pd.Categorical(test_data['first_active_year'], categories = year_labels, ordered = True)

# get time of the day
test_data['temp'] = test_data['purchase_time'].str.split(':')

def get_session(time_list):
    time_list[0] = int(time_list[0])
    if time_list[0] > 4 and time_list[0] < 12:
        return 'Morning'
    elif time_list[0] >= 12 and time_list[0] < 17:
        return 'Afternoon'
    elif time_list[0] >= 17 and time_list[0] < 21:
        return 'Evening'
    else:
        return 'Night'
    
test_data['purchase_session'] = test_data['temp'].apply(lambda x: get_session(x))

session_labels = ['Morning', 'Afternoon', 'Evening', 'Night']
test_data['purchase_session'] = pd.Categorical(test_data['purchase_session'], categories = session_labels, ordered = True)
## time of month
# as categorical variable, thressholds are arbitrary and could be different
def get_time_of_month_cat(date):
    date_temp = date.split('-')
    if int(date_temp[2]) < 10:
        time_of_month = 'Beginning'
    elif int(date_temp[2]) >= 10 and int(date_temp[2]) < 20:
        time_of_month = 'Middle'
    else:
        time_of_month = 'End'
    return time_of_month

test_data['time_of_month_cat'] = test_data['purchase_date'].apply(lambda x: get_time_of_month_cat(x))

tof_labels = ['Beginning', 'Middle', 'End']
test_data['time_of_month_cat'] = pd.Categorical(test_data['time_of_month_cat'], categories = tof_labels, ordered = True)

test_data['time_of_month_num'] = test_data['purchase_date'].str[8:].astype(int)
# Now the shape should be the same in terms of columns except that Train data will have one extra column because of the target value
print(f'Train data shape: {data.shape} , Test data shape: {test_data.shape}')
def get_end_of_month(date):
    date_temp = date.split('-')
    if int(date_temp[2]) >= 25:
        end_of_month = 'Y'
    else:
        end_of_month = 'N'
    return end_of_month
data['end_of_month'] = data['purchase_date'].apply(lambda x: get_end_of_month(x))
test_data['end_of_month'] = test_data['purchase_date'].apply(lambda x: get_end_of_month(x))
print(f'Train data shape: {data.shape} , Test data shape: {test_data.shape}')
data[:5]

correlation_data = data.corr()
correlation_data
print(correlation_data['target'].sort_values(ascending = False))
data_cat_feats = data.dtypes[data.dtypes == 'object'].index
data_num_feats = data.dtypes[data.dtypes != 'object'].index

print(f'Data categrical: {len(data_cat_feats)}')
print(f'Data Numerical: {len(data_num_feats)}')
# for col in data_cat_feats:
#     cols = data[col].value_counts().index.shape[0]
#     print(f'{cols} \t unique values in \t {col}')
# for col in data_num_feats:
#     cols = data[col].value_counts().index.shape[0]
#     print(f'{cols} \t unique values in \t {col}')
missing_values(data.select_dtypes(['object']))
missing_values(data.select_dtypes(include = [np.number]))
missing_values(data)
data['category_3'].value_counts()
data['category_3'] = data['category_3'].fillna(data['category_3'].mode()[0])
missing_values(data.select_dtypes(['object']))
data['category_3'].value_counts()
cols = ['category_2', 'avg_sales_lag3', 'avg_sales_lag6', 'avg_sales_lag12']

for col in cols:
    data[col] = data[col].fillna(data[col].mean())
missing_values(data)
missing_values(test_data.select_dtypes(['object']))
test_data['category_3'] = test_data['category_3'].fillna(test_data['category_3'].mode()[0])
missing_values(test_data)
cols = ['category_2', 'avg_sales_lag3', 'avg_sales_lag6', 'avg_sales_lag12']

for col in cols:
    test_data[col] = test_data[col].fillna(test_data[col].mean())
missing_values(test_data)
# Lets change the First_active_month column from string to date.

data['first_active_month'] = pd.to_datetime(data['first_active_month'])

# Also do this to the test dataset

test_data['first_active_month'] = pd.to_datetime(test_data['first_active_month'])

data['purchase_date'] = pd.to_datetime(data['purchase_date'])
test_data['purchase_date'] = pd.to_datetime(test_data['purchase_date'])
data['purchase_time'] = pd.to_timedelta(data['purchase_time'])
test_data['purchase_time'] = pd.to_timedelta(test_data['purchase_time'])
data[:5]
test_data[:5]
print(f'train: {data.shape}, Test: {test_data.shape}')

cols = ['merchant_id', 'card_id', 'temp']

for col in cols:
    data.drop([col], axis = 1, inplace = True)
    test_data.drop([col], axis = 1, inplace = True)
    
print(f'train: {data.shape}, test: {test_data.shape}')
catcols = data.select_dtypes(['object'])

for col in catcols:
    print(col)
    print(data[col].value_counts())
    print('==' * 20)
data['most_recent_sales_range'] = data['most_recent_sales_range'].map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4})
data['most_recent_purchases_range'] = data['most_recent_purchases_range'].map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4})
data['category_4'] = data['category_4'].map({'Y':1, 'N':0})
data['authorized_flag'] = data['authorized_flag'].map({'Y':0})
data['category_1'] = data['category_1'].map({'Y':1, 'N':0})
data['category_3'] = data['category_3'].map({'A':0, 'B':1, 'C':2})
data['end_of_month'] = data['end_of_month'].map({'Y':1, 'N':0})


test_data['most_recent_sales_range'] = test_data['most_recent_sales_range'].map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4})
test_data['most_recent_purchases_range'] = test_data['most_recent_purchases_range'].map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4})
test_data['category_4'] = test_data['category_4'].map({'Y':1, 'N':0})
test_data['authorized_flag'] = test_data['authorized_flag'].map({'Y':0})
test_data['category_1'] = test_data['category_1'].map({'Y':1, 'N':0})
test_data['category_3'] = test_data['category_3'].map({'A':0, 'B':1, 'C':2})
test_data['end_of_month'] = test_data['end_of_month'].map({'Y':1, 'N':0})
catcols = data.select_dtypes(['object'])

for col in catcols:
    print(col)
    print(data[col].value_counts())
    print('==' * 20)
data[:5]
data.info()
data_app = pd.get_dummies(data)
test_app = pd.get_dummies(test_data)
data_app[:5]
missing_values(test_app)













from xgboost import XGBRegressor
y = data_app.target
X = data_app.drop(['target'], axis = 1)
X.shape, test_app.shape
y[:5]
xgb = XGBRegressor()
X['purchase_date'] = X['purchase_date'].astype(int)
X['first_active_month'] = X['first_active_month'].astype(int)
X['purchase_time'] = X['purchase_time'].astype(int)
test_app['purchase_date'] = test_app['purchase_date'].astype(int)
test_app['first_active_month'] = test_app['first_active_month'].astype(int)
test_app['purchase_time'] = test_app['purchase_time'].astype(int)
# xgb_model = xgb.fit(X,y)
# test, X = test.align(X, join = 'inner', axis = 1)
test, X = test.align(X, join = 'left', axis = 0)
X, test_app = X.align(test_app, join = 'left', axis = 0)
X, y = X.align(y, join = 'left', axis = 0)
test.shape, X.shape, test_app.shape, y.shape

from sklearn.linear_model import Lasso
lassoreg = Lasso(alpha = 1.0, normalize = True, max_iter=1e5)
model_lasso = lassoreg.fit(X,y)
pred_lasso = model_lasso.predict(test_app)
test['card_id'].shape
pred_lasso.shape
# pred_lasso, test['card_id'] = pred_lasso.align(test['card_id'], join = 'inner', axis = 1)


sub_lasso = pd.DataFrame({'card_id': test['card_id'], 'target': pred_lasso})
sub_lasso.to_csv('Lasso_submission.csv', index = False)
sub_lasso.head()
import lightgbm as lgb
params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "min_child_weight" : 50,
        "learning_rate" : 0.05,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
train_set = lgb.Dataset(X, label = y.values)
val_set = lgb.Dataset(X)
lgb_model = lgb.train(params, train_set,valid_sets= val_set, early_stopping_rounds=1000,verbose_eval=1000, num_boost_round=1000)
lgb_pred = lgb_model.predict(test_app)
lgb_pred
sub_lgb = pd.DataFrame({'card_id':test['card_id'], 'target': lgb_pred})
sub_lgb.to_csv('lgb_submission.csv', index = False)
sub_lgb.head()
import xgboost as xgb
dtrain = xgb.DMatrix(X, label = y.values)
dtest = xgb.DMatrix(test_app)
xgb_params = {'learning_rate':0.001, 
              'n_estimators':1000,
              'max_depth':4,
              'min_child_weight':6,
              'gamma':0,
              'subsample':0.8,
              'colsample_bytree':0.8,
              'objective':'binary:logistic',
              'nthread':4,
              'scale_pos_weight':1,
              'seed':27,
              'silent':1
              }
xgb_model = xgb.train(params = {'silent':1, 'learning_rate':0.1, 'n_estimators':1000, 'min_child_weight':5},dtrain=dtrain, verbose_eval = False, num_boost_round=100)
xgb_pred = xgb_model.predict(dtest)
xgb_pred
xgb_sub = pd.DataFrame({'card_id':test['card_id'], 'target':xgb_pred})
xgb_sub.to_csv('xgb_submission.csv', index = False)
xgb_sub.tail(20)
