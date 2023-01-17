# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from scipy.stats import lognorm, gamma



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample_submission = pd.read_csv("../input/ltfs-2020/sample_submission_IIzFVsf.csv")

test = pd.read_csv("../input/ltfs-2020/test_1eLl9Yf.csv")

train = pd.read_csv("../input/ltfs-2020/train_fwYjLYX.csv")



# EXTERNAL DATASETS



# holiday calendar

# crop calendar (people may take loans before sowing time)

# school/academic calendar (people may take loans for school expenses)

# Climatic Calendar (monsoon) - https://www.civilsdaily.com/indian-climate-4-the-southwest-monsoon-season-jun-sep/

# http://prasoonmathur.blogspot.com/2015/01/sowing-and-harvesting-seasons-of-indian.html

# https://nfsm.gov.in/nfmis/rpt/calenderreport.aspx
train.head()
test.head()
train = train.drop(columns=['zone', 'branch_id'])

train['application_date'] = pd.to_datetime(train['application_date'])

#note: significant improvemanet on LB after spliting date into separet dmY columns

def convert_date(dataset):

        dataset['day'] = dataset.apply(lambda row: row.application_date.day, axis=1)

        dataset['month'] = dataset.apply(lambda row: row.application_date.month, axis=1)

        dataset['year'] = dataset.apply(lambda row: row.application_date.year, axis=1)



convert_date(train)
# dropping outliers made LB score worse

# t = train.query('case_count <= 12000')

# train = t

plt.scatter(train['case_count'], train['year'])

plt.show()
plt.scatter(train['case_count'], train['day'])

plt.show()
ax = sns.distplot(train['case_count']/1000, fit=gamma)

ax.autoscale()

plt.show()
def day(row):

    p = pd.Period(row.application_date.date(), freq='H')

    return p.dayofyear



def apply_day_of_year(dataset):

    dataset['day_of_year'] = dataset.apply(lambda row: day(row), axis=1)



apply_day_of_year(train)
def encode_climatic_calendar(row):

    if row['month'] in [1, 2]:

        # 0 - winter seson

        return 0

    if row['month'] in [3, 4, 5]:

        # 1 - pre-monsoon

        return 1

    if row['month'] in [6, 7, 8, 9]:

        # 2 - southwest monsoon

        return 2

    if row['month'] in [10, 11, 12]:

        # 3 - post monsoon

        return 3

    

def apply_climatic_calendar(dataset):

    dataset['climatic_calendar'] = dataset.apply(lambda row: encode_climatic_calendar(row), axis=1)

    

# apply_climatic_calendar(train)
train.head()
# https://en.wikipedia.org/wiki/Public_holidays_in_India

# National holidays

def is_national_day(row):

    if row['day'] in [26, 15, 2] and row['month'] in [1, 8, 10]:

        # 1 - holiday

        return 1

    else:

        return 0

    

def apply_is_national_day(dataset):

    dataset['is_national_day'] = dataset.apply(lambda row: is_national_day(row), axis=1)

    

# apply_is_national_day(train)
# https://en.wikipedia.org/wiki/Public_holidays_in_India

# Traditional holidays

def is_holiday(row):

    if row['day'] in [1, 14, 7] and row['month'] in [1, 4, 5, 11, 12]:

        # 1 - holiday

        return 1

    else:

        return 0

    

def apply_is_holiday(dataset):

    dataset['is_holiday'] = dataset.apply(lambda row: is_holiday(row), axis=1)

    

apply_is_holiday(train)
from PIL import Image

# AP: Andhra Pradesh

# MP: Madhya Pradesh

# Raj: Rajasthan

# UP: Uttar Pradesh

# MH: Maharashtra

# Guj: Gujarat

# Kar: Karnataka

# Ker: Kerala 

# TN: Tamil Nadu 

# HP: Himachal Pradesh

# Har: Haryana 

# WB: West Benga

# Bir: Bihar

# Punj: Punjab 

# Or: Orissa

im = Image.open('../input/season-calendar/season calendar.jpg')

im
state = train['state'].unique()

state = [x.lower().replace(" ", "_") for x in state]

print("state count: ", len(state))

def is_sowing(row, months):

    if row['month'] in months:

        # 1 - sowing

        return 1

    else:

        return 0

    

def apply_is_sowing_time(dataset):

    

    wb_months = [6, 7, 8, 9, 10, 11]

    dataset['west_bengal'] = dataset.apply(lambda row: is_sowing(row, wb_months), axis=1)

    

    up_months = [2, 3, 6, 7, 8, 9, 10, 11, 12]

    dataset['uttar_pradesh'] = dataset.apply(lambda row: is_sowing(row, up_months), axis=1)

    

    raj_months = [6, 7, 8, 10, 11]

    dataset['rajasthan'] = dataset.apply(lambda row: is_sowing(row, raj_months), axis=1)

    

    kar_months = [6, 7, 8, 9, 10]

    dataset['karnataka'] = dataset.apply(lambda row: is_sowing(row, kar_months), axis=1)

    

    # sugarcane 

    mh_months = [7, 1, 2, 3] 

    dataset['maharashtra'] = dataset.apply(lambda row: is_sowing(row, mh_months), axis=1)

     

    bir_months = [9, 10, 11] 

    dataset['bihar'] = dataset.apply(lambda row: is_sowing(row, bir_months), axis=1)

    

    ker_months = [9, 10] 

    dataset['kerala'] = dataset.apply(lambda row: is_sowing(row, ker_months), axis=1)

    

# apply_is_sowing_time(train)
# train = train.groupby(['application_date', 'segment','day', 'month', 'year', 'day_of_year', 'climatic_calendar','is_national_day', 'is_holiday', 'west_bengal', 'uttar_pradesh', 'rajasthan', 'karnataka', 'maharashtra', 'bihar', 'kerala']).sum().reset_index()

train = train.groupby(['application_date', 'segment','day', 'month', 'year', 'day_of_year', 'is_holiday']).sum().reset_index()

train = train.sort_values(['application_date', 'segment'])
train.query('month == 1')
train = train.drop(columns=['application_date'])

X_train = train.drop(['case_count'], axis=1)

y_train = train['case_count'].values



X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train)



X_test = test.drop(['id'], axis=1)

test_date = test['application_date'].values

X_test['application_date'] = pd.to_datetime(test['application_date'])



convert_date(X_test)

apply_day_of_year(X_test)

# apply_climatic_calendar(X_test)

# apply_is_national_day(X_test)

apply_is_holiday(X_test)

# apply_is_sowing_time(X_test)

# apply_holi_diwali(X_test)

 

X_test = X_test.drop(columns=['application_date'])

X_test.head()



lgb_train = lgb.Dataset(X_tr, y_tr)

lgb_val = lgb.Dataset(X_val, y_val)
params = {

        'objective': 'gamma',

        'boosting': 'gbdt',

        'learning_rate': 0.05 ,

        'verbose': 0,

        'num_leaves': 100,

        'bagging_fraction': 0.95,

        'bagging_freq': 1,

        'bagging_seed': 1,

        'feature_fraction': 0.9,

        'feature_fraction_seed': 1,

        'max_bin': 256,

        'metric' : 'mape',

        'n_estimators': 250,

    }



cv_results = lgb.cv(params, lgb_train, num_boost_round=1000, nfold=2, metrics='mape', early_stopping_rounds=100, seed=50)

print("CV best score: " + str(min(cv_results['mape-mean'])))

lgbm_model = lgb.train(params, train_set = lgb_train, valid_sets = lgb_val, verbose_eval=5)
predictions = lgbm_model.predict(X_test)

# Writing output to file

subm = pd.DataFrame()

subm['id'] = test['id']

subm['application_date'] = test_date

subm['segment'] = X_test['segment']

subm['case_count'] = predictions



subm.to_csv("/kaggle/working/" + 'submission.csv', index=False, float_format = '%.5f')

subm