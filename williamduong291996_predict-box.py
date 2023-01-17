# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

import re 

import lightgbm as lgb

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/train.csv')

# explore basic information from the train dataset

train_data.head()
train_data.info()
# train_data.original_language.value_counts
# convert date time into numeric format

train_data.spoken_languages.iloc[13]
import ast

def extract_language(row) :

    if isinstance(row, str):

        return list(map(lambda d: list(d.values())[0], ast.literal_eval(row)))

    else:

        return []

# find total language for each movie

train_data['total_languages'] = train_data.spoken_languages.apply(extract_language)

test_data['total_languages'] = test_data.spoken_languages.apply(extract_language)
train_data['total_languages'].head()
train_data['nb_total_languages'] = train_data['total_languages'].apply(len)

test_data['nb_total_languages'] = test_data['total_languages'].apply(len)
train_data['nb_total_languages'].value_counts()


# First look at correlation between revenue and other numeric attribute 

features = ["budget","popularity","runtime","revenue","nb_total_languages"]

sns.pairplot(train_data[features].dropna())
features = ["budget","popularity","runtime","nb_total_languages"]

X = train_data[features]

y = train_data.revenue.apply(np.log10)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=30, shuffle=True)
params = {'objective':'regression',

          'num_leaves' : 40,

          'min_data_in_leaf' : 20,

          'max_depth' : 4,

          'learning_rate': 0.01,

          "metric": 'rmse',

          "random_state" : 42,

          "verbosity": -1}
# create dataset for lightgbm

lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
gbm = lgb.train(params,

                lgb_train,

                num_boost_round=5000,

                valid_sets=lgb_eval,

                early_stopping_rounds=5)
# missing_crew = data[data.crew.isnull()]

# # missing_crew['revenue'].plot(kind='hist')

# missing_cast = data[data.cast.isnull()]

# # missing_cast['revenue'].plot(kind='hist')

# missing_cast.head(2)
X_test = test_data[features]

y_pred = 10**gbm.predict(X_test)
pd.DataFrame({'actual_revenue': test_data.revenue, 'revenue': y_pred})