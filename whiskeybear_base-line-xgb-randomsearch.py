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
# Set the current working directory for easier path calls

os.chdir('/kaggle/input/house-prices-advanced-regression-techniques')



# Load the training data

raw_data = pd.read_csv('train.csv')

test_data = pd.read_csv('test.csv')

raw_data.head()
print(f"Amount of columns = {len(list(raw_data))}\nAmount of rows = {len(raw_data)}\nSo our dataset has dimensions 1460x81")
# But first, let's stick the training and test data together so we can investigate and manipulate the data more easily. Also, let's set the target and IDs aside for now.

target = raw_data['SalePrice']

test_Id = test_data['Id']



# Let's remember where to split

raw_len, test_len = len(raw_data), len(test_data)



raw_data = raw_data.drop(['SalePrice'], axis=1)

raw_data = pd.concat([raw_data, test_data], axis=0)

raw_data = raw_data.drop(['Id'], axis=1)
# Counts the NaN types in a feature column and prints the number if it's not 0. (Also collects the really messy features in a list, and the less messy in another.)

messy_features = []

less_messy_features = []

for feature in list(raw_data):

    nan_count = raw_data[feature].isna().sum()

    if nan_count:

        print(f"{feature} contains, {nan_count} NaN values")

        if nan_count > 0.40 * len(raw_data):

            messy_features.append(feature)

        else:

            less_messy_features.append(feature)
print(f"Here are the extremely messy feature we collected:\n{messy_features}\nLet's remove them for now.")
less_messy_data = raw_data.drop(messy_features, axis=1)

n_features = len(list(less_messy_data))

print(f"We removed 5 features so our training data now has dimension:\n1460x{n_features}")
less_messy_data[less_messy_features].head(20)
# There are some features that appear numerical, but are actually categorical, let's change them before going further.

# less_messy_data['MSSubClass'] = less_messy_data['MSSubClass'].apply(str)

# less_messy_data['OverallCond'] = less_messy_data['OverallCond'].astype(str)

# less_messy_data['YrSold'] = less_messy_data['YrSold'].astype(str)

# less_messy_data['MoSold'] = less_messy_data['MoSold'].astype(str)
# Two collections of categorical and numerical features, respectively.

first_elements = [less_messy_data[i].iloc[0] for i in list(less_messy_data)]

categorical_features = []

numerical_features = []

for i in range(len(first_elements)):

    if type(first_elements[i]) == str:

        categorical_features.append(list(less_messy_data)[i])

    else:

        numerical_features.append(list(less_messy_data)[i])
for feature in categorical_features:

    most_occurring_category = less_messy_data[feature].value_counts().index[0]

    less_messy_data[feature] = less_messy_data[feature].fillna(most_occurring_category)

    

less_messy_data.head(20)
for feature in numerical_features:

    feature_mean = less_messy_data[feature].mean()

    less_messy_data[feature] = less_messy_data[feature].fillna(np.floor(feature_mean))

    

less_messy_data.head(20)
dummies = []

for cat in categorical_features:

    dummies.append(pd.get_dummies(less_messy_data[cat]))

    less_messy_data = less_messy_data.drop([cat], axis=1)

    

n_dummies = len(dummies)

dummies = [less_messy_data] + dummies

less_messy_data = pd.concat(dummies, axis=1)

    

less_messy_data.head(5)
# Let's just make sure that the one-hot column names are unique

columns = list(less_messy_data)

add_num = 1

for _ in range(10):

    for i in range(1, len(columns)):

        if columns[i] in columns[:i]:

            columns[i] = columns[i] + str(add_num)

    add_num += 1



less_messy_data.columns = columns
print(f"We now have {len(list(less_messy_data))} because of the one-hot encoding")
from sklearn.preprocessing import MinMaxScaler

remaining_numerical_features = list(less_messy_data)[:(n_features - n_dummies)]

scaler = MinMaxScaler()

scaled = scaler.fit_transform(less_messy_data[remaining_numerical_features])

less_messy_data[remaining_numerical_features] = scaled

less_messy_data.head(5)
test_data = less_messy_data.iloc[raw_len:, :]

less_messy_data = less_messy_data.iloc[:raw_len, :]
import xgboost as xgb

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

from scipy import stats

from sklearn.model_selection import RandomizedSearchCV



test_parameters = {'n_estimators': [150, 300, 450, 600, 750, 1000],

                  'learning_rate': [0.01, 0.07, 0.1, 0.25, 0.4, 0.6, 1],

                  'subsample': [0.3, 0.5, 0.7, 0.9],

                  'max_depth': [3, 4, 5, 6, 7, 8, 9],

                  'colsample_bytree': [0.5, 0.7, 0.9],

                  'min_child_weight': [1, 2, 3, 4]

                  }



xgb_randsearch = RandomizedSearchCV(XGBRegressor(), 

                                    param_distributions = test_parameters, 

                                    n_iter = 100, 

                                    scoring='neg_mean_squared_error',

                                    verbose = 3,

                                    n_jobs = -1)



xgb_randsearch.fit(less_messy_data, target)
# XGB

prediction = xgb_randsearch.predict(test_data)

prediction = pd.DataFrame(prediction, columns=['SalePrice'])

submission = np.zeros((len(prediction), 2))

submission[:, 0] = test_Id.values.flatten()

submission[:, 1] = prediction.values.flatten()

submission = pd.DataFrame(submission, columns=['Id', 'SalePrice'])

submission.Id = submission.Id.astype(int)

submission.to_csv('../../../kaggle/working/base_line_submission.csv', index=False)
submission