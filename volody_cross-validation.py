import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# apply ignore

import warnings

warnings.filterwarnings('ignore')
#load data

train_data = pd.read_csv('../input/learn-together/train.csv')

train_data.head()
#load data

test_data = pd.read_csv('../input/learn-together/test.csv')

test_data.head()
# check missing values

print('train_data missing values = {}'.format(train_data.isnull().values.any()))

print('test_data missing values  = {}'.format(test_data.isnull().values.any()))
# No missing values, separate target from predictors

y = train_data.Cover_Type              

train_data.drop(['Cover_Type'], axis=1, inplace=True)



# Select numeric columns only

numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]

X = train_data[numeric_cols].copy()

X_test = test_data[numeric_cols].copy()
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline



model = RandomForestRegressor(n_estimators=50,random_state=70,max_depth=120)

my_pipeline = Pipeline(steps=[('model', model)])
from sklearn.model_selection import cross_val_score



# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(my_pipeline, X, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print("MAE scores:\n", scores)
def get_score(n_estimators):

    """Return the average MAE over 3 CV folds of random forest model.

    

    Keyword argument:

    n_estimators -- the number of trees in the forest

    """



    print(n_estimators)

    

    # Multiply by -1 since sklearn calculates *negative* MAE

    model = RandomForestRegressor(n_estimators=n_estimators,random_state=70,max_depth=120)

    x_pipeline = Pipeline(steps=[

        ('model', model)

    ])

    

    # n_jobs if set to -1, all CPUs are used.

    scores = -1 * cross_val_score(x_pipeline, X, y,

                              cv=3, n_jobs=-1,

                              scoring='neg_mean_absolute_error')



    return scores.mean()
results = {50*i: get_score(50*i) for i in range(1, 20)} 
import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(results.keys(), results.values())

plt.show()