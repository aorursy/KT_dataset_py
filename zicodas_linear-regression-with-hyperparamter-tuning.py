# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import all libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

import re



import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import scale

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline



import warnings # supress warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('/kaggle/input/house-price-prediction-challenge/train.csv')

df_test = pd.read_csv('/kaggle/input/house-price-prediction-challenge/test.csv')

y_test = pd.read_csv('/kaggle/input/house-price-prediction-challenge/sample_submission.csv')
#Shape of Data

df_train.shape,df_test.shape,y_test.shape
df_train.head()
df_test.head()
#Below Addresses and Latitude and longitude have no significance in the predictions

df_train.drop('ADDRESS',axis=1,inplace=True)

df_test.drop('ADDRESS',axis=1,inplace=True)

df_train.drop(['LONGITUDE','LATITUDE'],axis=1,inplace=True)

df_test.drop(['LONGITUDE','LATITUDE'],axis=1,inplace=True)
df_train.loc[df_train['UNDER_CONSTRUCTION'] == 0, 'UNDER_CONSTRUCTION'] = 'NO'

df_train.loc[df_train['UNDER_CONSTRUCTION'] == 1, 'UNDER_CONSTRUCTION'] = 'YES'



df_test.loc[df_test['UNDER_CONSTRUCTION'] == 0, 'UNDER_CONSTRUCTION'] = 'NO'

df_test.loc[df_test['UNDER_CONSTRUCTION'] == 1, 'UNDER_CONSTRUCTION'] = 'YES'



df_train.loc[df_train['RERA'] == 0, 'RERA'] = 'NO'

df_train.loc[df_train['RERA'] == 1, 'RERA'] = 'YES'



df_test.loc[df_test['RERA'] == 0, 'RERA'] = 'NO'

df_test.loc[df_test['RERA'] == 1, 'RERA'] = 'YES'



df_train.loc[df_train['READY_TO_MOVE'] == 0, 'READY_TO_MOVE'] = 'NO'

df_train.loc[df_train['READY_TO_MOVE'] == 1, 'READY_TO_MOVE'] = 'YES'



df_test.loc[df_test['READY_TO_MOVE'] == 0, 'READY_TO_MOVE'] = 'NO'

df_test.loc[df_test['READY_TO_MOVE'] == 1, 'READY_TO_MOVE'] = 'YES'



df_train.loc[df_train['RESALE'] == 0, 'RESALE'] = 'NO'

df_train.loc[df_train['RESALE'] == 1, 'RESALE'] = 'YES'



df_test.loc[df_test['RESALE'] == 0, 'RESALE'] = 'NO'

df_test.loc[df_test['RESALE'] == 1, 'RESALE'] = 'YES'
X_train = df_train.iloc[:,:-1]

y_train = df_train['TARGET(PRICE_IN_LACS)']

X_test = df_test
# creating dummy variables for categorical variable

X_train_categorical = X_train.select_dtypes(include=['object'])

#X_train_categorical.head()

X_test_categorical = X_test.select_dtypes(include=['object'])

#X_test_categorical.head()
# convert into dummies

X_train_dummies = pd.get_dummies(X_train_categorical, drop_first=True)

#X_train_dummies.head()

X_test_dummies = pd.get_dummies(X_test_categorical, drop_first=True)

#X_test_dummies.head()
# drop categorical variables 

X_train = X_train.drop(list(X_train_categorical.columns), axis=1)

X_test = X_test.drop(list(X_test_categorical.columns), axis=1)
# concat dummy variables with X

X_train = pd.concat([X_train, X_train_dummies], axis=1)

X_test = pd.concat([X_test, X_test_dummies], axis=1)
# rescale the features

cols = X_train.columns

X_train = pd.DataFrame(scale(X_train))

X_train.columns = cols
# rescale the features

cols = X_test.columns

X_test = pd.DataFrame(scale(X_test))

X_test.columns = cols
# number of features

len(X_train.columns),len(X_test.columns)
# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

# specify range of hyperparameters

hyper_params = [{'n_features_to_select': list(range(2,20))}]



# specify model

lm = LinearRegression()

lm.fit(X_train, y_train)

rfe = RFE(lm)



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = rfe, 

                        param_grid = hyper_params, 

                        scoring= 'r2', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)



# fit the model

model_cv.fit(X_train, y_train)
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results.head()
# plotting cv results

plt.figure(figsize=(16,6))



plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])

plt.xlabel('number of features')

plt.ylabel('r-squared')

plt.title("Optimal Number of Features")

plt.legend(['test score', 'train score'], loc='upper left')
predictions = model_cv.predict(X_test)
from sklearn.metrics import mean_squared_log_error

np.sqrt(mean_squared_log_error(y_test, abs(predictions)))
df_submission = pd.DataFrame(predictions)

df_submission.to_csv('Submissions.csv')