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
import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns 

df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
# Check if any null values 

df.columns[df.isnull().any()]
sns.distplot(df['quality'])
# correlation of different input features with target variable

correlations = df.corr()['quality']
correlations
sns.heatmap(df.corr())
def extract_features(threshold):

    abs_correlation = correlations.abs()

    high_correlation = abs_correlation[abs_correlation>threshold].index.values.tolist()

    return high_correlation
# Extracting features with threshold 0.05

features = extract_features(0.05)

features.remove('quality')

features
X = df[features]

y = df['quality']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
# fitting linear regression to our dataset

regressor = LinearRegression()

regressor.fit(X_train, y_train)
# Coefficients of the 10 input variables

regressor.coef_
# Making predictions using our linear regression model 

y_pred_train = regressor.predict(X_train)

y_pred_test = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error
# train RMSE 

train_rmse = mean_squared_error(y_pred_train, y_train) ** 0.5

train_rmse
# test RMSE 

test_rmse = mean_squared_error(y_pred_test, y_test) ** 0.5

test_rmse