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
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv('../input/house-price-prediction-challenge/train.csv')
a_test = pd.read_csv('../input/house-price-prediction-challenge/test.csv')
b_test = pd.read_csv('../input/house-price-prediction-challenge/sample_submission.csv')
data.head()
data.info()
data.describe()
from sklearn.linear_model import LinearRegression

x_train = data.drop(['POSTED_BY','BHK_OR_RK','ADDRESS','LONGITUDE','LATITUDE','TARGET(PRICE_IN_LACS)'], axis=1)
y_train = data['TARGET(PRICE_IN_LACS)']

LinearRegression().fit(x_train,y_train)
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.distplot(data['TARGET(PRICE_IN_LACS)'])

plt.subplot(2,2,2)
sns.distplot(data['SQUARE_FT'])
plt.subplot(2,2,3)
sns.distplot(data['LONGITUDE'])
plt.subplot(2,2,4)
sns.distplot(data['LATITUDE'])
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.distplot(data['TARGET(PRICE_IN_LACS)'])
plt.subplot(2,2,2)
sns.distplot(data['SQUARE_FT'])
plt.subplot(2,2,3)
sns.distplot(data['LONGITUDE'])
plt.subplot(2,2,4)
sns.distplot(data['LATITUDE'])
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import math
import xgboost as xgb
data = pd.read_csv('../input/house-price-prediction-challenge/train.csv')
test = pd.read_csv('../input/house-price-prediction-challenge/train.csv', usecols=['UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'SQUARE_FT', 'READY_TO_MOVE', 'RESALE'])
X = data[['UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'SQUARE_FT', 'READY_TO_MOVE', 'RESALE']]
X.head(5)
X.dtypes
y = data.iloc[:,-1]
y.head(5)
test.head(5)
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10, seed=123)
xg_reg.fit(X, y)
pred = xg_reg.predict(test)
pred
housing_dmatrix = xgb.DMatrix(data=X, label=y)
params = {"objective":"reg:squarederror", "max_depth":4}
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4, num_boost_round=5, metrics='rmse', as_pandas=True, seed=123)
print(cv_results)
print((cv_results["test-rmse-mean"]).tail(1))
predictions = pd.DataFrame(pred)
predictions.rename(columns={0:'TARGET(PRICE_IN_LACS)'}, inplace=True)
predictions = predictions.astype('int32')
predictions.to_csv('my_submission.csv', index=False)
sub = pd.read_csv('my_submission.csv')
sub
