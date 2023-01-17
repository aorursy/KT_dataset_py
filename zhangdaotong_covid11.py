# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import lightgbm as lgb

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

import nltk

from sklearn.preprocessing import LabelBinarizer,LabelEncoder,StandardScaler,MinMaxScaler

from sklearn.linear_model import LogisticRegression,SGDClassifier,LinearRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split

import keras

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from keras.models import Sequential

from keras.layers import Dense,LSTM

import tensorflow as tf
train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")
train['ConfirmedCases'] = train['ConfirmedCases'].apply(int)

train['Fatalities'] = train['Fatalities'].apply(int)

cases = train.ConfirmedCases

fatalities = train.Fatalities

del train['ConfirmedCases']

del train['Fatalities']
train['Province_State'] = train['Province_State'].fillna("")

test['Province_State'] = test['Province_State'].fillna("")

train['geography'] = train['Country_Region']+train['Province_State']

test['geography'] = test['Country_Region']+test['Province_State']

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

train['geography'] = le.fit_transform(train['geography'])

test['geography'] = le.fit_transform(test['geography'])
train = train.drop(['Province_State','Country_Region'],axis=1)

test = test.drop(['Province_State','Country_Region'],axis=1)
del train['Id']

del test['ForecastId']
from sklearn.preprocessing import OrdinalEncoder



def create_features(df):

    df['day'] = df['Date'].dt.day

    df['month'] = df['Date'].dt.month

   

    return df
train['Date']=pd.to_datetime(train['Date'])

test['Date']=pd.to_datetime(test['Date'])

train = create_features(train)

test = create_features(test)

train = train.drop("Date",axis =1)

test = test.drop("Date",axis =1)

train
test
#from sklearn.preprocessing import LabelBinarizer,LabelEncoder,StandardScaler,MinMaxScaler

#scaler = MinMaxScaler()

#X_train = scaler.fit_transform(train)

#X_test = scaler.transform(test)
from xgboost import XGBRegressor

model_con = XGBRegressor(n_estimators = 3000 ,random_state = 0,max_depth = 20,learning_rate = 0.01

                        ,objective = 'reg:squarederror',min_child_weight = 1,gamma = 0.7,n_thread = 4)

model_con.fit(train, cases)
#from sklearn.ensemble import RandomForestRegressor

#regr = RandomForestRegressor(n_estimators = 2100 , random_state = 0 , max_depth = 22)

#regr.fit(X_train, cases)
#gbm = lgb.LGBMRegressor(objective='regression',num_leaves=128,learning_rate=0.05,n_estimators=5000)

#gbm.fit(X_train, cases)
c = model_con.predict(test)

for i in range(len(c)):

    if c[i]<0:

        #print(c[i])

        c[i] = 0

        c[i] = int(c[i])

    c[i] = int(c[i])

        
con_pred = np.around(c,decimals = 0)
con_pred
model_fal = XGBRegressor(n_estimators = 3000 ,random_state = 0,max_depth = 20,learning_rate = 0.01

                        ,objective = 'reg:squarederror',min_child_weight = 1,gamma = 0.7,n_thread = 4)

model_fal.fit(train,fatalities )
#regr1 = RandomForestRegressor(n_estimators = 2100 , random_state = 0 , max_depth = 22)

#gbm.fit(X_train, fatalities)
result_fal = model_fal.predict(test)
for i in range(len(result_fal)):

    if result_fal[i]<0:

        result_fal[i] = 0

        result_fal[i] = int(result_fal[i])

    result_fal[i] = int(result_fal[i])
fal_pre = np.around(result_fal,decimals = 0)
submission['ConfirmedCases'] = con_pred

submission['Fatalities'] = fal_pre
submission.to_csv("submission.csv" , index = False)