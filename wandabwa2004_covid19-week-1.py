# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from datetime import datetime

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression, BayesianRidge

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

submission_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")

test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
train_df.tail()
submission_df.head()
test_df.tail()
train_df["Date"] = train_df["Date"].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))

train_df["Date"] = train_df["Date"].apply(lambda x: x.timestamp())

train_df["Date"]  = train_df["Date"].astype(int)
train_df.isnull().sum() ##Province has lots of empty records so can  be dropped
train_df = train_df.drop(['Province/State'],axis=1)

train_df = train_df.dropna()
train_df.head()
test_df.isnull().sum() #Province seems  problematic again 
test_df["Date"] = test_df["Date"].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))

test_df["Date"] = test_df["Date"].apply(lambda x: x.timestamp())

test_df["Date"] = test_df["Date"].astype(int)

test_df = test_df.drop(['Province/State'],axis=1)

test_df = test_df.dropna()

test_df.head()
X = train_df[['Lat', 'Long', 'Date']]

Y1 = train_df[['ConfirmedCases']]



X_test = test_df[['Lat', 'Long', 'Date']]

Y2 = train_df[['Fatalities']]
X_test.tail()
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators=100) # Good model for  regression tasks

rf.fit(X,Y1)

pred1 = rf.predict(X_test)

pred1 = pd.DataFrame(pred1)

pred1.columns = ["ConfirmedCases_prediction"]



pred1.head()
rf_fatalities_model  = RandomForestRegressor(n_estimators=100)



rf_fatalities_model.fit(X,Y2)

pred2 = rf_fatalities_model.predict(X_test)

pred2 = pd.DataFrame(pred2)

pred2.columns = ["Death_prediction"]
pred2.tail()
submission  = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")

submission.columns

sub = submission[["ForecastId"]]
combined_preds = pd.concat([pred1,pred2,sub],axis=1)

combined_preds.head()

combined_preds.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']

combined_preds = combined_preds[['ForecastId','ConfirmedCases', 'Fatalities']]
combined_preds
combined_preds["ConfirmedCases"] = combined_preds["ConfirmedCases"].astype(int)

combined_preds["Fatalities"] = combined_preds["Fatalities"].astype(int)
combined_preds.head()
combined_preds.to_csv("submission.csv",index=False)