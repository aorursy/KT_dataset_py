# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

PATH_WEEK2='/kaggle/input/covid19-global-forecasting-week-2'

df_train = pd.read_csv(f'{PATH_WEEK2}/train.csv')

df_test = pd.read_csv(f'{PATH_WEEK2}/test.csv')

print("*"*100)

print(df_train.head())

print("*"*100)

print(df_test.head())

print("*"*100)
df_train.rename(columns={'Country_Region':'Country'}, inplace=True)

df_test.rename(columns={'Country_Region':'Country'}, inplace=True)



df_train.rename(columns={'Province_State':'State'}, inplace=True)

df_test.rename(columns={'Province_State':'State'}, inplace=True)



df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)

df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)



print("*"*50)

print(df_train.info())

print("*"*50)

print(df_test.info())

print("*"*50)
NULL_VAL = "NULL_VAL"



def fillState(state, country):

    if state == NULL_VAL: 

        return country

    

    return state



X_Train = df_train.loc[:, ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']]

X_Train['State'].fillna(NULL_VAL, inplace=True)

X_Train['State'] = X_Train.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

X_Train.loc[:, 'Date'] = X_Train.Date.dt.strftime("%m%d")

X_Train["Date"]  = X_Train["Date"].astype(int)



X_Test = df_test.loc[:, ['State', 'Country', 'Date', 'ForecastId']]

X_Test['State'].fillna(NULL_VAL, inplace=True)

X_Test['State'] = X_Test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

X_Test.loc[:, 'Date'] = X_Test.Date.dt.strftime("%m%d")

X_Test["Date"]  = X_Test["Date"].astype(int)



print("*"*50)

print(X_Train.head())

print("*"*50)

print(X_Test.head())

print("*"*50)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



X_Train.Country = le.fit_transform(X_Train.Country)

X_Train['State'] = le.fit_transform(X_Train['State'])



X_Test.Country = le.fit_transform(X_Test.Country)

X_Test['State'] = le.fit_transform(X_Test['State'])



print("*"*50)

print(X_Train.head())

print("*"*50)

print(X_Test.head())

print("*"*50)
# from sklearn.model_selection import GridSearchCV

# from xgboost import XGBRegressor



# y_Train = X_Train.ConfirmedCases

# hyperParam = {'n_estimators': [1000]}

# grid_cv = GridSearchCV(XGBRegressor(), hyperParam, cv=10, scoring="neg_mean_squared_error")

# grid_cv.fit(X_Train, y_Train)

# grid_cv.best_estimator_



# model = XGBRegressor()

# y_Train1 = X_Train.Fatalities



# grid_cv1 = GridSearchCV(XGBRegressor(), hyperParam, cv=10, scoring="neg_mean_squared_error")

# grid_cv1.fit(X_Train, y_Train1)

# grid_cv1.best_estimator_
from xgboost import XGBRegressor

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



countries = X_Train.Country.unique()



df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



for country in countries:

    states = X_Train.loc[X_Train.Country == country, :].State.unique()



    for state in states:

        condition_train = (X_Train.Country == country) & (X_Train.State == state)

       

        # Get X and y (train)

        X_Train_CS  = X_Train.loc[condition_train, ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']]

        y1_Train_CS = X_Train_CS.loc[:, 'ConfirmedCases']

        y2_Train_CS = X_Train_CS.loc[:, 'Fatalities']

        X_Train_CS  = X_Train_CS.loc[:, ['State', 'Country', 'Date']]



        # Get X and y (test)

        condition_test = (X_Test.Country == country) & (X_Test.State == state)

        X_Test_CS = X_Test.loc[condition_test, ['State', 'Country', 'Date', 'ForecastId']]

        

        # Save forcast id for submission

        X_Test_CS_Id = X_Test_CS.loc[:, 'ForecastId']

        X_Test_CS    = X_Test_CS.loc[:, ['State', 'Country', 'Date']]

        

        model1 = XGBRegressor(n_estimators=1000)

        model1.fit(X_Train_CS, y1_Train_CS)

        y1_pred = model1.predict(X_Test_CS)

        

        model2 = XGBRegressor(n_estimators=1000)

        model2.fit(X_Train_CS, y2_Train_CS)

        y2_pred = model2.predict(X_Test_CS)

        

        df = pd.DataFrame({'ForecastId': X_Test_CS_Id, 'ConfirmedCases': y1_pred, 'Fatalities': y2_pred})

        df_out = pd.concat([df_out, df], axis=0)

    # Done for state loop

# Done for country Loop



df_out.ForecastId = df_out.ForecastId.astype('int')

df_out.tail()
df_out.to_csv('submission.csv', index=False)