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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy. optimize import curve_fit



from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.arima_model import ARIMA

from random import random
PATH_WEEK2 = "/kaggle/input/covid19-global-forecasting-week-3"



# load train and test data

df_train = pd.read_csv(f"{PATH_WEEK2}/train.csv")

df_test = pd.read_csv(f"{PATH_WEEK2}/test.csv")



# show head

df_train.head()

df_test.head()



# change column name

df_train.rename(columns = {"Country_Region": "Country", 

                           "Province_State": "State"}, inplace = True)

df_test.rename(columns = {"Country_Region": "Country", 

                          "Province_State": "State"}, inplace = True)



# change Date column to datetime64 type

df_train["Date"] = pd.to_datetime(df_train["Date"], infer_datetime_format = True)

df_test["Date"] = pd.to_datetime(df_test["Date"], infer_datetime_format = True)



# show info

#df_train.info()

#df_test.info()



EMPTY_VAL = "UNKNOWN"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state



# copy to X_train

X_train = df_train.copy()



# replace empty State with Country

X_train["State"].fillna(EMPTY_VAL, inplace = True)

X_train["State"] = X_train.loc[:, ["State", "Country"]].apply(lambda x: fillState(x["State"], x["Country"]), axis =1)



# change Date column as int type

X_train.loc[:, "Date"] = X_train.Date.dt.strftime("%m%d")

X_train["Date"] = X_train["Date"].astype(int)



# check the result

X_train.head()



# do the same for test dataset

X_test = df_test.copy()



# replace empty State with Country

X_test["State"].fillna(EMPTY_VAL, inplace = True)

X_test["State"] = X_test.loc[:, ["State", "Country"]].apply(lambda x: fillState(x["State"], x["Country"]), axis =1)



# change Date column as int type

X_test.loc[:, 'Date'] = X_test.Date.dt.strftime("%m%d")

X_test["Date"] = X_test["Date"].astype(int)



X_test.head()

from warnings import filterwarnings

filterwarnings("ignore")



from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()



from xgboost import XGBRegressor



result_out = pd.DataFrame({"ForecastId": [], "ConfirmedCases": [], "Fatalities": []})



for country in X_train.Country.unique():

    states = X_train.loc[X_train.Country == country, :].State.unique()

    

    for state in states:

        # train dataset group by Country and State

       

        X_train_group = X_train.loc[(X_train.Country == country) & (X_train.State == state), 

                                ["State", "Country", "Date", "ConfirmedCases", "Fatalities" ]]



        y1_train_group = X_train_group.loc[:, "ConfirmedCases"]

        y2_train_group = X_train_group.loc[:, "Fatalities"]

        

        X_train_group = X_train_group.loc[:, ["State", "Country", "Date"]]

        X_train_group.Country = label_encoder.fit_transform(X_train_group.Country)

        X_train_group.State = label_encoder.fit_transform(X_train_group.State)



        # test dataset group by Country and State

        X_test_group = X_test.loc[(X_test.Country == country) & (X_test.State == state), 

                                ["State", "Country", "Date", "ForecastId" ]]

        X_test_group_id = X_test_group.loc[:, "ForecastId"]

        X_test_group = X_test_group.loc[:, ["State", "Country", "Date"]]

        X_test_group.Country = label_encoder.fit_transform(X_test_group.Country)

        X_test_group.State = label_encoder.fit_transform(X_test_group.State)



        # model and predict Confirmed Cases

        model_c = XGBRegressor(n_estimators = 500, learning_rate = 0.05)

        model_c.fit(X_train_group, y1_train_group)

        y1_pred = model_c.predict(X_test_group)

        

        # model and predict Fatalities

        model_f = XGBRegressor(n_estimators = 500, learning_rate = 0.05)

        model_f.fit(X_train_group, y2_train_group)

        y2_pred = model_f.predict(X_test_group)

        

        # prepare result

        result = pd.DataFrame({"ForecastId": X_test_group_id, "ConfirmedCases": y1_pred, "Fatalities": y2_pred })

        result_out = pd.concat([result_out, result], axis = 0)

    # state loop end

#country loop end

result_out.ForecastId = result_out.ForecastId.astype('int')

result_out.tail()

result_out.to_csv("submission.csv", index = False)