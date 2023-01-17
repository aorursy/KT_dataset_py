# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder



from sklearn.model_selection import train_test_split



from xgboost import XGBRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.impute import SimpleImputer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
PATH_WK4 = "/kaggle/input/covid19-global-forecasting-week-4"



df_train = pd.read_csv(f'{PATH_WK4}/train.csv')

df_test = pd.read_csv(f'{PATH_WK4}/test.csv')

display(df_train.head())

display(df_test.head())

display(df_train.dtypes)

#df_test.dtypes
df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)

df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)



df_train.loc[:, 'Date'] = df_train.Date.dt.strftime('%Y%m%d')

df_train.loc[:, 'Date'] = df_train['Date'].astype(int)



df_test.loc[:, 'Date'] = df_test.Date.dt.strftime('%Y%m%d')

df_test.loc[:, 'Date'] = df_test['Date'].astype(int)

display(df_train.info())

display(df_train.info())

train_date_min = df_train["Date"].min()

train_date_max = df_train["Date"].max()



print("Minimum date from training data: {}".format(train_date_min))

print("Maximum date from training data: {}".format(train_date_max))

test_date_min = df_test["Date"].min()

test_date_max = df_test["Date"].max()



print("Minimum date from testing data: {}".format(test_date_min))

print("Maximum date from testing data: {}".format(test_date_max))
df_train.rename(columns = {"Country_Region": "Country", "Province_State": "State"}, inplace = True)

df_test.rename(columns = {"Country_Region": "Country", "Province_State": "State"}, inplace = True)

#df_train.head(20)

EMPTY_VAL = "UNKNOWN"



def fill_state(state, country):

    if state == EMPTY_VAL: return country

    return state



df_train['State'].fillna(EMPTY_VAL, inplace=True)

df_train['State'] = df_train.loc[:, ['State', 'Country']].apply(lambda x : fill_state(x['State'], x['Country']), axis=1)



print(df_train.head(20), df_train.dtypes)



df_test['State'].fillna(EMPTY_VAL, inplace=True)

df_test['State'] = df_test.loc[:, ['State', 'Country']].apply(lambda x : fill_state(x['State'], x['Country']), axis=1)

print(df_test.head(10), df_test.dtypes)

china_data = df_train[df_train["Country"]=="China"]

china_data.head()

missing_val_count_by_column = (china_data.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column>0])
# confirmed cases in China

plt.figure(figsize = (10,6))

plt.xlabel("Confirmed Cases")

plt.title("Confirmed Cases in China")

sns.barplot(x = china_data["ConfirmedCases"], y = china_data["State"])
# fatalities in China

plt.figure(figsize = (10,6))

plt.xlabel("Fatalities")

plt.title("Fatalities in China")

sns.barplot(x = china_data["Fatalities"], y = china_data["State"])
plt.title("Confirmed Cases vs. Fatalities")

sns.regplot(x=df_train["ConfirmedCases"], y=df_train["Fatalities"], fit_reg=True)
#df_groupByCountry = df_train.loc[:, ['Country', 'State', 'ConfirmedCases', 'Fatalities']].groupby(['Country', 'State']).max().reset_index().groupby('Country').sum().sort_values(by='ConfirmedCases', ascending=False).reset_index()

#df_groupByCountry[:20].style.background_gradient(cmap='viridis_r')

plt.title("The Most Infected Patients in Countries")

sns.countplot(y="Country", data=df_train,order=df_train["Country"].value_counts(ascending=False).iloc[:10].index)
china = df_train[df_train["Country"]=="China"]



plt.figure(figsize=(10,6))

plt.plot(china["ConfirmedCases"])

plt.xlabel("Time")

plt.ylabel("Number of Confirmed Cases")
def avoid_data_leakage(df, date=test_date_min):

    return df[df['Date']<date]



df_train = avoid_data_leakage(df_train)

display(df_train.info())

df_train_max = df_train["Date"].max()

display(df_train_max)
EMPTY_VAL = "UNKNOWN"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state



# copy to X_train

X_train = df_train.copy()



# replace empty State with Country

X_train["State"].fillna(EMPTY_VAL, inplace = True)

X_train["State"] = X_train.loc[:, ["State", "Country"]].apply(lambda x: fillState(x["State"], x["Country"]), axis =1)



# check the result

display(X_train.head())



# do the same for test dataset

X_test = df_test.copy()



# replace empty State with Country

X_test["State"].fillna(EMPTY_VAL, inplace = True)

X_test["State"] = X_test.loc[:, ["State", "Country"]].apply(lambda x: fillState(x["State"], x["Country"]), axis =1)



display(X_test.head())

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

        model_c = XGBRegressor(n_estimators = 1000, learning_rate = 0.05)

        model_c.fit(X_train_group, y1_train_group)

        y1_pred = model_c.predict(X_test_group)

        

        # model and predict Fatalities

        model_f = XGBRegressor(n_estimators = 1000, learning_rate = 0.05)

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