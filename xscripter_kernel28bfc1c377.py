import numpy as np

import pandas as pd

from sklearn import preprocessing

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

from warnings import filterwarnings



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

df_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

df_sub = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")
df_train.rename(columns={'Country_Region':'Country'}, inplace=True)

df_test.rename(columns={'Country_Region':'Country'}, inplace=True)



df_train.rename(columns={'Province_State':'State'}, inplace=True)

df_test.rename(columns={'Province_State':'State'}, inplace=True)



df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)

df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)
y1_Train = df_train.iloc[:, -2]

y1_Train.head()
y2_Train = df_train.iloc[:, -1]

y2_Train.head()
EMPTY_VAL = "EMPTY_VAL"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state
#X_Train = df_train.loc[:, ['State', 'Country', 'Date']]

X_Train = df_train.copy()



X_Train['State'].fillna(EMPTY_VAL, inplace=True)

X_Train['State'] = X_Train.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



X_Train.loc[:, 'Date'] = X_Train.Date.dt.strftime("%m%d")

X_Train["Date"]  = X_Train["Date"].astype(int)



X_Train.head()
X_Test = df_test.copy()



X_Test['State'].fillna(EMPTY_VAL, inplace=True)

X_Test['State'] = X_Test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



X_Test.loc[:, 'Date'] = X_Test.Date.dt.strftime("%m%d")

X_Test["Date"]  = X_Test["Date"].astype(int)



X_Test.head()
le = preprocessing.LabelEncoder()



X_Train.Country = le.fit_transform(X_Train.Country)

X_Train['State'] = le.fit_transform(X_Train['State'])



X_Test.Country = le.fit_transform(X_Test.Country)

X_Test['State'] = le.fit_transform(X_Test['State'])
df_train.loc[df_train.Country == 'Afghanistan', :]
from warnings import filterwarnings

filterwarnings('ignore')
countries = X_Train.Country.unique()



df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



for country in countries:

    states = X_Train.loc[X_Train.Country == country, :].State.unique()

    #print(country, states)

    # check whether string is nan or not

    for state in states:

        X_Train_CS = X_Train.loc[(X_Train.Country == country) & (X_Train.State == state), ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']]

        

        y1_Train_CS = X_Train_CS.loc[:, 'ConfirmedCases']

        y2_Train_CS = X_Train_CS.loc[:, 'Fatalities']

        

        X_Train_CS = X_Train_CS.loc[:, ['State', 'Country', 'Date']]

        

        X_Train_CS.Country = le.fit_transform(X_Train_CS.Country)

        X_Train_CS['State'] = le.fit_transform(X_Train_CS['State'])

        

        X_Test_CS = X_Test.loc[(X_Test.Country == country) & (X_Test.State == state), ['State', 'Country', 'Date', 'ForecastId']]

        

        X_Test_CS_Id = X_Test_CS.loc[:, 'ForecastId']

        X_Test_CS = X_Test_CS.loc[:, ['State', 'Country', 'Date']]

        

        X_Test_CS.Country = le.fit_transform(X_Test_CS.Country)

        X_Test_CS['State'] = le.fit_transform(X_Test_CS['State'])

        

        

        #xgb1 = XGBRegressor()

        #parameters = {'objective':['reg:linear'],

        #      'learning_rate': [.05, 0.07, .1],

        #      'max_depth': [5, 6, 7],

        #      'min_child_weight': [4],

        #      'silent': [1],

        #      'subsample': [0.7],

        #      'colsample_bytree': [0.7],

        #      'n_estimators': [1000]}

        

        #model1 = GridSearchCV(xgb1,

        #                parameters,

        #                cv = 2,

        #                verbose=False)

        

        

        #model2 = GridSearchCV(xgb1,

        #                parameters,

        #                cv = 2,

        #                verbose=False)

        

        

        model1 = XGBRegressor(learning_rate=0.05, n_estimators=2000,cv=10)

        model1.fit(X_Train_CS, y1_Train_CS)

        y1_pred = model1.predict(X_Test_CS)

        

        model2 = XGBRegressor(learning_rate=0.05, n_estimators=2000,cv=10)

        model2.fit(X_Train_CS, y2_Train_CS)

        y2_pred = model2.predict(X_Test_CS)

        

        df = pd.DataFrame({'ForecastId': X_Test_CS_Id, 'ConfirmedCases': y1_pred, 'Fatalities': y2_pred})

        df_out = pd.concat([df_out, df], axis=0)
# print(model1.best_score_)

# print(model1.best_params_)
# print(model2.best_score_)

# print(model2.best_params_)
df_out.ForecastId = df_out.ForecastId.astype('int')

df_out.tail()
df_out.to_csv('submission.csv', index=False)