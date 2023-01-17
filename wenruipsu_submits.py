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

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn import preprocessing
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])



train['day_of_month'] = train['Date'].dt.day

train['day_of_week'] = train['Date'].dt.dayofweek

train['month'] = train['Date'].dt.month

train['num_of_week'] = train['Date'].dt.week

test['day_of_month'] = test['Date'].dt.day

test['day_of_week'] = test['Date'].dt.dayofweek

test['month'] = test['Date'].dt.month

test['num_of_week'] = test['Date'].dt.week
train['Province_State'].fillna("None", inplace = True)

test['Province_State'].fillna("None", inplace = True)
train['Province_State'] = train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : x['Country_Region'] if x['Province_State']== "None" else x['Province_State'], axis=1)
test['Province_State'] = test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : x['Country_Region'] if x['Province_State']== "None" else x['Province_State'], axis=1)
labelencoder = preprocessing.LabelEncoder()



train['Country_Region'] = labelencoder.fit_transform(train['Country_Region'])

train['Province_State'] = labelencoder.fit_transform(train['Province_State'])



test['Country_Region'] = labelencoder.fit_transform(test['Country_Region'])

test['Province_State'] = labelencoder.fit_transform(test['Province_State'])
y1_train = train['ConfirmedCases']

y2_train = train['Fatalities']

test_ForecastId = test['ForecastId']



X_train = train.drop(columns=['Id', 'Date','ConfirmedCases', 'Fatalities'])

X_test  = test.drop(columns=['ForecastId', 'Date'])
def root_mean_squared_logarithmic_error(y_pred,y_true):

    _y_true = y_true.get_label()

    return 'rmsle' , float(np.sqrt(np.mean(np.square(np.log(y_pred + 1) - np.log(_y_true + 1)))))
model1 = XGBRegressor(n_estimators=2000,max_depth=7,booster ='dart',

                     random_state =2020)

model1.fit(X_train, y1_train)

          #eval_metric = root_mean_squared_logarithmic_error)

y1_pred = model1.predict(X_test)



model2 = XGBRegressor(n_estimators=2000,max_depth=7,booster ='dart',

                     random_state =2020)

model2.fit(X_train, y2_train)

          #eval_metric = root_mean_squared_logarithmic_error)

y2_pred = model2.predict(X_test)



df_submit = pd.DataFrame({'ForecastId': test_ForecastId, 'ConfirmedCases': y1_pred, 'Fatalities': y2_pred})

df_submit['ConfirmedCases'] = df_submit['ConfirmedCases'].map(lambda x: 0 if x<0 else int(x))

df_submit['Fatalities'] = df_submit['Fatalities'].map(lambda x: 0 if x<0 else int(x))
# model3 = LGBMRegressor(n_estimators=2000,max_depth=7,

#                          random_state =2020)

# model3.fit(X_train, y1_train,

#           eval_metric = root_mean_squared_logarithmic_error)

# y3_pred = model3.predict(X_test)



# model4 = LGBMRegressor(n_estimators=2000,max_depth=7,

#                           random_state =2020)

# model4.fit(X_train, y2_train,

#           eval_metric = root_mean_squared_logarithmic_error)

# y4_pred = model4.predict(X_test)



# df_submit = pd.DataFrame({'ForecastId': test_ForecastId, 'ConfirmedCases': y3_pred, 'Fatalities': y4_pred})

# df_submit['ConfirmedCases'] = df_submit['ConfirmedCases'].map(lambda x: 0 if x<0 else int(x))

# df_submit['Fatalities'] = df_submit['Fatalities'].map(lambda x: 0 if x<0 else int(x))
# model1 = XGBRegressor(n_estimators=2000,

#                      random_state =2020)

# model1.fit(X_train, y1_train,

#           eval_metric = root_mean_squared_logarithmic_error)

# y1_pred = model1.predict(X_test)



# model2 = XGBRegressor(n_estimators=2000,

#                      random_state =2020)

# model2.fit(X_train, y2_train,

#           eval_metric = root_mean_squared_logarithmic_error)

# y2_pred = model2.predict(X_test)





# model3 = LGBMRegressor(n_estimators=2000,

#                       random_state =2020)

# model3.fit(X_train, y1_train,

#           eval_metric = root_mean_squared_logarithmic_error)

# y3_pred = model3.predict(X_test)



# model4 = LGBMRegressor(n_estimators=2000,

#                       random_state =2020)

# model4.fit(X_train, y2_train,

#           eval_metric = root_mean_squared_logarithmic_error)

# y4_pred = model4.predict(X_test)



# df_submit = pd.DataFrame({'ForecastId': test_ForecastId, 'ConfirmedCases': (y1_pred+y3_pred)/2, 'Fatalities': (y2_pred+y4_pred)/2})

# df_submit['ConfirmedCases'] = df_submit['ConfirmedCases'].map(lambda x: 0 if x<0 else int(x))

# df_submit['Fatalities'] = df_submit['Fatalities'].map(lambda x: 0 if x<0 else int(x))
df_submit.to_csv('submission.csv', index=False)