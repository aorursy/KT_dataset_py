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
# Imports

import xgboost as xgb
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

submission_data_schema = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")

test_data.head()
train_data['Date'] = pd.to_datetime(train_data['Date'])

print(train_data.describe())

print(max(train_data.Date))

print(test_data.head())

train_data.drop(['Id'], axis=1, inplace=True)

train_data.head()
print(len(train_data.Country_Region.unique()))

countries_without_state = train_data[pd.isnull(train_data['Province_State'])]

print(len(countries_without_state.Country_Region.unique()))

countries_with_state = train_data.dropna(subset=['Province_State'])

print(countries_with_state.Country_Region.unique())

for country in countries_with_state.Country_Region.unique():

    if country in countries_without_state.Country_Region.unique():

        print(country)

# 4 countries have both states data and country level data:

# Denmark

# France

# Netherlands

# United Kingdom
# Approach 2

# Date features

from datetime import datetime

train_data['Day'] = train_data.apply(lambda row: row.Date.day, axis=1)

train_data['Month'] = train_data.apply(lambda row: row.Date.month, axis=1)



# Impute Province_State

train_data['Province_State'] = train_data[['Province_State']].fillna('FULL_COUNTRY')

train_data.tail(100)
# Stationarize ConfirmdedCases and Fatalities. This has to be done per region

output_country = pd.DataFrame()

count = 0

for country in train_data.Country_Region.unique():

    country_data = train_data[train_data.Country_Region == country]

    for state in country_data.Province_State.unique():

        state_data = country_data[country_data.Province_State == state]

        state_data['cc_stationary'] = state_data[['ConfirmedCases']].diff()

        state_data['f_stationary'] = state_data[['Fatalities']].diff()

        output_country = output_country.append(state_data)

        count += state_data.shape[0]

train_data = train_data.merge(output_country, on=['Date', 'Country_Region', 'Province_State', 

                                     'Day', 'Month', 'ConfirmedCases', 'Fatalities'])

train_data.fillna(0, inplace=True)

train_data.head()
# # Evaluation using K-fold cross validation



# # Using LabelBinarizer and sparse matrix

# # Prepare features and prediction data

# features_data = train_data[['Country_Region', 'Province_State', 'Day', 'Month']]

# # LabelBinarize train_data

# from sklearn import preprocessing

# cr_lb = preprocessing.LabelBinarizer(sparse_output=True)

# ps_lb = preprocessing.LabelBinarizer(sparse_output=True)

# one_hot_countries = cr_lb.fit_transform(features_data[['Country_Region']])

# one_hot_states = ps_lb.fit_transform(features_data[['Province_State']])



# from scipy import sparse

# training_input = features_data[['Day', 'Month']].values

# training_input = sparse.hstack((training_input, one_hot_countries, one_hot_states))



# # y_confirmed_cases = train_data[['ConfirmedCases']]

# # y_fatalities = train_data[['Fatalities']]



# y_confirmed_cases = train_data[['cc_stationary']]

# y_fatalities = train_data[['f_stationary']]

# y_confirmed_cases.clip(0, inplace=True)

# y_fatalities.clip(0, inplace=True)



# # train_test_split - 5 folds

# from sklearn.model_selection import KFold

# SPLIT_COUNT = 10

# kf = KFold(n_splits=SPLIT_COUNT)



# # Running XGBoosRegressor

# from sklearn.ensemble import RandomForestRegressor

# from sklearn.metrics import mean_squared_log_error



# average_error = 0

# for train_index, test_index in kf.split(features_data):

#     X_train, X_test = training_input.tocsr()[train_index,:], training_input.tocsr()[test_index,:]

#     y_cc_train, y_cc_test = y_confirmed_cases.iloc[train_index], y_confirmed_cases.iloc[test_index]

#     y_f_train, y_f_test = y_fatalities.iloc[train_index], y_fatalities.iloc[test_index]

#     model_cc = xgb.XGBRegressor()

#     model_cc.fit(X_train, y_cc_train)

#     model_f = xgb.XGBRegressor()

#     model_f.fit(X_train, y_f_train)

#     predictions_cc = pd.DataFrame(model_cc.predict(X_test))

#     predictions_f = pd.DataFrame(model_f.predict(X_test))

#     predictions_cc.clip(0, inplace=True)

#     predictions_f.clip(0, inplace=True)

#     cc_error = np.sqrt(mean_squared_log_error(predictions_cc, y_cc_test))

#     f_error = np.sqrt(mean_squared_log_error(predictions_f, y_f_test))

#     median_error = (cc_error + f_error)/2

#     average_error += median_error

#     print(median_error)

# print(average_error/SPLIT_COUNT)



# # Average score Random forest: 1.7072718462044407

# # Average score XGBoost: 1.5573904740299072



# # Average after stationarization

# # XGBoost 1.1587040498615342
# Using LabelBinarizer and sparse matrix

# Prepare features and prediction data

features_data = train_data[['Country_Region', 'Province_State', 'Day', 'Month']]

# LabelBinarize train_data

from sklearn import preprocessing

cr_lb = preprocessing.LabelBinarizer(sparse_output=True)

ps_lb = preprocessing.LabelBinarizer(sparse_output=True)

one_hot_countries = cr_lb.fit_transform(features_data[['Country_Region']])

one_hot_states = ps_lb.fit_transform(features_data[['Province_State']])



from scipy import sparse

training_input = features_data[['Day', 'Month']].values

training_input = sparse.hstack((training_input, one_hot_countries, one_hot_states))



X_test = test_data

X_test['Date'] = pd.to_datetime(X_test['Date'])

X_test['Day'] = X_test.apply(lambda row: row.Date.day, axis=1)

X_test['Month'] = X_test.apply(lambda row: row.Date.month, axis=1)

X_test['Province_State'] = X_test[['Province_State']].fillna('FULL_COUNTRY')

X_test = X_test[['Country_Region', 'Province_State', 'Day', 'Month']]

one_hot_c = cr_lb.transform(X_test[['Country_Region']])

one_hot_s = ps_lb.transform(X_test[['Province_State']])

testing_input = X_test[['Day', 'Month']].values

testing_input = sparse.hstack((testing_input, one_hot_c, one_hot_s))



y_confirmed_cases = train_data[['cc_stationary']]

y_fatalities = train_data[['f_stationary']]

y_confirmed_cases.clip(0, inplace=True)

y_fatalities.clip(0, inplace=True)



# Running XGBoosRegressor



model_cc = xgb.XGBRegressor()

model_cc.fit(training_input, y_confirmed_cases)

model_f = xgb.XGBRegressor()

model_f.fit(training_input, y_fatalities)

predictions_cc = pd.DataFrame(model_cc.predict(testing_input))

predictions_f = pd.DataFrame(model_f.predict(testing_input))

predictions_cc.clip(0, inplace=True)

predictions_f.clip(0, inplace=True)

# predictions_cc.head()
# Submitting work of day 1

X_test['PredictedConfirmedCases_s'] = predictions_cc

X_test['PredictedFatalities_s'] = predictions_f

X_test['ForecastId'] = test_data['ForecastId']



LAST_TRAIN_DATE = '2020-04-01'

output_df = pd.DataFrame()

for country in train_data.Country_Region.unique():

    country_data = X_test[X_test['Country_Region'] == country]

    for state in country_data.Province_State.unique():

        state_data = country_data[country_data['Province_State'] == state]

        input_data = train_data[((train_data.Province_State == state) 

                                 & (train_data.Date == LAST_TRAIN_DATE)

                                 & (train_data.Country_Region == country)

                                )]

        state_data.iloc[0, state_data.columns.get_loc('PredictedConfirmedCases_s')] = input_data.iloc[0].ConfirmedCases + state_data.iloc[0].PredictedConfirmedCases_s

        state_data.iloc[0, state_data.columns.get_loc('PredictedFatalities_s')] = input_data.iloc[0].Fatalities + state_data.iloc[0].PredictedFatalities_s

        state_data['ConfirmedCases'] = state_data[['PredictedConfirmedCases_s']].cumsum()

        state_data['Fatalities'] = state_data[['PredictedFatalities_s']].cumsum()

        output_df = output_df.append(state_data)

output_df['ConfirmedCases'] = output_df['ConfirmedCases'].apply(np.ceil)

output_df['Fatalities'] = output_df['Fatalities'].apply(np.ceil)

output_df.head()

output_df = output_df[['ForecastId', 'ConfirmedCases', 'Fatalities']]

output_df.to_csv('submission.csv', index=False)