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
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")
test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")
train_data.head()
test_data.head()
#change the NaN values with the country names in the train data 
for row in range(len(train_data['Province/State'])):
    if str(train_data['Province/State'][row]) == 'nan':
        train_data['Province/State'][row] = train_data['Country/Region'][row]
        
#change the NaN values with the country names in the test data 
for row in range(len(test_data['Province/State'])):
    if str(test_data['Province/State'][row]) == 'nan':
        test_data['Province/State'][row] = test_data['Country/Region'][row]
train_data.head()
test_data.head()
train_data[train_data['Country/Region'] == 'Turkey']
day_number = train_data[train_data['Province/State'] == 'Turkey'].shape[0]
day_number
train_data = train_data.drop(['Long', 'Lat'], axis = 1)
test_data = test_data.drop(['Long', 'Lat'], axis = 1)
state_code = 0
train_data['Province/State'][0] = 0

#there are 284 countries 
for instance in range(0, 284):
    #encode each province for each day
    for state_index in range(0, day_number):
        #the code must be unique for each province
        train_data['Province/State'][state_index + (day_number * instance)] = state_code
    state_code += 1
train_data[train_data['Country/Region'] == 'Turkey']
train_data = train_data.drop('Country/Region', axis = 1)
test_data = test_data.drop('Country/Region', axis = 1)
#create new data sets 
CC_dataset = pd.DataFrame()
Fatal_dataset = pd.DataFrame()
target = []


for forecast_id in range(0, 284):
    #take sample data of Confirmed Cases and Fatalities for one country
    sample_CC_data = train_data.loc[train_data['Province/State'] == forecast_id]
    sample_Fatal_data = train_data.loc[train_data['Province/State'] == forecast_id]

    #keep data in the features of Confirmed Cases and Fatalities
    CC_column = sample_CC_data['ConfirmedCases']
    sample_CC_data = sample_CC_data.drop(['ConfirmedCases', 'Date', 'Fatalities'], axis = 1)

    Fatal_column = sample_Fatal_data['Fatalities']
    sample_Fatal_data = sample_Fatal_data.drop(['Fatalities', 'Date', 'ConfirmedCases'], axis = 1)

    #anymore, we can take first row because others are same
    sample_CC_data = sample_CC_data.head(1)
    sample_Fatal_data = sample_Fatal_data.head(1)

    #create list
    CC_list = list(CC_column)
    Fatal_list = list(Fatal_column)

    #create columns in new CC data frame: CC1,CC2,...
    for CC_index in range(0,len(CC_list)):
        sample_CC_data['CC' + str(CC_index+1)] = CC_list[CC_index]
        
    #create columns in new Fatality data frame: F1,F2,...
    for Fatal_index in range(0,len(Fatal_list)):
        sample_Fatal_data['F' + str(Fatal_index+1)] = Fatal_list[Fatal_index]

    #target list is ready 
    #CC_target = CC_list[-25:]
    #target.append(CC_target)

    #adding sample data for one country to new data frames 
    CC_dataset = CC_dataset.append(sample_CC_data)
    Fatal_dataset = Fatal_dataset.append(sample_Fatal_data)
CC_dataset.head()
Fatal_dataset.head()
#create empty data sets for calculating daily cases and fatalities
CC_daily_dataset = pd.DataFrame()
Fatality_daily_dataset = pd.DataFrame()

country_number = CC_dataset.shape[0]

for state_id in range(0, country_number):
    sample_case = CC_dataset.loc[CC_dataset['Province/State'] == state_id]
    #we must keep the data of case for first day
    sample_case['Day1CC'] =  sample_case['CC1']
    
    sample_fatality = Fatal_dataset.loc[Fatal_dataset['Province/State'] == state_id]
    #we must keep the data of fatality for first day
    sample_fatality['Day1F'] = sample_fatality['F1']
    
    for case_index in range(1, day_number):
        #extracting the CC data value of the day from the data value of previous day
        sample_case['Day' + str(case_index + 1) + 'CC'] = sample_case['CC' + str(case_index + 1)] - sample_case['CC' + str(case_index)]
        #we do not need the CC data value of the day, anymore
        sample_case = sample_case.drop(['CC' + str(case_index)], axis = 1)
        
        #extracting the fatality data value of the day from the data value of previous day
        sample_fatality['Day' + str(case_index + 1) + 'F'] = sample_fatality['F' + str(case_index + 1)] - sample_fatality['F' + str(case_index)]
        #we do not need the fatality data value of the day, anymore
        sample_fatality = sample_fatality.drop(['F' + str(case_index)], axis = 1)
    
    #drop the data of last day
    sample_case = sample_case.drop(['CC' + str(day_number)], axis = 1)
    sample_fatality = sample_fatality.drop(['F' + str(day_number)], axis = 1)
    
    #assign avaliable daily data to new daily data sets 
    CC_daily_dataset = CC_daily_dataset.append(sample_case)
    Fatality_daily_dataset = Fatality_daily_dataset.append(sample_fatality)

CC_daily_dataset
Fatality_daily_dataset
last_day_CC = CC_dataset.iloc[:,-1].values
last_day_F = Fatal_dataset.iloc[:,-1].values
CC_daily_dataset = CC_daily_dataset.drop(['Id', 'Province/State'], axis = 1)
Fatality_daily_dataset = Fatality_daily_dataset.drop(['Id', 'Province/State'], axis = 1)
features_CC = CC_dataset
train_features_CC = CC_dataset.iloc[:,:-10]
target_CC = CC_dataset.iloc[:,-10:]

features_F = Fatal_dataset
train_features_F = Fatal_dataset.iloc[:,:-10]
target_F = Fatal_dataset.iloc[:,-10:]


#28 days for predict, 35 days for train and test
train_test_CC_daily_dataset = CC_daily_dataset.iloc[:,:-28]
target_CC_daily_dataset = CC_daily_dataset.iloc[:,-28:]

train_test_fatal_daily_dataset = Fatality_daily_dataset.iloc[:,:-28]
target_fatal_daily_dataset = Fatality_daily_dataset.iloc[:,-28:]
train_test_CC_daily_dataset
target_CC_daily_dataset
target_CC = np.array(target_CC)
target_CC
target_F = np.array(target_F)
target_F
features_CC = np.array(features_CC)
features_F = np.array(features_F)

train_features_CC = np.array(train_features_CC)
train_features_F = np.array(train_features_F)

train_test_CC_daily = np.array(train_test_CC_daily_dataset)
train_test_F_daily = np.array(train_test_fatal_daily_dataset)

target_CC_daily = np.array(target_CC_daily_dataset)
target_F_daily = np.array(target_fatal_daily_dataset)

#train and test split based on instance countries. the test size is 0.2

from sklearn.model_selection import train_test_split

train_CC_daily, test_CC_daily, train_target_CC_daily, test_target_CC_daily = train_test_split(train_test_CC_daily, 
                                                                                              target_CC_daily, 
                                                                                              test_size = 0.2)

train_F_daily, test_F_daily, train_target_F_daily, test_target_F_daily = train_test_split(train_test_F_daily, 
                                                                                          target_F_daily, 
                                                                                          test_size = 0.2)

'''
train_CC_daily_dataset = np.expand_dims(train_CC_daily_dataset, axis = 2)
train_fatal_daily_dataset = np.expand_dims(train_fatal_daily_dataset, axis = 2)

test_CC_daily_dataset = np.expand_dims(test_CC_daily_dataset, axis = 2)
test_fatal_daily_dataset = np.expand_dims(test_fatal_daily_dataset, axis = 2)

#train_target_CC_daily_dataset = np.expand_dims(train_target_CC_daily_dataset, axis = 2)
#train_target_fatal_daily_dataset = np.expand_dims(train_target_fatal_daily_dataset, axis = 2)

#test_target_CC_daily_dataset = np.expand_dims(test_target_CC_daily_dataset, axis = 2)
#test_target_fatal_daily_dataset = np.expand_dims(test_target_fatal_daily_dataset, axis = 2)
'''

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
#scaler = MinMaxScaler(feature_range = (0, 1), copy = False)

#features_CC = scaler.fit_transform(features_CC)
#target_CC = scaler.fit_transform(target_CC)

#features_F = scaler.fit_transform(features_F)
#target_F = scaler.fit_transform(target_F)

#norm_features_CC = normalize(features_CC)
#norm_target_CC = normalize(target_CC)

#norm_features_F = normalize(features_F)
#norm_target_F = normalize(target_F)
features_CC = np.expand_dims(features_CC, axis = 2)
train_features_CC = np.expand_dims(train_features_CC, axis = 2)

train_features_CC = np.expand_dims(train_features_CC, axis = 2)
train_features_F = np.expand_dims(train_features_F, axis = 2)

train_CC_daily = np.expand_dims(train_CC_daily, axis = 2)
train_F_daily = np.expand_dims(train_F_daily, axis = 2)

test_CC_daily = np.expand_dims(test_CC_daily, axis = 2)
test_F_daily = np.expand_dims(test_F_daily, axis = 2)
from keras.layers import LSTM, Dense, Dropout, Flatten, BatchNormalization, Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.regularizers import l1, l2, l1_l2
model = Sequential()
model.add(LSTM(512, input_shape = (train_CC_daily.shape[1], 1), 
               return_sequences=True, activation='tanh'))
model.add(LSTM(256, return_sequences=True, activation='relu'))
model.add(LSTM(512, return_sequences=False, activation='tanh'))
model.add(Dense(28, activation = 'linear'))
model.compile(optimizer = 'Adam', loss = 'mean_squared_logarithmic_error', metrics=['accuracy'])
rnn = model.fit(train_CC_daily, train_target_CC_daily, batch_size=64, epochs=20, 
                validation_data=(test_CC_daily, test_target_CC_daily))
prediction_CC.shape
prediction_CC = model.predict(train_CC_daily)
prediction_F = model.predict(train_F_daily)
prediction_CC
last_case = train_features_CC[0][-1] + abs(round(prediction_CC[0][0]))
np.append(train_features_CC[0], last_case)
train_features_CC[0]
prediction_CC
#prediction_CC = scaler.inverse_transform(prediction_CC)
#prediction_F = scaler.inverse_transform(prediction_F)
#prediction_F = scaler.inverse_transform(prediction_F)
#prediction_F
for index in range(len(prediction_CC)):
    for inner_index in range(len(prediction_CC[index])):
        if prediction_CC[index][inner_index] >= 0:
            prediction_CC[index][inner_index] = round(prediction_CC[index][inner_index])
        elif prediction_CC[index][inner_index] < 0:
            prediction_CC[index][inner_index] = round(abs(prediction_CC[index][inner_index]))
for index in range(len(prediction_F)):
    for inner_index in range(len(prediction_F[index])):
        if prediction_F[index][inner_index] >= 0:
            prediction_F[index][inner_index] = round(prediction_F[index][inner_index])
        elif prediction_F[index][inner_index] < 0:
            prediction_F[index][inner_index] = round(abs(prediction_F[index][inner_index]))
prediction_CC
prediction_F
for index in range(len(prediction_CC)):
    prediction_CC[index][0] = last_day_CC[index]
    for inner_index in range(1, len(prediction_CC[index])):
        prediction_CC[index][inner_index] = prediction_CC[index][inner_index] + prediction_CC[index][inner_index-1]
prediction_CC
for index in range(len(prediction_F)):
    prediction_F[index][0] = last_day_F[index]
    for inner_index in range(1, len(prediction_F[index])):
        prediction_F[index][inner_index] = prediction_F[index][inner_index] + prediction_F[index][inner_index-1]
prediction_F
prediction_CC.shape
target_CC = CC_dataset.iloc[:,-12:]
target_CC = np.array(target_CC)
target_CC
target_F = Fatal_dataset.iloc[:,-12:]
target_F = np.array(target_F)
target_F
confirmed_cases = np.concatenate((target_CC, prediction_CC), axis = 1)
fatalities = np.concatenate((target_F, prediction_F), axis = 1)
confirmed_cases
fatalities
submitted_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")
submitted_data['ConfirmedCases'].shape
confirmed_cases = confirmed_cases.reshape(12212,)
fatalities = fatalities.reshape(12212,)
submitted_data['Fatalities'] = fatalities
submitted_data['ConfirmedCases'] = confirmed_cases
submitted_data.to_csv('submission.csv', index = False, encoding = 'utf-8-sig')
