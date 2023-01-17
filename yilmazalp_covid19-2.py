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
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
train_data.head()
train_data.shape
for row in range(len(train_data['Province_State'])):

    if str(train_data['Province_State'][row]) == 'nan':

        train_data['Province_State'][row] = train_data['Country_Region'][row]
for row in range(len(test_data['Province_State'])):

    if str(test_data['Province_State'][row]) == 'nan':

        test_data['Province_State'][row] = test_data['Country_Region'][row]
day_number = train_data[train_data['Province_State'] == 'Florida'].shape[0]

day_number
country_number = train_data.shape[0]/day_number

country_number
state_code = 0

train_data['Province_State'][0] = 0

for instance in range(0, int(country_number)):

    for state_index in range(0, day_number):

        train_data['Province_State'][state_index + (day_number * instance)] = state_code

    state_code += 1
train_data[train_data['Country_Region'] == 'Turkey']
train_data[train_data['Country_Region'] == 'Turkey']['Date']
from datetime import datetime



time1 = train_data[train_data['Country_Region'] == 'Turkey']['Date'].values[-1]

time2 = '2020-04-30'

FMT = '%Y-%m-%d'



time_diff = datetime.strptime(time2, FMT) - datetime.strptime(time1, FMT)
future_day_number = time_diff.days

future_day_number
train_data = train_data.drop('Country_Region', axis = 1)

test_data = test_data.drop('Country_Region', axis = 1)
CC_dataset = pd.DataFrame()

Fatal_dataset = pd.DataFrame()

target = []





for forecast_id in range(0, int(country_number)):

    #take sample data of Confirmed Cases and Fatalities for one country

    sample_CC_data = train_data.loc[train_data['Province_State'] == forecast_id]

    sample_Fatal_data = train_data.loc[train_data['Province_State'] == forecast_id]



    #take feature of Confirmed Cases and Fatalities

    CC_column = sample_CC_data['ConfirmedCases']

    sample_CC_data = sample_CC_data.drop(['ConfirmedCases', 'Date', 'Fatalities'], axis = 1)



    

    Fatal_column = sample_Fatal_data['Fatalities']

    sample_Fatal_data = sample_Fatal_data.drop(['Fatalities', 'Date', 'ConfirmedCases'], axis = 1)



    #we can take first row, because others are same

    sample_CC_data = sample_CC_data.head(1)

    sample_Fatal_data = sample_Fatal_data.head(1)



    #the data of KG_N 

    CC_list = list(CC_column)

    Fatal_list = list(Fatal_column)

    #CC_list.append(sample_data['ConfirmedCases'])

    #sample_test = test_data.loc[test_data['ID'] == consumer_id]



    #data of last 5 months is ready for target value. Because, we try to forecast 

    # sale(KG_N) data for future 5 months 



    for CC_index in range(0,len(CC_list)):

        sample_CC_data['CC' + str(CC_index+1)] = CC_list[CC_index]

        

    for Fatal_index in range(0,len(Fatal_list)):

        sample_Fatal_data['F' + str(Fatal_index+1)] = Fatal_list[Fatal_index]



    #target list is ready 

    #CC_target = CC_list[-25:]

    #target.append(CC_target)



    #we create a new data frame. After processing of one consumer, we add this to data frame  

    CC_dataset = CC_dataset.append(sample_CC_data)

    Fatal_dataset = Fatal_dataset.append(sample_Fatal_data)
CC_dataset.head()
Fatal_dataset.head()
country_number = CC_dataset.shape[0]

country_number
CC_daily_dataset = pd.DataFrame()

Fatality_daily_dataset = pd.DataFrame()





for state_id in range(0, country_number):

    sample_case = CC_dataset.loc[CC_dataset['Province_State'] == state_id]

    sample_case['Day1CC'] =  sample_case['CC1']

    

    sample_fatality = Fatal_dataset.loc[Fatal_dataset['Province_State'] == state_id]

    sample_fatality['Day1F'] = sample_fatality['F1']

    

    for case_index in range(1, day_number):

        sample_case['Day' + str(case_index + 1) + 'CC'] = sample_case['CC' + str(case_index + 1)] - sample_case['CC' + str(case_index)]

        sample_case = sample_case.drop(['CC' + str(case_index)], axis = 1)

        

        sample_fatality['Day' + str(case_index + 1) + 'F'] = sample_fatality['F' + str(case_index + 1)] - sample_fatality['F' + str(case_index)]

        sample_fatality = sample_fatality.drop(['F' + str(case_index)], axis = 1)

    

    sample_case = sample_case.drop(['CC' + str(day_number)], axis = 1)

    sample_fatality = sample_fatality.drop(['F' + str(day_number)], axis = 1)

    

    CC_daily_dataset = CC_daily_dataset.append(sample_case)

    Fatality_daily_dataset = Fatality_daily_dataset.append(sample_fatality)

CC_daily_dataset
CC_daily_dataset.shape
last_day_CC = CC_dataset.iloc[:,-1].values

last_day_F = Fatal_dataset.iloc[:,-1].values
CC_daily_dataset[CC_daily_dataset['Province_State'] == 223]
CC_daily_dataset = CC_daily_dataset.drop(['Id', 'Province_State'], axis = 1)

Fatality_daily_dataset = Fatality_daily_dataset.drop(['Id', 'Province_State'], axis = 1)
features_CC = CC_daily_dataset.iloc[:, :-1*(future_day_number)]

target_CC = CC_daily_dataset.iloc[:, -1*(future_day_number):]



features_F = CC_daily_dataset.iloc[:, :-1*(future_day_number)]

target_F = Fatality_daily_dataset.iloc[:, -1*(future_day_number):]



from sklearn.model_selection import train_test_split

train_features_CC, test_features_CC, train_target_CC, test_target_CC = train_test_split(features_CC, target_CC, 

                                                                                        test_size = 0.2)

train_features_F, test_features_F, train_target_F, test_target_F = train_test_split(features_F, target_F, 

                                                                                        test_size = 0.2)



'''

test_features_CC = CC_daily_dataset.iloc[:,-6:-1]

test_target_CC = CC_daily_dataset.iloc[:,-1:]



features_F = Fatality_daily_dataset

train_features_F = Fatality_daily_dataset.iloc[:,:-7]

train_target_F = Fatality_daily_dataset.iloc[:,-7:-6]



test_features_F = Fatality_daily_dataset.iloc[:,-6:-1]

test_target_F = Fatality_daily_dataset.iloc[:,-1:]

'''
train_features_CC.shape
target_CC
features_CC
target_CC
target_CC
train_features_CC = np.array(train_features_CC)

train_features_F = np.array(train_features_F)



test_features_CC = np.array(test_features_CC)

test_features_F = np.array(test_features_F)



train_target_CC = np.array(train_target_CC)

train_target_F = np.array(train_target_F)



test_target_CC = np.array(test_target_CC)

test_target_F = np.array(test_target_F)



features_CC = np.array(features_CC)

features_F = np.array(features_F)



target_CC = np.array(target_CC)

target_F = np.array(target_F)
target_CC[223]
from sklearn.linear_model import LinearRegression

import random 



random_number = random.randint(0, 10000)





model_CC = LinearRegression()

train_features_CC, test_features_CC, train_target_CC, test_target_CC = train_test_split(features_CC, target_CC, 

                                                                                        test_size = 0.2, 

                                                                                        random_state = random_number)





regressor = model_CC.fit(train_features_CC, train_target_CC)

prediction_CC = regressor.predict(features_CC)
for country_index in range(country_number):

    for day_index in range(len(prediction_CC[country_index])):

        prediction_CC[country_index][day_index] += target_CC[country_index][-1]
from sklearn.linear_model import LinearRegression

import random 



random_number = random.randint(0, 10000)





model_F = LinearRegression()

train_features_F, test_features_F, train_target_F, test_target_F = train_test_split(features_F, target_F, 

                                                                                        test_size = 0.2, 

                                                                                        random_state = random_number)





regressor_F = model_F.fit(train_features_F, train_target_F)

prediction_F = regressor_F.predict(features_F)
for country_index in range(country_number):

    for day_index in range(len(prediction_F[country_index])):

        prediction_F[country_index][day_index] += target_F[country_index][-1]
for index in range(len(prediction_CC)):

    for inner_index in range(len(prediction_CC[index])):

        if prediction_CC[index][inner_index] >= 0:

            prediction_CC[index][inner_index] = round(prediction_CC[index][inner_index])

        elif prediction_CC[index][inner_index] < 0:

            prediction_CC[index][inner_index] = 0
for index in range(len(prediction_F)):

    for inner_index in range(len(prediction_F[index])):

        if prediction_F[index][inner_index] >= 0:

            prediction_F[index][inner_index] = round(prediction_F[index][inner_index])

        elif prediction_F[index][inner_index] < 0:

            prediction_F[index][inner_index] = 0
prediction_CC
prediction_F
last_day_CC[223]
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
prediction_CC[223]
target_CC = CC_dataset.iloc[:,-13:]

target_CC = np.array(target_CC)

target_CC
target_F = Fatal_dataset.iloc[:,-13:]

target_F = np.array(target_F)

target_F
target_CC[223]
confirmed_cases = np.concatenate((target_CC, prediction_CC), axis = 1)

fatalities = np.concatenate((target_F, prediction_F), axis = 1)
confirmed_cases[223]
submitted_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")
submitted_data['ConfirmedCases'].shape
submitted_data['Fatalities'].shape
fatalities.shape
dim_CC = confirmed_cases.shape[0]*confirmed_cases.shape[1]

dim_F = fatalities.shape[0]*fatalities.shape[1]
confirmed_cases = confirmed_cases.reshape(dim_CC,)

fatalities = fatalities.reshape(dim_F,)
submitted_data['Fatalities'] = fatalities

submitted_data['ConfirmedCases'] = confirmed_cases
submitted_data.to_csv('submission.csv', index = False, encoding = 'utf-8-sig')