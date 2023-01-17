# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Grid Search ARIMA inspired by work from Jason Brownlee



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string

from datetime import datetime

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error

import warnings



# evaluate an ARIMA model for a given order (p,d,q)

def evaluate_arima_model(X, arima_order):

    # prepare training dataset

    X.index = pd.RangeIndex(len(X.index))

    train_size = int(len(X) * 0.8)

    train, test = X[0:train_size], X[train_size:]

    history = X.TargetValue.tolist()

    # make predictions

    predictions = list()

    for t, row in test.iterrows():

        model = ARIMA(history, order=arima_order)

        model_fit = model.fit(disp=0)

        yhat = model_fit.forecast()[0]

        predictions.append(yhat)

        history.append(test.TargetValue[t])

        # calculate out of sample error

    error = mean_squared_error(test, predictions)

    return error



# evaluate combinations of p, d and q values for an ARIMA model

def evaluate_models(dataset, p_values, d_values, q_values):

    dataset = dataset.astype('float64')

    best_score, best_cfg = float("inf"), None

    for p in p_values:

        for d in d_values:

            for q in q_values:

                order = (p,d,q)

                try:

                    mse = evaluate_arima_model(dataset, order)

                    if mse < best_score:

                        best_score, best_cfg = mse, order

#                    print('ARIMA%s MSE=%.3f' % (order,mse))

                except:

                    continue

    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

    return best_cfg



def timeseries_models(x):



    p_values = [0]

    d_values = [0, 1] #[0, 1]

    q_values = [0, 1] #[0, 1]

    warnings.filterwarnings("ignore")



    x = pd.DataFrame({'Date': x.Date, 'TargetValue': x.TargetValue})

    x.set_index('Date', inplace=True)

    pdq = evaluate_models(x, p_values, d_values, q_values)

    

    model = ARIMA(x, order= pdq)

    model_fit = model.fit(disp=0)

   

    forecasts, stderr, interval = model_fit.forecast(steps = 33, alpha = 0.05)

    maxdate = test_data.Date.max()

    mindate = '2020-01-24'

#    model_fit.plot_predict(start = mindate, end = maxdate)

    

    return forecasts, stderr, interval



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('C:\\Users\\Vincent\\Documents\\Python\\Kaggle\\Titanic'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")



#train_data.head()

#train_data.info()

#test_data.head()

#test_data.info()



# Observations

# Date is in string. Convert to datetime format

# Id is unique

# Target: ConfirmedCases and Fatalities

# Target values are counted multiple times. ie., Santa Clara, California, US has value for the same date

#   treat each of them as separate



# Convert Date to Datatime for Time Series Analysis

#train_data['Date'] = pd.to_datetime(train_data['Date'], format = '%Y-%m-%d')

#test_data['Date'] = pd.to_datetime(test_data['Date'], format = '%Y-%m-%d')



# Create unique area

test_data.loc[(test_data.County.isnull()), 'County'] = 'None'

train_data.loc[(train_data.County.isnull()), 'County'] = 'None'

test_data.loc[(test_data.Province_State.isnull()), 'Province_State'] = 'None'

train_data.loc[(train_data.Province_State.isnull()), 'Province_State'] = 'None'

test_data['Name'] = test_data['Country_Region'] + '_' + test_data['Province_State'] + '_' + test_data['County']

train_data['Name'] = train_data['Country_Region'] + '_' + train_data['Province_State'] + '_' + train_data['County']



name_list = train_data.Name.unique()

target_list = train_data.Target.unique()



#name_list = ['Uruguay_None_None', 'US_Ohio_None']

df_out = pd.DataFrame()



#for c, name in enumerate(name_list):

#    print(str(c), 'out of', str(len(name_list)), name)

#    for target in target_list:

#        print(target)

#        df_temp = pd.DataFrame()

#        train_df = train_data.loc[(train_data['Name'] == name) & (train_data['Target'] == target)]

#        forecasts, stderr, interval = timeseries_models(train_df)

#        df_temp['0.5'] = forecasts

#        df_temp['0.05'] = interval[:,[0]]

#        df_temp['0.95'] = interval[:,[1]]

#        df_temp['Name'] = name

#        df_temp['Target'] = target

#        df_temp['Date'] = pd.date_range(start="2020-05-09",end="2020-06-10")

#        df_temp['Date']=df_temp['Date'].dt.strftime('%Y-%m-%d')

#        df_out = pd.concat([df_out, df_temp])

 

## df_out.to_csv('temp_output.csv', index = False)



df_out = pd.read_csv("/kaggle/input/modeloutput/temp_output.csv")

  

# %%  

#test_data1 = test_data.loc[test_data['Name'] == 'US_Ohio_None']

test_data1 = test_data.copy()

test_data1 = test_data1[['ForecastId','Date','Name','Target']]

test_data1 = test_data1.merge(train_data[['Date','Name','Target','TargetValue']], 

        left_on=['Date','Name','Target'], right_on=['Date','Name','Target'], how='left')

test_data1['0.05'] = test_data1.TargetValue

test_data1['0.5'] = test_data1.TargetValue

test_data1['0.95'] = test_data1.TargetValue

test_data1 = test_data1.loc[test_data1['Date'] <= '2020-05-08']



test_data2 = test_data.copy()

test_data2 = test_data2.merge(df_out, left_on=['Date','Name','Target'], right_on=['Date','Name','Target'], how = 'left')

test_data2 = test_data2.loc[test_data2['Date'] > '2020-05-08']



test_data1 = test_data1[['ForecastId','0.05','0.5','0.95']]

test_data2 = test_data2[['ForecastId','0.05','0.5','0.95']]

test_data3 = pd.concat([test_data1, test_data2])

test_data3 = test_data3.melt(id_vars = ['ForecastId'])

test_data3['ForecastId_Quantile'] = test_data3['ForecastId'].astype(str)+'_'+test_data3['variable']

test_data3['TargetValue'] = test_data3['value']

test_data4 = test_data3[['ForecastId_Quantile', 'TargetValue']]

test_data4.loc[test_data4.TargetValue.isnull() , 'TargetValue'] = 0

test_data4['TargetValue'] = test_data4.TargetValue.astype(int)

test_data4.to_csv('submission.csv', index = False)

print('Done')
