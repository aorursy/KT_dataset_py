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

import datetime 

import calendar 

import itertools

import statsmodels.api as sm

from pylab import rcParams

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

test = test.fillna('missing')
train['Date'] = pd.to_datetime(train['Date'])

train['Date'] = train['Date'].apply(lambda x: x.strftime("%d-%m-%Y"))

train['Date'] = pd.to_datetime(train['Date'])

new_train = train.copy()

new_train = new_train.fillna('missing')

new_train['new_col'] = list(zip(new_train.Country_Region,new_train.Province_State))
train['Date'] = train['Date'].apply(lambda x: x.strftime("%d-%m-%Y"))

train['Date'] = pd.to_datetime(train['Date'])

new_train = train.copy()

new_train = new_train.fillna('missing')

new_train['new_col'] = list(zip(new_train.Country_Region,new_train.Province_State))





count = 0

ConfirmedCases_submission_df = pd.DataFrame()

Fatalities_submission_df = pd.DataFrame()



for i,j in new_train['new_col'].unique().tolist():



#------------------------------------------------------------ConfirmedCases-------------------------------------------------------------------------------------

    y = new_train[['Date','ConfirmedCases']][(new_train['Country_Region'] == i) & (new_train['Province_State'] == j)].groupby('Date').agg('sum').reset_index()

    y = y.set_index('Date')

    



#     decomposition = sm.tsa.seasonal_decompose(y, model='additive')



    p = d = q = range(0, 2)

    pdq = list(itertools.product(p, d, q))

    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]



    for param in pdq:

        for param_seasonal in seasonal_pdq:

            try:

                mod = sm.tsa.statespace.SARIMAX(y,

                                                order=param,

                                                seasonal_order=param_seasonal,

                                                enforce_stationarity=False,

                                                enforce_invertibility=False)



                results = mod.fit()



    #             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

            except:

                continue



    mod = sm.tsa.statespace.SARIMAX(y,

                                    order=(1, 1, 1),

                                    seasonal_order=(1, 1, 0, 12),

                                    enforce_stationarity=False,

                                    enforce_invertibility=False)



    results = mod.fit()



    pred = results.get_prediction(start = '02-04-2020', dynamic=False)

    pred_ci = pred.conf_int().reset_index()



    pred.predicted_mean



#     y_forecasted = pred.predicted_mean

#     y_truth = y['ConfirmedCases']



    # Compute the mean square error

#     mse = ((y_forecasted - y_truth) ** 2).mean()

#     print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

#     print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))



    pred_uc = results.get_forecast(steps=40)

    pred_ci = pred_ci.append(pred_uc.conf_int().reset_index())



    d = pred_ci.copy()

    d['Country'] = i

    d['Province_State'] = j

    ConfirmedCases_submission_df = ConfirmedCases_submission_df.append(d)



    

    

#------------------------------------------------------------Fatalities-------------------------------------------------------------------------------------

    y_fatal = new_train[['Date','Fatalities']][(new_train['Country_Region'] == i) & (new_train['Province_State'] == j)].groupby('Date').agg('sum').reset_index()

    y_fatal = y_fatal.set_index('Date')

   



    p_fatal = d_fatal = q_fatal = range(0, 2)

    pdq_fatal = list(itertools.product(p_fatal, d_fatal, q_fatal))

    seasonal_pdq_fatal = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p_fatal, d_fatal, q_fatal))]



    for param in pdq_fatal:

        for param_seasonal in seasonal_pdq_fatal:

            try:

                mod_fatal = sm.tsa.statespace.SARIMAX(y_fatal,

                                                order=param,

                                                seasonal_order=param_seasonal,

                                                enforce_stationarity=False,

                                                enforce_invertibility=False)



                results_fatal = mod_fatal.fit()



    #             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

            except:

                continue



    mod_fatal = sm.tsa.statespace.SARIMAX(y_fatal,

                                    order=(1, 1, 1),

                                    seasonal_order=(1, 1, 0, 12),

                                    enforce_stationarity=False,

                                    enforce_invertibility=False)



    results_fatal = mod_fatal.fit()



    pred_fatal = results_fatal.get_prediction(start = '02-04-2020', dynamic=False)

    pred_ci_fatal = pred_fatal.conf_int().reset_index()



    pred_fatal.predicted_mean



#     y_forecasted = pred_fatal.predicted_mean

#     y_truthv = y_fatal['Fatalities']



    # Compute the mean square error

#     mse = ((y_forecasted - y_truth) ** 2).mean()

#     print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

#     print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))



    pred_uc_fatal = results_fatal.get_forecast(steps=40)

    pred_ci_fatal = pred_ci_fatal.append(pred_uc_fatal.conf_int().reset_index())



    d_fatal = pred_ci_fatal.copy()

    d_fatal['Country'] = i

    d_fatal['Province_State'] = j

    Fatalities_submission_df = Fatalities_submission_df.append(d_fatal)



    count += 1

    print("Country - ",i,j," ",count," of ", len(new_train['new_col'].unique().tolist())," completed")
ConfirmedCases_submission_df['index'] = ConfirmedCases_submission_df['index'].astype(str)

ConfirmedCases_submission_df.rename(columns={'index':'Date'},inplace = True)



Fatalities_submission_df = Fatalities_submission_df.astype(str)

Fatalities_submission_df.rename(columns={'index':'Date'},inplace = True)



submission = pd.merge(test,ConfirmedCases_submission_df,left_on=['Date','Province_State','Country_Region'],right_on=['Date','Province_State','Country'])

submission = pd.merge(submission,Fatalities_submission_df,left_on=['Date','Province_State','Country_Region'],right_on=['Date','Province_State','Country'])

submission = submission[['ForecastId','upper Fatalities','lower ConfirmedCases']]

submission.rename(columns={'ForecastId':'ForecastId','lower ConfirmedCases':'ConfirmedCases','upper Fatalities':'Fatalities'}, inplace=True)



submission = submission.replace([np.inf, -np.inf], np.nan)

submission = submission.ffill(axis = 0) 

submission = submission.fillna(0) 



submission['ConfirmedCases'] = submission['ConfirmedCases'].astype(int).apply(lambda x: int(round(x)))

submission['Fatalities'] = submission['Fatalities'].apply(lambda x: 0 if int(float(x)) <= 0 else int(round(float(x))))

# submission[submission < 0] = np.nan

# submission = submission.ffill(axis = 0) 



submission[submission < 0] = 0





submission.to_csv('submission.csv', index=False)

print(submission.shape," - Evaluation : Submission must have 13458 rows")