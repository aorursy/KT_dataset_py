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

from scipy.optimize import curve_fit

from datetime import timedelta

import math

%matplotlib inline
path = "/kaggle/input/covid19-global-forecasting-week-4/"
train_data = pd.read_csv(path+"train.csv")

#train_data = train_data[(train_data.Country_Region=="US") ]

train_df = train_data



train_df['area'] = [str(i)+str(' - ')+str(j) for i,j in zip(train_data['Country_Region'], train_data['Province_State'])]

train_df['Date'] = pd.to_datetime(train_df['Date'])

full_data = train_df
#train_df.tail(50)
today = full_data['Date'].max()+timedelta(days=1) 

print (today)

print ('Last update of this dataset was ' + str(train_df.loc[len(train_df)-1]['Date']))

def get_country_data(train_df, area, metric):

    country_data = train_df[train_df['area']==area]

    country_data = country_data.drop(['Id','Province_State', 'Country_Region'], axis=1)

    country_data = pd.pivot_table(country_data, values=['ConfirmedCases','Fatalities'], index=['Date'], aggfunc=np.sum) 

    country_data = country_data[country_data[metric]!=0]

    tmp = country_data.shift(periods=1,fill_value=0)

    country_data['prior_confirmed'] = tmp.ConfirmedCases

    country_data['prior_deaths'] = tmp.Fatalities

    country_data['DailyConfirmed'] = country_data.ConfirmedCases - country_data.prior_confirmed

    country_data['DailyFatalities'] = country_data.Fatalities - country_data.prior_deaths

    return country_data        
area_info = pd.DataFrame(columns=['area', 'cases_start_date', 'deaths_start_date', 'init_ConfirmedCases', 'init_Fatalities','init_DailyConfirmed','init_DailyFatalities'])

for i in range(len(train_df['area'].unique())):

    area = train_df['area'].unique()[i]

    area_cases_data = get_country_data(train_df, area, 'ConfirmedCases')

    #print (area_cases_data)

   

    area_deaths_data = get_country_data(train_df, area, 'Fatalities')

    cases_start_date = area_cases_data.index.min()

    deaths_start_date = area_deaths_data.index.min()

    if len(area_cases_data) > 0:

        confirmed_cases = max(area_cases_data['ConfirmedCases'])

        last = area_cases_data.tail(1)

        last_daily = np.float(last['DailyConfirmed'])



    else:

        confirmed_cases = 0

        last_daily = 0

    if len(area_deaths_data) > 0:

        fatalities = max(area_deaths_data['Fatalities'])

        last = area_deaths_data.tail(1)

        last_death = np.float(last['DailyFatalities'])

    else:

        fatalities = 0

        last_death = 0

    #print (last_daily)

    #print (last_death)

    area_info.loc[i] = [area, cases_start_date, deaths_start_date, confirmed_cases, fatalities,last_daily,last_death]

area_info = area_info.fillna(pd.to_datetime(today))

area_info['init_cases_day_no'] = pd.to_datetime(today)-area_info['cases_start_date']

area_info['init_cases_day_no'] = area_info['init_cases_day_no'].dt.days.fillna(0).astype(int)

area_info['init_deaths_day_no'] = pd.to_datetime(today)-area_info['deaths_start_date']

area_info['init_deaths_day_no'] = area_info['init_deaths_day_no'].dt.days.fillna(0).astype(int)

#area_info['init_DailyConfirmed'] = last_daily.astype(float)

#area_info['init_DailyFatalities'] = last_death.astype(float)

area_info.head()
def make_cdf (y):

    cdf = []

    for i in range(1,len(y)+1): 

        total = np.sum(y[:i])

        #print (total)

        cdf.append(total)

        #print (cdf)

    return cdf
from scipy.special import factorial

def gamma_pdf(x, k, lam, ymax):



    k = np.float(k)

    #print ('k is ' + str(k))

    

    num = ymax * (np.power(lam,k) * np.power(x,(k-1)) * np.exp(-lam*x))

    if k < 0.5:

        k = 1

    else:

         k = np.round(k)    

    den = (factorial (k-1))

    return num/den



    

def gamma_fit(train_df, area, metric,to_fit, est_count):

    area_data = get_country_data(train_df, area, metric)

    x_data = range(len(area_data.index))

    y_data = area_data[to_fit]

    x_data = np.array(x_data,dtype='float64')

    y_data = np.array(y_data,dtype='float64')

    #x_data = x_data.ravel()

    #y_data = y_data.ravel()

    #_data = np.asarray(x_data).ravel()

    #y_data = np.asarray(y_data).ravel()

    #print (y_data)

    if len(y_data) < 5:

        estimated_k = 6  

        estimated_lam = 0.1 

        ymax = np.float(est_count)

    elif max(y_data) == 0:

        estimated_k = 6  

        estimated_lam = 0.1 

        ymax = np.float(est_count)

    else:

        

        p0_est=[6.0 ,0.1,est_count]

        try:

            popt, pcov = curve_fit(gamma_pdf, x_data, y_data,bounds=([0,0,0],100000000),p0=p0_est, maxfev=1000000)

                                   #bounds=([0,0,0],100000000), p0=p0_est, maxfev=1000000)

            estimated_k, estimated_lam, ymax = popt

        except RuntimeError:

            print(area)

            print("Runtime Error - curve_fit failed") 

            estimated_k = 6  

            estimated_lam = 0.1 

            ymax = est_count

        #else:

        #    print(area)

        #    print("Catch all Error - curve_fit failed") 

        #    estimated_k = 5  

        #    estimated_lam = 0.1 

        #    ymax = est_count



    estimated_parameters = pd.DataFrame(np.array([[area, estimated_k, estimated_lam, ymax]]), columns=['area', 'k', 'lam', 'ymax'])

    return estimated_parameters
def get_parameters(metric, to_fit):

    parameters = pd.DataFrame(columns=['area', 'k', 'lam', 'ymax'], dtype=np.float)

    for area in train_df['area'].unique():

        #print ('Area fitting is ' + area)

        if metric == 'ConfirmedCases':

            init = area_info[area_info.area == area]['init_ConfirmedCases']

        else:

            init = area_info[area_info.area == area]['init_Fatalities']

        init = init.astype(float)

        #print (init)

        # establish an initial guess for maxy

        est_count = init * 4.0

        #print (est_count)

        estimated_parameters = gamma_fit(train_df, area, metric, to_fit, est_count)

        parameters = parameters.append(estimated_parameters)

    if True:

        try:

            parameters['k'] = pd.to_numeric(parameters['k'], downcast="float")

            parameters['lam'] = pd.to_numeric(parameters['lam'], downcast="float")

            parameters['ymax'] = pd.to_numeric(parameters['ymax'], downcast="float")

        except RuntimeError: 

            print ('run time error')

        except TypeError:

            print ('type error')

        #else:

        #    print ("error on parameter conversion")

        #parameters = parameters.replace({'k': {-1: parameters[parameters['ymax']>0].median()[0]}, 

        #                                 'lam': {-1: parameters[parameters['ymax']>0].median()[1]}, 

        #                                 'ymax': {-1: parameters[parameters['ymax']>0].median()[2]}})

    return parameters
cases_parameters = get_parameters('ConfirmedCases','DailyConfirmed')

cases_parameters.tail(20)
deaths_parameters = get_parameters('Fatalities','DailyFatalities')

deaths_parameters.tail(20)
fit_df = area_info.merge(cases_parameters, on='area', how='left')

fit_df = fit_df.rename(columns={"k": "cases_k", "lam": "cases_lam", "ymax": "cases_ymax"})

fit_df = fit_df.merge(deaths_parameters, on='area', how='left')

fit_df = fit_df.rename(columns={"k": "deaths_k", "lam": "deaths_lam", "ymax": "deaths_ymax"})



fit_df.head()

test_data = pd.read_csv(path+"test.csv")

test_df = test_data.copy()

#test_df = test_data[(test_data.Country_Region=="US") & (test_data.Province_State != 'x')].copy()

test_df['area'] = [str(i)+str(' - ')+str(j) for i,j in zip(test_df['Country_Region'], test_df['Province_State'])]



test_df = test_df.merge(fit_df, on='area', how='left')

test_df = test_df.merge(cases_parameters, on='area', how='left')

#print (len(test_df))



#test_df = test_df.rename(columns={"k": "cases_k", "lam": "cases_lam", "ymax": "cases_ymax"})

#test_df = test_df.merge(deaths_parameters, on='area', how='left')

#test_df = test_df.rename(columns={"k": "deaths_k", "lam": "deaths_lam", "ymax": "deaths_ymax"})

#test_df.cases_k = test_df.cases_k.astype(float)

#test_df.deaths_k = test_df.deaths_k.astype(float)

#for i,t in test_df.iterrows():

#    print (i)

#    print (t.area)

#    print (t.cases_k)

test_df['Date'] = pd.to_datetime(test_df['Date'])

test_df['cases_start_date'] = pd.to_datetime(test_df['cases_start_date'])

test_df['deaths_start_date'] = pd.to_datetime(test_df['deaths_start_date'])



test_df['cases_day_no'] = test_df['Date']-test_df['cases_start_date']

test_df['cases_day_no'] = test_df['cases_day_no'].dt.days.fillna(0).astype(int)

test_df['deaths_day_no'] = test_df['Date']-test_df['deaths_start_date']

test_df['deaths_day_no'] = test_df['deaths_day_no'].dt.days.fillna(0).astype(int)

test_df['DailyFatalities_fit'] = 0



fit_df[(fit_df.area>'US') & (fit_df.area < 'UT')]
#y = gamma_pdf(d, ['cases_k']), t['cases_lam'], t['cases_ymax'])

pred_yd = []

pred_yc = []

for (idx, df) in test_df.iterrows():

    #print('for death day ' + str(df['deaths_day_no']))

    y  = gamma_pdf(df['deaths_day_no'], df['deaths_k'], df['deaths_lam'], df['deaths_ymax'])

    #print (y)

    pred_yd.append([df.area,y])

    #print ('for confirmed day ' + str(df['cases_day_no']))

    yc = gamma_pdf(df['cases_day_no'], df['cases_k'], df['cases_lam'], df['cases_ymax'])

    pred_yc.append([df.area,yc])

    #test_df['DailyCases_pred'] = round(test_df['DailyConfirmed_fit']+test_df['DailyConfirmed_error'])



    #test_df['DailyFatalities_pred'] = round(test_df['DailyFatalities_fit']+test_df['DailyFatalities_error'])

yd_df = pd.DataFrame( pred_yd)

yc_df = pd.DataFrame( pred_yc)
yc_df.columns = ['Area','Predicted']

yd_df.columns = ['Area','Predicted']
def make_pred(df):

    cdf_all = pd.DataFrame()

    for a in df['Area'].unique():

        tmp = df[df.Area==a]

        cdf = make_cdf (tmp.Predicted)

        cdf = pd.DataFrame(cdf)

        cdf_all = pd.concat([cdf_all, cdf])

    return cdf_all



cdfc = make_pred(yc_df)

cdfd = make_pred(yd_df)

cdfc.columns =['Pred']

cdfd.columns =['Pred']





    
test_df['DailyFatalities_fit'] = cdfd.Pred.values

test_df['DailyCases_fit'] = cdfc.Pred.values
# generate submission

submission = pd.DataFrame(data={'ForecastId': test_df['ForecastId'], 'ConfirmedCases': test_df['DailyCases_fit'], 'Fatalities': test_df['DailyFatalities_fit']}).fillna(0.5)

submission.to_csv("/kaggle/working/submission.csv", index=False)
submission.head()