# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



filelist = []

i = 0

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        filelist.append(os.path.join(dirname, filename))

        print('file',i,':',os.path.join(dirname, filename))

        i+=1



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import datetime



import matplotlib.pyplot as plt



import statsmodels.api as sm

from linearmodels.panel import PooledOLS

from linearmodels.panel import RandomEffects

from linearmodels import PanelOLS

from linearmodels.panel import compare



from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_model import ARIMA

import itertools

import warnings

warnings.simplefilter('ignore')
df =  pd.read_csv(filelist[2])

df.drop(['Unnamed: 0'], axis=1, inplace=True)

df
X = [x for x in list(df.columns) if x not in ['Id','ConfirmedCases', 'Fatalities','Province_State', 'Country_Region','Date']] 

X
#manage date for panel regression

df.Date = pd.DatetimeIndex(df['Date'])

df.fillna({'Province_State':'All'},inplace=True) 

df['Area']=df['Country_Region']+'_'+df['Province_State']

df.set_index(['Area','Date'],inplace= True)

df
mod = PooledOLS(df['Fatalities'], sm.add_constant(df[X[:]]))

pooled_res = mod.fit(cov_type='robust')

pooled_res
mod = RandomEffects(df['Fatalities'], sm.add_constant(df[X]))

re_res = mod.fit(cov_type='robust')

re_res
mod = PanelOLS(df['Fatalities'], sm.add_constant(df[X[-2:]+X[:2]]), entity_effects=True,time_effects=True)

fe_res = mod.fit(cov_type='robust')

fe_res
compare({'FE':fe_res,'RE':re_res,'Pooled':pooled_res})
filelist
test =  pd.read_csv(filelist[6])

print(test.describe())

test = test.sort_values(by=['Country_Region','Date'])

test.Date = pd.DatetimeIndex(test['Date'])

test.fillna({'Province_State':'All'},inplace=True) 

test['Area']=test['Country_Region']+'_'+test['Province_State']

test
df =  pd.read_csv(filelist[4])

print(df.describe())

df = df.sort_values(by=['Country_Region','Date'])

df.Date = pd.DatetimeIndex(df['Date'])

df.fillna({'Province_State':'All'},inplace=True) 

df['Area']=df['Country_Region']+'_'+df['Province_State']

df
len(set(df.Area))
#Arima modeling for ts



def arima(ts):

    p=d=q=range(0,6) #set maximum of(p,d,q) to be (6,6,6)

    a=99999

    pdq=list(itertools.product(p,d,q))

    global param

    #Determining the best parameters

    for var in pdq:

        try:

            model = ARIMA(ts, order=var)

            result = model.fit()



            if (result.aic<=a) :

                a=result.aic

                param=var

        except:

            continue

    return param
optimal=pd.read_csv(filelist[1])

optimal
results=pd.DataFrame()

k = 10 #overlapping period



for i in df.Area.unique():

    print(f'Predicted Area: {i} \n')

    tstest=test.loc[(test['Area']==f'{i}')]  

    tstest=tstest[['Date']]



    ts=df.loc[(df['Area']==f'{i}')]  

    ts=ts[['Date','ConfirmedCases','Fatalities']]

    

    startdate = tstest.reset_index()['Date'].min()

    enddate = tstest.reset_index()['Date'].max()

    print('Starttest:',startdate)

    print('Endtest:',enddate)

    

    endtrain = startdate + datetime.timedelta(days=k)

    ts = ts.loc[ts['Date']<endtrain]

    

    starttrain = ts.reset_index()['Date'].min()

    endtrain = ts.reset_index()['Date'].max()



    print('\nStartTrain:',starttrain)

    print('EndTrain:',endtrain)

    

    #set date index to train data

    ts.index= ts['Date']

    print(ts)

    

    #Create empty dataframe

    product=pd.DataFrame()

    #product['Date']=pd.date_range(startdate, enddate)

    

    #Array for each country and each type

    for case in ['ConfirmedCases','Fatalities']:

        tsC = ts[case].values

        order_str = optimal.loc[optimal['Area']==f'{i}'][case].iloc[0]

        res = tuple(map(int, order_str.replace('(','').replace(')','').split(', ')))

        model = ARIMA(ts[case].dropna(), order=res)

        if ~(ts[case]==0).all():

            result = model.fit()

            pred= result.predict(start=startdate, end=enddate,typ='levels')

            fig, ax = plt.subplots()

            ax = ts[case].loc[starttrain:].plot(ax=ax)

            fig = result.plot_predict(start=startdate, end=enddate, dynamic=False, ax=ax, plot_insample=False) 

            plt.legend()

            plt.title(f'True vs predicted values: {case}')

            #if i =='Taiwan*_All':

                #plt.savefig(f'Figure\\{case}\\Taiwan_All.png',bbox_inches='tight')

            #else:

                #plt.savefig(f'Figure\\{case}\\{i}.png',bbox_inches='tight')

            plt.show()

            product[f'{case}'] = pred

            

    product['Area'] = f'{i}'    

    results = pd.concat([product,results])
#get result after looping

#results = results.reset_index().rename(index=str, columns={'index':'Date'})

#results.set_index(['Area','Date'], inplace=True)

#results.to_csv('File\\predicted_ARIMA.csv')

#results = results.reset_index()

results = pd.read_csv(filelist[3])

results.drop(['Unnamed: 0'], axis=1, inplace=True)

results.Date = pd.DatetimeIndex(results['Date'])

results
nonarima.columns
arima_re =  results.loc[~pd.isnull(results['Fatalities'])]

nonarima = results.loc[pd.isnull(results['Fatalities'])]

nonarima = pd.merge(nonarima,df, on=['Area','Date'], how='left')

nonarima['Fatalities'] = nonarima['Fatalities_y'].ffill()

nonarima['ConfirmedCases'] = nonarima['ConfirmedCases_x']

nonarima.drop(['ConfirmedCases_x', 'ConfirmedCases_y','Fatalities_x','Fatalities_y','Id','Country_Region','Province_State'], axis=1, inplace=True)

nonarima
forecast = pd.merge(test,arima_re, on=['Area','Date'], how='left')

forecast = pd.merge(forecast,nonarima, on=['Area','Date'], how='left')

forecast['Fatalities'] = forecast['Fatalities_y'].fillna(forecast['Fatalities_x'])

forecast['ConfirmedCases'] = forecast['ConfirmedCases_y'].fillna(forecast['ConfirmedCases_x'])

forecast.drop(['Province_State','Country_Region','Date','Area','ConfirmedCases_x', 'ConfirmedCases_y','Fatalities_x','Fatalities_y'], axis=1, inplace=True)

forecast.to_csv('submission.csv',index=False)

forecast