# import necessary modules

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt

import os

import warnings

#from datetime import datetime

from scipy import stats

from scipy.stats import norm, skew, probplot 



warnings.filterwarnings('ignore')
dftrain = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv', parse_dates=['Date']).sort_values(by=['Country_Region', 'Date'])

dftest = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv', parse_dates=['Date']).sort_values(by=['Country_Region', 'Date'])

dftrain.head()
confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv').sort_values(by='Country/Region')#.set_index('Country/Region')

deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')#.set_index('Country/Region')

recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')#.set_index('Country/Region')
confirmed['Country_Region'] = confirmed['Country/Region']

confirmed['Province_State'] = confirmed['Province/State']

confirmed.head()
def transpose_df(df):

    df = df.drop(['Lat','Long'],axis=1).groupby('Country/Region').sum().T

    df.index = pd.to_datetime(df.index)#.date

    return df
confirmedT = transpose_df(confirmed)

deathsT = transpose_df(deaths)

recoveredT = transpose_df(recovered)

mortalityT = deathsT/confirmedT
dftrain = dftrain.join(confirmed[['Country_Region', 'Province_State', 'Lat', 'Long']].set_index(['Province_State', 'Country_Region']), on=['Province_State', 'Country_Region'])#, how='outer')#.set_index(['Province_State', 'Country_Region']))
def add_day(df):

    df['Date'] = df.index

    df['Dayofyear'] = df['Date'].dt.dayofyear

    return df
dftrain['Dayofyear'] = dftrain['Date'].dt.dayofyear

dftest['Dayofyear'] = dftest['Date'].dt.dayofyear
allcountries = dftrain['Country_Region'].unique().tolist()
from math import radians, cos, sin, asin, sqrt



def haversine(lat1, lon1, lat2, lon2):

    """

    Calculate the great circle distance between two points 

    on the earth (specified in decimal degrees)

    """

    # convert decimal degrees to radians 

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])



    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    c = 2 * asin(sqrt(a)) 

    r = 6371 # Radius of earth in kilometers. Use 3956 for miles

    return c * r
list_countries = dftrain[dftrain['Date'] == '2020-01-22']['Country_Region'].tolist()

list_states = dftrain[dftrain['Date'] == '2020-01-22']['Province_State'].tolist()

#

df_distance = pd.DataFrame(index=[list_countries, list_states],columns=[list_countries, list_states])

df_distance.index.names = ['Country_Region', 'Province_State']

df_distance.columns.names = ['Country_Region', 'Province_State']

df_distance['Lat'] = dftrain[dftrain['Date'] == '2020-01-22']['Lat'].tolist()

df_distance['Long']  = dftrain[dftrain['Date'] == '2020-01-22']['Long'].tolist()

#

for country, state in list(zip(list_countries, list_states)):

    lat, lon = df_distance.loc[(country, state),['Lat', 'Long']]

    df_distance[(country, state)] = df_distance[['Lat', 'Long']].apply(lambda x: haversine(lat, lon, x[0], x[1]), axis=1)

#

df_distance.head()
from datetime import datetime

lockdown = confirmedT.copy()

lockdown.loc[:,:] = 0

country_lockdown = pd.DataFrame({

                    'Argentina' : datetime(2020,3,19), # 2020-03-19

                    'Australia' : datetime(2020,3,23), # 2020-03-23

                    'Austria' : datetime(2020,3,16), # 2020-03-16

                    'Belgium' : datetime(2020,3,18), # 2020-03-18

                    'Colombia' : datetime(2020,3,25), # 2020-03-25

                    'Czechia' : datetime(2020,3,16), # 2020-03-16

                    'Denmark' : datetime(2020,3,11), # 2020-03-11

                    'El Salvador' : datetime(2020,3,12), # 2020-03-12

                    'Fiji' : datetime(2020,3,9), # 2020-03-20

                    'France' : datetime(2020,3,17), # 2020-03-17

                    'Greece' : datetime(2020,3,23), # 2020-03-23

                    'Honduras' : datetime(2020,3,17), # 2020-03-17

                    'Ireland' : datetime(2020,3,12), # 2020-03-12

                    'Italy' : datetime(2020,3,9), # 2020-03-09

                    'Lebanon' : datetime(2020,3,15), # 2020-03-15

                    'Lithuania' : datetime(2020,3,16), # 2020-03-16

                    'Malaysia' : datetime(2020,3,18), # 2020-03-18

                    'Morocco' : datetime(2020,3,19), # 2020-03-19

                    'Philippines' : datetime(2020,3,15), # 2020-03-15

                    'Poland' : datetime(2020,3,13), # 2020-03-13

                    'Romania' : datetime(2020,3,25), # 2020-03-25

                    'South Africa' : datetime(2020,3,26), # 2020-03-26

                    'Spain' : datetime(2020,3,14), # 2020-03-14

                    'Tunisia' : datetime(2020,3,22), # 2020-03-22

                    'United Kingdom' : datetime(2020,3,23), # 2020-03-23

                    'Venezuela' : datetime(2020,3,17), # 2020-03-17} 

                    },index=['Start Lockdown Country']).T

state_lockdown = pd.DataFrame({

                    ('US', 'California') : datetime(2020,3,19), # 2020-03-19

                    ('US', 'Nevada') : datetime(2020,3,20), # 2020-03-20

                    ('US', 'Connecticut') : datetime(2020,3,23), # 2020-03-23

                    ('US', 'Illinois') : datetime(2020,3,21), # 2020-03-21

                    ('US', 'Massachusetts') : datetime(2020,3,24), # 2020-03-24

                    ('US', 'Michigan') : datetime(2020,3,24), # 2020-03-24

                    ('US', 'New York') : datetime(2020,3,20), # 2020-03-20

                    ('US', 'Oregon') : datetime(2020,3,24), # 2020-03-24

                   },index=['Start Lockdown State']).T

state_lockdown.head()
dftrainlockdown = dftrain.join(country_lockdown, on='Country_Region').join(state_lockdown, on=['Country_Region', 'Province_State'])

dftrainlockdown['Lockdown'] = 0

dftrainlockdown.loc[dftrainlockdown['Date'] > dftrainlockdown['Start Lockdown Country'],'Lockdown'] = 1

dftrainlockdown.loc[dftrainlockdown['Date'] > dftrainlockdown['Start Lockdown State'],'Lockdown'] = 1
pop = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv').set_index('Country (or dependency)')

pop.sort_values(by='Country (or dependency)').head()
flights = pd.read_csv('../input/datacountries/API_IS.AIR.PSGR_DS2_en_csv_v2_887266.csv').set_index('Country Name')['2018']

flights = flights.rename('FlightPassengers_2018')
dftrainall = dftrain.join(pop, on='Country_Region')

dftrainall['Lockdown'] = dftrainlockdown['Lockdown']

dftrainall = dftrainall.join(flights, on='Country_Region')

#

#

dftrainall['Mortality'] = dftrainall['Fatalities']/dftrainall['ConfirmedCases']

dftrainall['ConfirmedCases_by_pop'] = dftrainall['ConfirmedCases']/dftrainall['Population (2020)']

dftrainall['ConfirmedCases_by_Km²'] = dftrainall['ConfirmedCases']/dftrainall['Land Area (Km²)']

#

dftrainall.tail()
allcountries_ordered = confirmed.set_index(['Country_Region']).iloc[:,-2].sort_values(ascending=False).index.tolist()
confirmedT, deathsT, recoveredT, mortalityT = add_day(confirmedT), add_day(deathsT), add_day(recoveredT), add_day(mortalityT)
def df_day1(df, confirmed):

    df_day1 = pd.DataFrame({'Days since 100 cases' : np.arange(1000)}).set_index('Days since 100 cases')

    countries_df = df.columns.tolist()[:-2]

    countries_conf = confirmed.columns.tolist()[:-2]

    #print(len(countries_df), len(confirmed.columns.tolist()[:-2]))

    for ic, country in enumerate(countries_df):

        for ic2, country2 in enumerate(countries_conf):

            if country == country2:

                dfsub = df[confirmed[country] > 100.][country]

                df_day1[country] = np.nan

                df_day1.loc[:len(dfsub)-1,country] = (dfsub).tolist()

        #try:

        #except:

        #    pass

    df_day1 = df_day1.dropna(how='all')

    #df_day1 = df_day1.fillna(0.)

    return df_day1

confirmed_day1 = df_day1(confirmedT, confirmedT)

deaths_day1 = df_day1(deathsT, confirmedT)

recovered_day1 = df_day1(recoveredT, confirmedT)

mortality_day1 = df_day1(mortalityT, confirmedT)

confirmednorm_day1 = confirmed_day1/confirmed_day1.loc[0,:]

maxday = confirmed_day1.shape[0]
date_day1 = confirmedT.copy()

for column in date_day1:

    date_day1[column] = confirmedT.index.tolist()

date_day1 = df_day1(date_day1, confirmedT)

date_day1.head()
plt.figure(figsize=(15,10))

plt.subplots_adjust(wspace=0.2, hspace=0.2)

#

ylabels = ['Cumulative confirmed cases', 'Number of deceased', 'New cases per day',' Deceased per day']

ys = [[confirmedT.sum(axis=1),confirmedT.drop('China',axis=1).sum(axis=1)], 

    [deathsT.sum(axis=1),deathsT.drop('China',axis=1).sum(axis=1)],

    [confirmedT.sum(axis=1).diff().rolling(2).mean(),confirmedT.drop('China',axis=1).sum(axis=1).diff().rolling(2).mean()],

    [deathsT.sum(axis=1).diff().rolling(2).mean(),deathsT.drop('China',axis=1).sum(axis=1).diff().rolling(2).mean()],]

loglin = ['log', 'log', 'linear', 'linear']

for iy, y in enumerate(ys):

    plt.subplot(2,2,1+iy)

    plt.xticks(rotation=30)

    plt.xlabel('Date')

    plt.ylabel(ylabels[iy])

    plt.yscale(loglin[iy])

    for y2 in y:

        plt.plot(y2)

    plt.legend(['All countries', 'All except China'])

#

plt.show()
plt.figure(figsize=(15,10))

plt.subplots_adjust(wspace=0.2, hspace=0.2)

#

ylabels = ['Cumulative confirmed cases', 'Number of deceased', 'New cases per day',' Deceased per day']

ys = [confirmedT[allcountries_ordered[:11]], deathsT[allcountries_ordered[:11]],

    confirmedT[allcountries_ordered[:11]].diff().rolling(2).mean(), deathsT[allcountries_ordered[:11]].diff().rolling(2).mean()]

loglin = ['log', 'log', 'linear', 'linear']

for iy, y in enumerate(ys):

    plt.subplot(2,2,1+iy)

    plt.xticks(rotation=30)

    plt.xlabel('Date')

    plt.ylabel(ylabels[iy])

    plt.yscale(loglin[iy])

    plt.plot(y)

    plt.legend(allcountries_ordered[:11])

#

plt.show()
plt.figure(0,figsize=[20,10])

plt.subplots_adjust(wspace=0.17, hspace=0.15)

maxday = 35

#

ylabels = ['Normalised number of cases','Number of deceased','Mortality rate','Percent increase of new cases','New cases per day','Deceased per day']

ys = [confirmednorm_day1, deaths_day1, mortality_day1, confirmed_day1.diff().pct_change().rolling(1).mean()*100., confirmed_day1.diff(), deaths_day1.diff()]

loglin = ['log', 'log', 'linear', 'linear', 'linear', 'linear']

for iy, y in enumerate(ys):

    plt.subplot(2,3,iy+1)

    plt.xlabel('Number of days since 100 confirmed cases')

    plt.ylabel(ylabels[iy])

    plt.xlim(0,maxday+1)

    plt.yscale(loglin[iy])

    plt.plot(y[allcountries_ordered[:7]].rolling(2).mean())

    plt.legend(allcountries_ordered[:7])

#plt.show()

plt.savefig('cases_vs_day1.png',bbox_inches='tight',transparent=False)
#italy = pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/covid19_italy_region.csv').set_index('Date')

#italy.index = pd.to_datetime(italy.index).date

list_regions = ['Total', 'Lombardia', 'Veneto', 'Piemonte', 'Emilia Romagna', 'Toscana', 'Campania', 'Sicilia']

urlitaly = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni-'

dateini = datetime(2020,2,24)

datenow = datetime.now()

list_dates = pd.date_range(dateini, datenow).tolist()

dateitaly = ['%4s%2s%2s' % (str(date.year), str(date.month).zfill(2), str(date.day).zfill(2)) for date in list_dates[:-1]]

#

list_italy = []

for date in dateitaly:

    italy2 = pd.read_csv(urlitaly+date+'.csv').set_index('data')

    italy2.head()

    list_italy.append(italy2)

italy = pd.concat(list_italy)

italy.index = pd.to_datetime(italy.index).date

italy.head(5) 
italy_tot = italy.groupby([italy.index]).sum()

italy_tot['denominazione_regione'] = 'Total'

italy = italy.append(italy_tot)
totalcases = [] ; totaldeaths = [] ; mortality = [] ; percentincrease = [] ; casesperday = [] ; deathsperday = [] 

for region in list_regions:

    italy2 = italy[italy['denominazione_regione'] == region]

    totalcases.append(italy2['totale_casi'])

    totaldeaths.append(italy2['deceduti'])

    mortality.append(italy2['deceduti']/italy2['totale_casi'])

    percentincrease.append(italy2['totale_casi'].pct_change().rolling(3).mean())

    casesperday.append(italy2['totale_casi'].diff())

    deathsperday.append(italy2['deceduti'].diff())
plt.figure(0,figsize=[20,10])

plt.subplots_adjust(wspace=0.17, hspace=0.2)

#

ylabels = ['Number of cases','Number of deceased','Mortality rate','Percent increase of new cases','New cases per day','Deceased per day']

ys = [totalcases, totaldeaths, mortality, percentincrease, casesperday, deathsperday]

loglin = ['log', 'log', 'linear', 'linear', 'log', 'log']

for iy, y in enumerate(ys):

    plt.subplot(2,3,iy+1)

    plt.xlabel('Date')

    plt.ylabel(ylabels[iy])

    if iy == 3:

        plt.ylim(0,0.8)

    plt.xticks(rotation=30)

    #plt.xlim(0,maxday)

    plt.yscale(loglin[iy])

    for region in y:

        plt.plot(region.rolling(2).mean()) #y.rolling(2).mean())

    plt.legend(list_regions)

#plt.show()

plt.savefig('cases_vs_time_italy.png',bbox_inches='tight',transparent=False)
def create_lags(df, maxshift):

    # Shifts

    shifts = np.arange(1,maxshift+1)

    # Create a dictionary of time-shifted data

    many_shifts = {'lag_{}'.format(ii): df.shift(ii) for ii in shifts}

    # Convert them into a dataframe

    many_shifts = pd.DataFrame(many_shifts).fillna(0.)

    return many_shifts
# Accuracy metrics

from statsmodels.tsa.stattools import acf

def forecast_accuracy(forecast, actual):

    rmsle = np.sqrt(np.mean((np.log(forecast+1)-np.log(actual+1))**2))

    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # Mean Absolute Percentage Error

    me = np.mean(forecast - actual)             # ME

    mae = np.mean(np.abs(forecast - actual))    # MAE

    mpe = np.mean((forecast - actual)/actual)   # MPE

    rmse = np.mean((forecast - actual)**2)**.5  # RMSE

    corr = np.corrcoef(forecast, actual)[0,1]   # corr

    mins = np.amin(np.hstack([forecast[:,None], 

                              actual[:,None]]), axis=1)

    maxs = np.amax(np.hstack([forecast[:,None], 

                              actual[:,None]]), axis=1)

    minmax = 1 - np.mean(mins/maxs)             # minmax

    #acf1 = acf(fc-test)[1]                      # ACF1

    return({'mape':mape, 'me':me, 'mae': mae, 'rmsle' : rmsle,

            'mpe': mpe, 'rmse':rmse, #'acf1':acf1, 

            'corr':corr, 'minmax':minmax})
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import ElasticNet

#

# define list of models and parameters

list_models = [('ElasticNet', ElasticNet()),

                ('ElasticNet_wdiff', ElasticNet()),

                ('ElasticNet_wdiffwrolling', ElasticNet()),

              ]

#

list_params = [{'alpha' : np.logspace(-4,2,10),

                'l1_ratio' : np.array([0.6,0.7,0.8,0.9,1.])},

                {'alpha' : np.logspace(-4,2,10),

                'l1_ratio' : np.array([0.6,0.7,0.8,0.9,1.])},

                {'alpha' : np.logspace(-4,2,10),

                'l1_ratio' : np.array([0.6,0.7,0.8,0.9,1.])},

              ]
order_countries = date_day1.loc[0,:].sort_values().index.tolist()#[:11]

order_countries.remove('Diamond Princess')

order_countries = order_countries[:11]
import xgboost as xgb

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

#

Nlags = 5

list_inpdate = ['Dayofyear']

list_rescv = ['params', 'mean_test_score', 'std_test_score', 'mean_test_score', 'std_test_score', 'rank_test_score']

#

list_train_fc = [] ; list_valid_fc = [] ; list_train = [] ; list_valid = []

list_confirmed_train_fc = [] ; list_confirmed_valid_fc = [] ; list_confirmed_train = [] ; list_confirmed_valid = []

list_bestparams = [] ; list_bestest = [] ; list_cvres = [] ; list_inpfeat = [] ; list_error = []

df_error = pd.DataFrame(index=order_countries)

#

for im, model in enumerate(list_models): 

    list_bestparams2 = [] ; list_bestest2 = [] ; list_cvres2 = [] ; list_inpfeat2 = []

    list_train_fc2 = [] ; list_valid_fc2 = [] ; list_train2 = [] ; list_valid2 = []

    list_confirmed_train_fc2 = [] ; list_confirmed_valid_fc2 = [] ; list_confirmed_train2 = [] ; list_confirmed_valid2 = []

    list_error2 = [] ; list_index = []

    for ic, country in enumerate(order_countries): 

        print('Analyzing '+country+' with '+model[0])

        #

        # here the daily number of confirmed cases is used to make the TS more stationary, 

        # a rolling mean of 2 is also used to smooth the curves since some data are not updated on a daily basis

        inpfeature = list_inpdate + ['lag_'+str(i+1) for i in range(Nlags)] # + list_inpfeat 

        #

        if im == 0: # model directly the total number of confirmed cases

            dfcountry = confirmedT[[country, 'Dayofyear']].replace([np.inf, -np.inf], np.nan).dropna() 

        elif im == 1: # model the daily number of new confirmed cases

            dfcountry = confirmedT[[country, 'Dayofyear']].diff().rolling(1).mean().replace([np.inf, -np.inf], np.nan).dropna() 

        elif im == 2: # model the daily number of new confirmed cases averaged over a week (to smooth the data)

            dfcountry = confirmedT[[country, 'Dayofyear']].diff().rolling(7).mean().replace([np.inf, -np.inf], np.nan).dropna() 

        #

        dflag = create_lags(dfcountry[country], Nlags)

        masktrain =  dfcountry.index <= datetime(2020,3,15)

        datetrain, datevalid = dfcountry[masktrain].index[0], dfcountry[masktrain].index[-1] 

        dfconfirmedcountry = confirmedT[[country, 'Dayofyear']].replace([np.inf, -np.inf], np.nan).dropna() 

        confirmedstart = dfconfirmedcountry.loc[datetrain,country]

        confirmedend = dfconfirmedcountry.loc[datevalid,country]

        #

        # join all features

        dftrain2 = dflag[masktrain].join(dfcountry[masktrain])

        dfvalid2 = dflag[~masktrain].join(dfcountry[~masktrain])

        if ic > 0: # and im == 1

            inpfeature += order_countries[:ic]

            for ic2 in range(ic):

                dftrain2 = dftrain2.join(list_train2[ic2])

                dfvalid2 = dfvalid2.join(list_valid2[ic2])

        #

        # define training and validation sets

        X = dftrain2[inpfeature]#.drop(country, axis=1)

        y = dftrain2[country] 

        X_valid = dfvalid2[inpfeature]#.drop(country, axis=1)

        y_valid = dfvalid2[country]

        # 

        # run grid search

        param_search = list_params[im]

        tscv = TimeSeriesSplit(n_splits=5)

        gsearch = GridSearchCV(estimator=model[1],        # choice of model

                               cv=tscv,                   # choice of splitting

                               param_grid=param_search,   # grid of parameters

                               verbose=1,                 # print messages

                               return_train_score=True,   # return train score in CV grid result

                               n_jobs=-1,                 # number of CPUs to be used

                               scoring='neg_mean_squared_error' # metrics to be used

                              )

        gsearch.fit(X, y)

        #

        # save results

        list_bestparams2.append(gsearch.best_params_)

        list_bestest2.append(gsearch.best_estimator_)

        cvres = pd.DataFrame(gsearch.cv_results_)[list_rescv].sort_values(by='rank_test_score',ascending=True)

        list_cvres2.append(cvres)

        #print(cvres.head())

        print('Best-fit parameters: ',gsearch.best_params_,'\n \n')

        #

        # get the forecast TS

        #n_periods = len(valid.index)

        valid_fc = pd.Series(gsearch.predict(X_valid), index=X_valid.index) #gsearch.predict(X_valid) 

        train_fc = pd.Series(gsearch.predict(X), index=X.index) #gsearch.predict(X)

        #

        # save the forecast 

        list_train2.append(y) ; list_valid2.append(y_valid)

        list_train_fc2.append(train_fc) ; list_valid_fc2.append(valid_fc)

        list_inpfeat2.append(inpfeature)

        #

        # re-compute number of confirmed cases

        if im == 0: 

            list_confirmed_train2.append(y) ; list_confirmed_valid2.append(y_valid)

            list_confirmed_train_fc2.append(train_fc) ; list_confirmed_valid_fc2.append(valid_fc)

        elif im == 1: # 

            list_confirmed_train2.append(y.cumsum()+confirmedstart) ; list_confirmed_valid2.append(y_valid.cumsum()+confirmedend)

            list_confirmed_train_fc2.append(train_fc.cumsum()+confirmedstart) ; list_confirmed_valid_fc2.append(valid_fc.cumsum()+confirmedend)

        elif im == 2: 

            #ytrain = (confirmedT[[country, 'Dayofyear']].diff().dropna()).loc[:datetime.datetime(2020,3,15),country]

            #yvalid = (confirmedT[[country, 'Dayofyear']].diff().dropna()).loc[datetime.datetime(2020,3,15):,country]

            list_confirmed_train2.append(y.cumsum()+confirmedstart) ; list_confirmed_valid2.append(y_valid.cumsum()+confirmedend)

            list_confirmed_train_fc2.append(train_fc.cumsum()+confirmedstart) ; list_confirmed_valid_fc2.append(valid_fc.cumsum()+confirmedend)

        #

        # evaluate the error

        list_error2.append(forecast_accuracy(list_confirmed_valid_fc2[-1], list_confirmed_valid2[-1])['rmsle'])

        #list_index.append(storedept)



    #

    list_train.append(list_train2) ; list_valid.append(list_valid2)

    list_train_fc.append(list_train_fc2) ; list_valid_fc.append(list_valid_fc2)

    list_confirmed_train.append(list_confirmed_train2) ; list_confirmed_valid.append(list_confirmed_valid2)

    list_confirmed_train_fc.append(list_confirmed_train_fc2) ; list_confirmed_valid_fc.append(list_confirmed_valid_fc2)

    list_bestparams.append(list_bestparams2) ; list_bestest.append(list_bestest2) ; list_cvres.append(list_cvres2)

    list_inpfeat.append(list_inpfeat2)

    #

    df_error['RMSLE_'+model[0]] = list_error2 
df_error.head(10)
# Compare predicted and actual test TS

def plot_forecast(country, train, trainfc, valid, validfc, validfclower, validfcupper, plottitle):

    #fig, axes = plt.subplots(1, 1, figsize=(10,3), dpi=100, sharex=True)

    plt.xticks(rotation=15)

    plt.plot(train, label='Training set', color='C0')

    plt.plot(trainfc, label='Training set forecast', color='C0', ls='--')

    plt.plot(valid, label='Validation set', color='C1', ls='-')

    plt.ylabel('Confirmed cases')

    try:

        plt.fill_between(validfclower.index, 

                         validfclower, 

                         validfcupper, 

                         color='k', alpha=.15)

    except:

        pass

    plt.plot(validfc, label='Validation set forecast', color='C1', ls='--')

    plt.legend()

    plt.title("Forecast of "+str(country)+" with "+plottitle)

    #plt.show()
order_countries2 = order_countries[:11]
plt.figure(0,figsize=[20,30])

plt.subplots_adjust(wspace=0.2, hspace=0.45)

iy = 0

for ic, country in enumerate(order_countries2): #allcountries_ordered:

    for im, model in enumerate(list_models): 

        plt.subplot(len(order_countries2),len(list_models), iy+1)

        train = list_confirmed_train[im][ic] ; valid = list_confirmed_valid[im][ic]

        train_fc = list_confirmed_train_fc[im][ic] ; valid_fc = list_confirmed_valid_fc[im][ic]

        plot_forecast(country, train, train_fc, valid, valid_fc, 0., 0., model[0])

        iy += 1

plt.show()
# plot the coefficients derived by the best estimators of each model

list_models2 = list(zip([model[0] for model in list_models], list_bestest)) ; model_score = []

list_usedfeat = X.columns.tolist()

#

iy = 0

fig = plt.figure(0,figsize=[20,30])

fig.subplots_adjust(wspace=0.20, hspace=0.45)

for ic, country in enumerate(order_countries2):

    for im, model in enumerate(list_models2): 

        ax = fig.add_subplot(len(order_countries2),len(list_models), iy+1)

        bestest = list_bestest[im][ic]

        #plt.subplot(len(list_bestest),1,im+1)

        importances = pd.DataFrame(bestest.coef_, index=list_inpfeat[im][ic], columns=['Importance']).sort_values(by='Importance',ascending=False)

        ax = sns.barplot(x='Importance', y=importances.index, data=importances)

        plt.title("Coefficients of "+str(country)+" with model "+model[0])

        iy += 1

plt.show()

#plt.savefig('feature_importance.png',bbox_inches='tight',transparent=True)

#plt.close(0)g('feature_importance.png',bbox_inches='tight',transparent=True)

#plt.close(0)