import numpy as np

import pandas as pd



# EDA packages

import pandas_profiling as pp



# visualization packages

import seaborn as sns

from matplotlib import pyplot as plt

import matplotlib.dates as mdates

import plotly.graph_objects as go

import pycountry

import plotly.express as px



# forecast packages

from fbprophet import Prophet



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# from this competition

train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

submission_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')



# other data sets

covid_19_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
train_df.head()
covid_19_df.head()
# check country name differences between train_df & covid_19_df



# list1 = train_df['Country_Region'].unique().tolist()

# list2 = covid_19_df['Country/Region'].unique().tolist()

# non_matches = [

#     list(set(list1).difference(list2)),

#     list(set(list2).difference(list1))

# ]

# non_matches
# standardize country names to facilitate merging datasets

cntry_rn_dic = {'Mainland China' : 'China',

                'Hong Kong' : 'China',

                'Macau': 'China',

                'UK': 'United Kingdom',

                'South Korea': 'Korea, South',

                'Taiwan':'Taiwan*',

                "('St. Martin',)":'St. Martin',

                'The Bahamas':'Bahamas, The',

                'The Gambia':'Gambia, The',

                'Czech Republic': 'Czechia',

                'Ivory Coast':"Cote d'Ivoire"

               }



covid_19_df['Country/Region'] = covid_19_df['Country/Region'].replace(cntry_rn_dic)
# standardize date format

covid_19_df['ObservationDate']=pd.to_datetime(covid_19_df['ObservationDate'])

train_df['Date']=pd.to_datetime(train_df['Date'])

test_df['Date']=pd.to_datetime(test_df['Date'])
train_df2 = pd.merge(train_df, covid_19_df.drop(['SNo','Last Update'],axis=1).drop_duplicates(), how='left',

                     left_on=['Date','Province_State','Country_Region'],

                     right_on=['ObservationDate','Province/State','Country/Region']

                    )
train_df2[['Confirmed','Deaths','Recovered']] = train_df2[['Confirmed','Deaths','Recovered']].fillna(0)

train_df2['Province_State'] = train_df2['Province_State'].fillna('Unknown')
train_df2 = train_df2.drop(['ObservationDate','Province/State','Country/Region','Confirmed','Deaths'], axis=1)

train_df2 = train_df2.drop([3601,3603,3938,3940],axis=0)
#Calculate Remaining Active Cases

train_df2['Remaining'] = train_df2['ConfirmedCases'] - train_df2['Recovered'] - train_df2['Fatalities']
# stats from previous day

prev_df = train_df2.groupby(['Country_Region','Province_State']).shift(1)[['ConfirmedCases','Fatalities','Recovered','Remaining']]

prev_df.columns = ['ConfirmedCases_lag1','Fatalities_lag1','Recovered_lag_1','Remaining_lag1']
train_df2 = pd.merge(train_df2, prev_df, how='left', left_index=True, right_index=True)
# calculate daily incrementals

train_df2['Incr_Confirmed'] = train_df2['ConfirmedCases'] - train_df2['ConfirmedCases_lag1']

train_df2['Incr_Fatalities'] = train_df2['Fatalities'] - train_df2['Fatalities_lag1']

train_df2['Incr_Recovered'] = train_df2['Recovered'] - train_df2['Recovered_lag_1']

train_df2['Incr_Remaining'] = train_df2['Remaining'] - train_df2['Remaining_lag1']

train_df2 = train_df2.drop(['ConfirmedCases_lag1','Fatalities_lag1','Recovered_lag_1','Remaining_lag1'],axis=1)

train_df2[['Incr_Confirmed','Incr_Fatalities','Incr_Recovered','Incr_Remaining']

         ] = train_df2[['Incr_Confirmed','Incr_Fatalities','Incr_Recovered','Incr_Remaining']].fillna(0)
# Identify Date Order since first confirmed case

train_df2['InRange'] = train_df2['ConfirmedCases'].ne(0)

train_df2['Infected_Date_Order'] = train_df2.groupby(['Province_State','Country_Region','InRange'])['Date'].rank("dense", ascending=True)

train_df2['Infected_Date_Order'] =np.where(train_df2['InRange']==False,0,train_df2['Infected_Date_Order'])

train_df2 = train_df2.drop('InRange',axis=1)
train_df2
gc.collect()
# split training data into contry/province

dfs=[]

i=0

for region, df_region in train_df2.groupby(['Country_Region','Province_State']):

    df=df_region

    df.name=str(region)

    df=df.rename(columns={'Date': 'ds', 

                          'Incr_Confirmed': 'y'})

    dfs.append(df)

    #print(region)

    #print(df_region)

    i=+1
%%time

outputs = []

for df in dfs:

    train = df[df.ds < '2020-03-19']

    test = df[df.ds >= '2020-03-19']



    model = Prophet(interval_width=0.95, 

                #weekly_seasonality=False, 

                #daily_seasonality=True,

                #holidays=holidays,

                changepoint_prior_scale=14,

                seasonality_mode='multiplicative')



    # model.add_seasonality(name='biweekly', period=14, fourier_order=100, mode = 'multiplicative')



    # reg_var = ['Incr_Confirmed', 'Remaining']



    #for var in reg_var:

        #model.add_regressor(var)



    model.fit(train)

    future = model.make_future_dataframe(freq='D',periods = 43)

    forecast = model.predict(future)

    forecast['Province_State'] = train['Province_State'].values[1]

    forecast['Country_Region'] = train['Country_Region'].values[1]

    out=forecast[['ds','yhat','Province_State','Country_Region']]

    #out = pd.merge(test[['Id', 'Province_State', 'Country_Region', 'ds','y']],forecast[['ds','yhat']],how='left')

    outputs.append(out)
yhat_conf_df = pd.concat(outputs).reset_index()

yhat_conf_df = yhat_conf_df.rename(columns={'ds':'Date', 'yhat':'Incr_Confirmed'}).drop('index',axis=1)
gc.collect()
# split training data into contry/province

dfs=[]

i=0

for region, df_region in train_df2.groupby(['Country_Region','Province_State']):

    df=df_region

    df.name=str(region)

    df=df.rename(columns={'Date': 'ds', 

                          'Incr_Fatalities': 'y'})

    dfs.append(df)

    #print(region)

    #print(df_region)

    i=+1
%%time

outputs = []

for df in dfs:

    train = df[df.ds < '2020-03-19']

    test = df[df.ds >= '2020-03-19']



    model = Prophet(interval_width=0.95, 

                changepoint_prior_scale=14,

                seasonality_mode='multiplicative')



    model.fit(train)

    

    future = model.make_future_dataframe(freq='D',periods = 43)

    forecast = model.predict(future)

    forecast['Province_State'] = train['Province_State'].values[1]

    forecast['Country_Region'] = train['Country_Region'].values[1]

    out=forecast[['ds','yhat','Province_State','Country_Region']]



    outputs.append(out)
yhat_fatal_df = pd.concat(outputs).reset_index()

yhat_fatal_df = yhat_fatal_df.rename(columns={'ds':'Date', 'yhat':'Incr_Fatalities'}).drop('index',axis=1)
test_df.Province_State = test_df.Province_State.fillna('Unknown')



pred_df = pd.merge(yhat_conf_df,yhat_fatal_df)

test_df = pd.merge(test_df,pred_df, how='left')
gc.collect()
# split training data into contry/province

dfs=[]

i=0

for region, df_region in train_df2.groupby(['Country_Region','Province_State']):

    df=df_region

    df.name=str(region)

    df=df.rename(columns={'Date': 'ds', 

                          'ConfirmedCases': 'y'})

    dfs.append(df)

    #print(region)

    #print(df_region)

    i=+1
%%time

outputs = []

for df in dfs:

    train = df[df.ds < '2020-03-19']

    test = df[df.ds >= '2020-03-19']



    model = Prophet(interval_width=0.95, 

                changepoint_prior_scale=14,

                seasonality_mode='multiplicative')

    

    reg_var = ['Incr_Confirmed']



    for var in reg_var:

        model.add_regressor(var)



    model.fit(train)

    

    #future = model.make_future_dataframe(freq='D',periods = 43)

    #future = pd.merge(future,test_df[['ds']+reg_var],how='left', left)

    future = test_df[(test_df['Province_State']==train['Province_State'].values[1]) &

                     (test_df['Country_Region']==train['Country_Region'].values[1])]

    future=future.rename(columns={'Date':'ds'})

    

    forecast = model.predict(future)

    forecast['ForecastId'] = future['ForecastId'].values

    forecast['Province_State'] = train['Province_State'].values[1]

    forecast['Country_Region'] = train['Country_Region'].values[1]

    out=forecast[['ForecastId','ds','yhat','Province_State','Country_Region']]



    outputs.append(out)
pred_conf = pd.concat(outputs).reset_index()

pred_conf = pred_conf.rename(columns={'yhat':'ConfirmedCases'}).drop('index',axis=1)
gc.collect()
# split training data into contry/province

dfs=[]

i=0

for region, df_region in train_df2.groupby(['Country_Region','Province_State']):

    df=df_region

    df.name=str(region)

    df=df.rename(columns={'Date': 'ds', 

                          'Fatalities': 'y'})

    dfs.append(df)

    #print(region)

    #print(df_region)

    i=+1
%%time

outputs = []

for df in dfs:

    train = df[df.ds < '2020-03-19']

    test = df[df.ds >= '2020-03-19']



    model = Prophet(interval_width=0.95, 

                changepoint_prior_scale=14,

                seasonality_mode='multiplicative')

    

    reg_var = ['Incr_Fatalities']



    for var in reg_var:

        model.add_regressor(var)



    model.fit(train)

    

    future = test_df[(test_df['Province_State']==train['Province_State'].values[1]) &

                     (test_df['Country_Region']==train['Country_Region'].values[1])]

    future=future.rename(columns={'Date':'ds'})

    

    forecast = model.predict(future)

    forecast['ForecastId'] = future['ForecastId'].values

    forecast['Province_State'] = train['Province_State'].values[1]

    forecast['Country_Region'] = train['Country_Region'].values[1]

    out=forecast[['ForecastId','ds','yhat','Province_State','Country_Region']]



    outputs.append(out)
pred_fatal = pd.concat(outputs).reset_index()

pred_fatal = pred_fatal.rename(columns={'yhat':'Fatalities'}).drop('index',axis=1)
gc.collect()
submission_df = pd.merge(pred_conf,pred_fatal)[['ForecastId','ConfirmedCases','Fatalities']]

submission_df.to_csv("submission.csv",index=False)
submission_df.sample(20)
'''train = dfs[50][dfs[50].ds <= '2020-03-19'].iloc[1:]

test = dfs[50][dfs[50].ds > '2020-03-19']



model = Prophet(interval_width=0.95, 

                #weekly_seasonality=False, 

                #daily_seasonality=True,

                #holidays=holidays,

                changepoint_prior_scale=14,

                seasonality_mode='multiplicative')



# model.add_seasonality(name='biweekly', period=14, fourier_order=100, mode = 'multiplicative')



reg_var = ['Incr_Confirmed', 'Remaining']



#for var in reg_var:

#    model.add_regressor(var)



model.fit(train)'''
'''future = model.make_future_dataframe(freq='D',periods = 42)

#future = pd.merge(future,test[['ds']+reg_var],how='left')

#for var in reg_var:

#    forecast[var] = test[var]

forecast = model.predict(future)

#forecast = model.predict(test[['ds']+reg_var])'''
'''fig1 = model.plot(forecast, xlabel=u'Date', ylabel=u'Expected contract inception').set_size_inches(10,5)

fig2 = plt.plot(test.ds, test.y, marker = 'o',color='darkred')

plt.title('GAM prediction interval', fontsize=20)

plt.show()'''
'''model.plot_components(forecast).set_size_inches(10,5)'''