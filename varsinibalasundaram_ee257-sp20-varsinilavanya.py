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

import seaborn as sns

from sklearn import preprocessing

from sklearn import linear_model

from sklearn.metrics import mean_squared_error     

import matplotlib.dates as dates

import datetime as dt
test=pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

train=pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')
train.head()
train.info()
train.describe()
test.head()

test.info()
test.describe()
confirmed_total_date_India = train[(train['Country_Region']=='India') & train['ConfirmedCases']!=0].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_India = train[(train['Country_Region']=='India') & train['ConfirmedCases']!=0].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_India = confirmed_total_date_India.join(fatalities_total_date_India)



confirmed_total_date_US = train[(train['Country_Region']=='US') & train['ConfirmedCases']!=0].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_US = train[(train['Country_Region']=='US') & train['ConfirmedCases']!=0].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_US = confirmed_total_date_US.join(fatalities_total_date_US)



confirmed_total_date_Italy = train[(train['Country_Region']=='Italy') & train['ConfirmedCases']!=0].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Italy = train[(train['Country_Region']=='Italy') & train['ConfirmedCases']!=0].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Italy = confirmed_total_date_Italy.join(fatalities_total_date_Italy)



India = [i for i in total_date_India.ConfirmedCases['sum'].values]

India_30 = India[0:70]

US = [i for i in total_date_US.ConfirmedCases['sum'].values]

US_30 = US[0:70]

italy = [i for i in total_date_Italy.ConfirmedCases['sum'].values]

italy_30 = italy[0:70] 



# Plots

plt.figure(figsize=(12,6))

plt.plot(India_30)

plt.plot(US_30)

plt.plot(italy_30)

plt.legend(["India","US", "Italy"], loc='upper left')

plt.title("COVID-19 infections from the first| confirmed case", size=15)

plt.xlabel("Days", size=13)

plt.ylabel("Infected cases", size=13)

plt.ylim(0, 300000)

plt.show()
confirmed_total_date_India = train[train['Country_Region']=='India'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_India = train[train['Country_Region']=='India'].groupby(['Date']).agg({'Fatalities':['sum']})



plt.figure(figsize=(17,10))

plt.subplot(2, 2, 1)

confirmed_total_date_India.plot(ax=plt.gca(), title='India Confirmed')

plt.ylabel("Confirmed infection cases", size=13)



plt.subplot(2, 2, 2)

fatalities_total_date_India.plot(ax=plt.gca(), title='India Fatalities')

plt.ylabel("Fatalities cases", size=13)

confirmed_total_date_Italy = train[train['Country_Region']=='Italy'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Italy = train[train['Country_Region']=='Italy'].groupby(['Date']).agg({'Fatalities':['sum']})



plt.figure(figsize=(17,10))

plt.subplot(2, 2, 1)

confirmed_total_date_Italy.plot(ax=plt.gca(), title='Italy Confirmed')

plt.ylabel("Confirmed infection cases", size=13)



plt.subplot(2, 2, 2)

fatalities_total_date_Italy.plot(ax=plt.gca(), title='Italy Fatalities')

plt.ylabel("Fatalities cases", size=13)

confirmed_total_date_US = train[train['Country_Region']=='US'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_US = train[train['Country_Region']=='US'].groupby(['Date']).agg({'Fatalities':['sum']})



plt.figure(figsize=(17,10))

plt.subplot(2, 2, 1)

confirmed_total_date_US.plot(ax=plt.gca(), title='US Confirmed')

plt.ylabel("Confirmed infection cases", size=13)



plt.subplot(2, 2, 2)

fatalities_total_date_US.plot(ax=plt.gca(), title='US Fatalities')

plt.ylabel("Fatalities cases", size=13)

train_data_by_country = train.groupby(['Date','Country_Region'],as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum' })

max_train_date = train['Date'].max()

train_data_by_country_confirm = train_data_by_country.query('(Date == @max_train_date) & (ConfirmedCases > 100)').sort_values('ConfirmedCases', ascending=False)

train_data_by_country_confirm.set_index('Country_Region', inplace=True)

display(train_data_by_country_confirm.head())



from itertools import cycle, islice

discrete_col = list(islice(cycle(['blue', 'r', 'g', 'k', 'b', 'c', 'm']), None, len(train_data_by_country_confirm.head(30))))

plt.rcParams.update({'font.size': 22})

train_data_by_country_confirm.head(20).plot(figsize=(20,15), kind='barh', color=discrete_col)

plt.legend(["Confirmed Cases", "Fatalities"]);

plt.xlabel("Covid-19 Affected")

plt.title("First 30 Countries with Highest Confirmed Cases")

ylocs, ylabs = plt.yticks()

for i, v in enumerate(train_data_by_country_confirm.head(20)["ConfirmedCases"][:]):

    plt.text(v+0.01, ylocs[i]-0.25, str(int(v)), fontsize=12)

for i, v in enumerate(train_data_by_country_confirm.head(20)["Fatalities"][:]):

    if v > 0: #disply for only >300 fatalities

        plt.text(v+0.01,ylocs[i]+0.1,str(int(v)),fontsize=12)
def getColumnInfo(df):

    n_province =  df['Province_State'].nunique()

    n_country  =  df['Country_Region'].nunique()

    n_days     =  df['Date'].nunique()

    start_date =  df['Date'].unique()[0]

    end_date   =  df['Date'].unique()[-1]

    return n_province, n_country, n_days, start_date, end_date



def reformat_time(reformat, ax):

    ax.xaxis.set_major_locator(dates.WeekdayLocator())

    ax.xaxis.set_major_formatter(dates.DateFormatter('%b %d'))    

    if reformat: #reformat again if you wish

        date_list = train_data_by_date.reset_index()["Date"].tolist()

        x_ticks = [dt.datetime.strftime(t,'%Y-%m-%d') for t in date_list]

        x_ticks = [tick for i,tick in enumerate(x_ticks) if i%8==0 ]# split labels into same number of ticks as by pandas

        ax.set_xticklabels(x_ticks, rotation=90)

    # cosmetics

    ax.yaxis.grid(linestyle='dotted')

    ax.spines['right'].set_color('none')

    ax.spines['top'].set_color('none')

    ax.spines['left'].set_color('none')

    ax.spines['bottom'].set_color('none')



train['Date'] = pd.to_datetime(train['Date'])

train_data_by_date = train.groupby(['Date'],as_index=True).agg({'ConfirmedCases': 'sum','Fatalities': 'sum'})

                                                                     

num0 = train_data_by_date._get_numeric_data() 

num0[num0 < 0.0] = 0.0





## ======= Sort by countries with fatalities > 500 ========



train_data_by_country_max = train.groupby(['Country_Region'],as_index=True).agg({'ConfirmedCases': 'max', 'Fatalities': 'max'})

train_data_by_country_fatal = train_data_by_country_max[train_data_by_country_max['Fatalities']>500]

train_data_by_country_fatal = train_data_by_country_fatal.sort_values(by=['Fatalities'],ascending=False).reset_index()

display(train_data_by_country_fatal.head(20))



df_merge_by_country = pd.merge(train,train_data_by_country_fatal['Country_Region'],on=['Country_Region'],how='inner')

df_max_fatality_country = df_merge_by_country.groupby(['Date','Country_Region'],as_index=False).agg({'ConfirmedCases': 'sum',

                                                                                                     'Fatalities': 'sum'})



                                                                                                                                                                                                       

                                                                                                    

num1 = df_max_fatality_country._get_numeric_data() 

num1[num1 < 0.0] = 0.0

df_max_fatality_country.set_index('Date',inplace=True)

#display(df_max_fatality_country.head(20))



countries = train_data_by_country_fatal['Country_Region'].unique()



plt.rcParams.update({'font.size': 16})



fig,(ax0,ax1) = plt.subplots(1,2,figsize=(15, 8))

fig,(ax2,ax3) = plt.subplots(1,2,figsize=(15, 8))#,sharey=True)



train_data_by_date.ConfirmedCases.plot(ax=ax0, x_compat=True, title='Confirmed Cases Globally', legend='Confirmed Cases',

                                       color=discrete_col)#, logy=True)

reformat_time(0,ax0)



train_data_by_date.Fatalities.plot(ax=ax2, x_compat=True, title='Fatalities Globally', legend='Fatalities', color='r')

reformat_time(0,ax2)



for country in countries:

    match = df_max_fatality_country.Country_Region==country

    df_fatality_by_country = df_max_fatality_country[match] 

    df_fatality_by_country.ConfirmedCases.plot(ax=ax1, x_compat=True, title='Cumulative Confirmed Cases Nationally')

    reformat_time(0,ax1)

    df_fatality_by_country.Fatalities.plot(ax=ax3, x_compat=True, title='Cumulative Fatalities Nationally')

    reformat_time(0,ax3)

    

ax1.legend(countries)

ax3.legend(countries)

def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state

train.rename(columns={'Country_Region':'Country'}, inplace=True)

test.rename(columns={'Country_Region':'Country'}, inplace=True)



EMPTY_VAL = "EMPTY_VAL"



train.rename(columns={'Province_State':'State'}, inplace=True)

train['State'].fillna(EMPTY_VAL, inplace=True)

train['State'] = train.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



test.rename(columns={'Province_State':'State'}, inplace=True)

test['State'].fillna(EMPTY_VAL, inplace=True)

test['State'] = test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

submission = train.loc[:, ['Country', 'State', 'ConfirmedCases', 'Fatalities']].groupby(['Country', 'State']).max().reset_index().groupby('Country').sum().sort_values(by='ConfirmedCases', ascending=False).reset_index()

submission[:35].style.background_gradient(cmap='PuBu')

submission.to_csv('submission.csv',index=False)