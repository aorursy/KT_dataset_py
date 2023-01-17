# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import datetime

from matplotlib import pyplot as plt

import matplotlib.ticker as ticker

import seaborn as sns



%matplotlib inline



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def update_covid_data():

    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

    dfconfirm = pd.read_csv(url, error_bad_lines=False,header=0)

    id_cols = dfconfirm.columns[:4]

    val_cols = dfconfirm.columns[4:]

    dfconfirm = dfconfirm.melt(id_vars=id_cols,value_vars=val_cols,var_name ='Date',value_name='ConfirmedCases')

    dfconfirm['Date']= dfconfirm['Date'].apply(lambda x : datetime.datetime.strptime(x,'%m/%d/%y'))

    dfconfirm.columns = ['State', 'Country', 'Lat', 'Long', 'Date',

           'ConfirmedCases']

    dfconfirm['LatLong'] =dfconfirm.apply(lambda x:str(x['Lat']) + ',' + str(x['Long']),axis=1)

    dfconfirm['DailyIncConirmedCases']=0

    for ll in dfconfirm['LatLong'].unique():

            dfconfirm.loc[dfconfirm['LatLong']==ll,'DailyIncConirmedCases'] = dfconfirm.loc[dfconfirm['LatLong']==ll,'ConfirmedCases'] - dfconfirm.loc[dfconfirm['LatLong']==ll,'ConfirmedCases'].shift(periods=1,fill_value=0)

    dfconfirm.drop(['Lat','Long'],inplace=True,axis=1)

    print("Confirmed database refreshed")

    

    

    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

    dfdeaths = pd.read_csv(url, error_bad_lines=False,header=0)

    id_cols = dfdeaths.columns[:4]

    val_cols = dfdeaths.columns[4:]

    dfdeaths = dfdeaths.melt(id_vars=id_cols,value_vars=val_cols,var_name ='Date',value_name='Fatalities')

    dfdeaths['Date']= dfdeaths['Date'].apply(lambda x : datetime.datetime.strptime(x,'%m/%d/%y'))

    dfdeaths.columns = ['State', 'Country', 'Lat', 'Long', 'Date',

           'Fatalities']

    dfdeaths['LatLong'] =dfdeaths.apply(lambda x:str(x['Lat']) + ',' + str(x['Long']),axis=1)

    dfdeaths['DailyFatalities']=0

    for ll in dfdeaths['LatLong'].unique():

            dfdeaths.loc[dfdeaths['LatLong']==ll,'DailyFatalities'] = dfdeaths.loc[dfdeaths['LatLong']==ll,'Fatalities'] - dfdeaths.loc[dfdeaths['LatLong']==ll,'Fatalities'].shift(periods=1,fill_value=0)



    dfdeaths.drop(['Lat','Long'],inplace=True,axis=1)

    print("fatalities database refreshed")

    

    

    

    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

    dfrecover = pd.read_csv(url, error_bad_lines=False,header=0)

    id_cols = dfrecover.columns[:4]

    val_cols = dfrecover.columns[4:]

    dfrecover = dfrecover.melt(id_vars=id_cols,value_vars=val_cols,var_name ='Date',value_name='Recovered')

    dfrecover['Date']= dfrecover['Date'].apply(lambda x : datetime.datetime.strptime(x,'%m/%d/%y'))

    dfrecover.columns = ['State', 'Country', 'Lat', 'Long', 'Date',

           'Recovered']

    dfrecover['LatLong'] =dfrecover.apply(lambda x:str(x['Lat']) + ',' + str(x['Long']),axis=1)

    dfrecover['DailyRecovered']=0

    for ll in dfrecover['LatLong'].unique():

            dfrecover.loc[dfrecover['LatLong']==ll,'DailyRecovered'] = dfrecover.loc[dfrecover['LatLong']==ll,'Recovered'] - dfrecover.loc[dfrecover['LatLong']==ll,'Recovered'].shift(periods=1,fill_value=0)

    dfrecover.drop(['Lat','Long'],inplace=True,axis=1)

    print("Recovered database refreshed")

    

    dfconfirm.set_index(['LatLong','Date'],inplace=True)

    dfdeaths.set_index(['LatLong','Date'],inplace=True)

    dfrecover.set_index(['LatLong','Date'],inplace=True)

    dfconfirm.sort_index(inplace=True)

    dfdeaths.sort_index(inplace=True)

    dfrecover.sort_index(inplace=True)

    

    



    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'

    usdeaths = pd.read_csv(url, error_bad_lines=False,header=0)

    usdeaths.drop(['UID','iso2','iso3','code3','FIPS','Admin2','Combined_Key'],axis=1,inplace=True)

    id_cols = usdeaths.columns[:5]

    val_cols = usdeaths.columns[5:]

    usdeaths = usdeaths.melt(id_vars=id_cols,value_vars=val_cols,var_name ='Date',value_name='Fatalities')

    usdeaths['Date']= usdeaths['Date'].apply(lambda x : datetime.datetime.strptime(x,'%m/%d/%y'))

    usdeaths.columns = ['State', 'Country', 'Lat', 'Long', 'Population','Date',

           'Fatalities']

    usdeaths.drop(['Lat','Long'],inplace=True,axis=1)

    usd = usdeaths.groupby(['State','Date']).sum()

    usd.sort_index(inplace=True)

    usd['DailyFatalities'] = usd['Fatalities'] - usd['Fatalities'].shift(periods=1,fill_value=0)

    usd.reset_index(inplace=True)

    print("USA fatalities database refreshed")

    

    return dfconfirm,dfdeaths,dfrecover, usd

    

    
#get daily updates of corona cases by countries

dfc,dfd,dfr,usd = update_covid_data()

alldf= pd.merge(dfc,dfr,how='inner',left_on=['LatLong','Date','State','Country'],right_on=['LatLong','Date','State','Country']).merge(dfd,how='inner',left_on=['LatLong','Date','State','Country'],right_on=['LatLong','Date','State','Country'])

alldf.reset_index(drop=False,inplace=True)

alldf.loc[alldf['State'].isnull(),'State']  = alldf.loc[alldf['State'].isnull(),'Country']
lockdf = pd.read_csv('/kaggle/input/covid19-lockdown-dates-by-country/countryLockdowndates.csv')

lockdf.drop('Reference',axis=1,inplace=True)

lockdf.columns = ['Country', 'State', 'LockdownDate', 'Type']

lockdf.loc[lockdf['State'].isnull(),'State']  = lockdf.loc[lockdf['State'].isnull(),'Country']

lockdf = alldf.merge(lockdf,how='left',on=['Country','State'])

lockdf['LockdownDate'] = lockdf['LockdownDate'].apply(lambda x: datetime.datetime.strptime(x,'%d/%m/%Y') if not pd.isna(x) else datetime.date(2020,12,31))
world_data= alldf.groupby('Date').sum()[['ConfirmedCases','Fatalities','Recovered']]

world_data = world_data.reset_index()

world_data['MortalityRate'] = np.round(world_data['Fatalities']/world_data['ConfirmedCases'],3)*100



import matplotlib.ticker as ticker

fig=plt.figure(figsize=(16,7))

ax1=sns.lineplot(x='Date',y='ConfirmedCases',data=world_data,color='b',legend='brief')



ax2 = sns.lineplot(x='Date',y='Fatalities',data=world_data,color='r',legend='brief')

ax3 = sns.lineplot(x='Date',y='Recovered',data=world_data,color='g',legend='brief')



plt.xticks(rotation=45,fontsize=10)

plt.ylabel(ylabel='Cases',fontsize=14)

plt.xlabel(xlabel='',fontsize=14)

ax4 = plt.twinx()

sns.lineplot(x='Date',y='MortalityRate',data=world_data,color='gray',legend='brief',ax=ax4)

ax4.xaxis.set_major_locator(ticker.MultipleLocator(3))

fig.legend(labels=['Confirmed Cases','Deaths Cases','Recovered Cases','Mortality Rate'],loc='upper left',fontsize=10)

plt.show()
world_data = alldf.groupby('Date').sum()['ConfirmedCases']

world_data = world_data.reset_index()



from datetime import date

from matplotlib.ticker import FormatStrFormatter

starting_date = world_data['Date'].head(1).values[0]



every_laks = {

    

    "Lakhs":[],

    "Days":[]

}

for i in range(1,100):

    

    if i > world_data['ConfirmedCases'].max() // 100000:

        break

    date = world_data[world_data['ConfirmedCases']>=i*100000].head(1)['Date'].values[0]

    daysince = date-starting_date

    daysince  = daysince.astype('timedelta64[D]') / np.timedelta64(1, 'D')

    #print("Days since first {} lakhs cases {}".format(i,daysince))

    starting_date = date

    every_laks['Lakhs'].append(i)

    every_laks['Days'].append(daysince)

    

every_laks = pd.DataFrame(every_laks)    

#fig = plt.figure(figsize=(10,8))

ax = every_laks.plot(y='Days',x='Lakhs',figsize=(16,6),color='r',alpha=0.5)



#ax = sns.lineplot(x="Days",y="Lakhs",data=every_laks)    



ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))

#ax.grid(b=False, which='minor', color='white', linestyle='--')

plt.minorticks_on()

plt.yticks(fontsize=10)

plt.legend([''])

plt.xlabel('Cases in Lakhs',fontsize=14)

plt.ylabel('Days Since every Lakh',fontsize=14)

x = every_laks['Lakhs']

y = every_laks['Days']

for i,j in zip(x,y):

    ax.annotate(str(np.int(j)) + ' Days',xy=(i,j),fontsize=9,rotation=-10)





plt.show()

country_data= alldf.query('Date>"2020-02-20"').groupby(['Country','Date']).sum()[['DailyIncConirmedCases','DailyFatalities','DailyRecovered']]

country_data = country_data.reset_index()

country_data['MortalityRate'] = np.round(country_data['DailyFatalities']/country_data['DailyIncConirmedCases'],4)*100



fig=plt.figure(figsize=(15,6))

sns.lineplot(x='Date',y='MortalityRate',data=country_data.query('Country=="Italy"'),color='b')

sns.lineplot(x='Date',y='MortalityRate',data=country_data.query('Country=="Spain"'),color='g')

sns.lineplot(x='Date',y='MortalityRate',data=country_data.query('Country=="France"'),color='r')

sns.lineplot(x='Date',y='MortalityRate',data=country_data.query('Country=="US"'),color='gray')

sns.lineplot(x='Date',y='MortalityRate',data=country_data.query('Country=="India"'),color='pink')

plt.legend(['Italy','Spain','France','USA','India'])

plt.ylabel('Mortality Rate (%)')

plt.axhline(y=5,linestyle='--')

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
dfmalaria = pd.read_csv('/kaggle/input/malaria-by-country/malaria_country.csv')

dfcumulative = alldf.groupby('Country').sum()['DailyIncConirmedCases']

dfcumulative = dfcumulative.reset_index()

dfcumulative = pd.merge(dfcumulative,dfmalaria,how='inner',left_on='Country',right_on='Location')

dfcumulative.drop(['Location','Indicator','Period'],axis=1,inplace=True)

dfcumulative.columns = ['Country', 'TotalCoronaCases',

       'TotalMalariaCases']

dfcumulative = pd.melt(dfcumulative, id_vars=['Country']).sort_values(['variable','value'])





fig=plt.figure(figsize=(13,8))

sns.barplot(x='Country', y='value', hue='variable', data=dfcumulative.query('Country!="China"'))

plt.xticks(rotation=90,fontsize=12)

plt.ylabel('Cases')

plt.title('Corona vs Malaria by Country');

plt.tight_layout()

plt.show()
import matplotlib.transforms as transforms

def visualize_lockdown(cntry):

    fig = plt.figure(figsize=(14,6))

    sns.reset_orig()

    sns.set_style('whitegrid')

    ax = sns.lineplot('Date','DailyIncConirmedCases',data= lockdf[lockdf['Country']==cntry],color='b',alpha=0.4)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    

    lockdate = lockdf[lockdf['State']==cntry]['LockdownDate'].sample(1)

    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    plt.axvline(x=lockdate,color='r',linestyle='--')

    plt.text(lockdate, 0.95, 'Lock Down Start', transform=trans,rotation=270,fontsize=13,color='red')

    recoverydate = lockdate + datetime.timedelta(days=14)

    

    plt.axvline(x=recoverydate,color='r',linestyle='--',label='True')

    plt.text(recoverydate, 0.5, 'Incubation Period Ends', transform=trans,rotation=270,fontsize=13,color='red')

    plt.ylabel('Daily Addtion of Cases')

    plt.xticks(rotation=45)

    plt.title("How lockdown impacted the addition of daily cases in " + str(cntry),fontsize=13,color='blue')

    

    plt.savefig(cntry + '.png')

    plt.show()
visualize_lockdown('Italy')
visualize_lockdown('Germany')
visualize_lockdown('Spain')
visualize_lockdown('France')
visualize_lockdown('Iran')
visualize_lockdown('United Kingdom')
visualize_lockdown('India')
visualize_lockdown('Turkey')
visualize_lockdown('Switzerland')
visualize_lockdown('Belgium')
visualize_lockdown('Netherlands')
visualize_lockdown('Austria')
sns.set_style('whitegrid')

fig = plt.figure(figsize=(20,8))

fatalus = usd.query('Date>="2020-02-20"')

fataldf = alldf.query('Date>="2020-02-20"')

ax = sns.lineplot(x='Date',y='DailyFatalities',data=fataldf[fataldf['Country']=='Italy'],color='r',linestyle='--',alpha=0.5,ci=None)

ax = sns.lineplot(x='Date',y='DailyFatalities',data=fataldf[fataldf['Country']=='France'],color='g',linestyle='--',alpha=0.5,ci=None)

ax = sns.lineplot(x='Date',y='DailyFatalities',data=fataldf[fataldf['Country']=='Spain'],color='b',linestyle='--',alpha=0.5,ci=None)

ax = sns.lineplot(x='Date',y='DailyFatalities',data=fataldf[fataldf['Country']=='Germany'],color='black',linestyle='--',alpha=0.5,ci=None)

ax = sns.lineplot(x='Date',y='DailyFatalities',data=fataldf[fataldf['Country']=='India'],color='gray',linestyle='--',alpha=0.5,ci=None)

ax = sns.lineplot(x='Date',y='DailyFatalities',data=fatalus.query('State=="New York"'),color='purple',linestyle='--',alpha=1,ci=None)

ax.xaxis.set_major_locator(ticker.MultipleLocator(3))

plt.xticks(rotation=45)

plt.legend(['Italy','France','Spain','Germany','India','New York'])

plt.tight_layout()

plt.show()