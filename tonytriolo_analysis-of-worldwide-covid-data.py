import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
df_us = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')

df_deaths_us = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')

df_recovered_us = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')



df_deaths_us.head()
df_global = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

df_deaths_global = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

df_recovered_global = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

df_global.head()
covid_data = {'global_cases':df_global, 'global_deaths':df_deaths_global, 'global_recovered':df_recovered_global,'us_cases':df_us, 'us_deaths':df_deaths_us, 'us_recovered':df_recovered_us }
def plot_data(data,column_name,column_val):

    if column_name=='Country/Region':

        dataframe = data['global_cases']

        deaths = data['global_deaths']

    elif column_name=='Province_State':

        dataframe = data['us_cases']

        deaths = data['us_deaths']

        

    cc = dataframe[dataframe[column_name]==column_val].iloc[:,11:].T.sum(axis = 1)

    cc = pd.DataFrame(cc)

    cc.columns = ['Cases']

    cc = cc.loc[cc['Cases']>0]



    y = np.array(cc['Cases'])

    x = np.arange(cc.size)

    dy = np.append([0],np.diff(y))

    

    cd = deaths[deaths[column_name]==column_val].iloc[:,12:].T.sum(axis = 1)

    cd = pd.DataFrame(cd)

    cd.columns = ['Deaths']

    cd = cd.loc[cd['Deaths']>0]



    yd = np.array(cd['Deaths'])

    xd = np.arange(cd.size)

    dd = np.append([0],np.diff(yd))

    

    plt.figure(1,figsize=(15,15))

    ax1=plt.subplot(221)

    ax1.semilogy(x,y)

    plt.title(column_val+' Number of cases (semilog)')

    plt.xlabel('days')

    plt.ylabel('Cumulative Cases (log)')

    ax1.minorticks_on()

    ax1.grid(which='minor',linestyle=':')

    ax1.grid(which='major',linestyle='-')

       

    ax2=plt.subplot(222)

    ax2.bar(x,dy)

    plt.title(column_val+' Daily increase in cases (linear)')

    plt.xlabel('days')

    plt.ylabel('Change in cases')

    ax2.grid()

    

    try:

        ax3=plt.subplot(223)

        ax3.loglog(y,dy,'b.')

        plt.title(column_val+' Cumulative cases vs Change in cases (loglog)')

        plt.xlabel('Cululative cases (log)')

        plt.ylabel('Change in cases (log)')

        ax3.minorticks_on()

        ax3.grid(which='minor',linestyle=':')

        ax3.grid(which='major',linestyle='-')

    except:

        print(y.size,dy.size)

        

    ax4=plt.subplot(224)

    ax4.bar(xd,dd)

    plt.title(column_val+' Daily increase in deaths (linear)')

    plt.xlabel('days')

    plt.ylabel('Change in deaths')

    ax2.grid()

        
plot_data(covid_data,'Country/Region','Korea, South')
plot_data(covid_data,'Country/Region','Italy')
plot_data(covid_data,'Country/Region','China')
plot_data(covid_data,'Country/Region','US')
plot_data(covid_data,'Province_State','New York')
plot_data(covid_data,'Province_State','New Jersey')
plot_data(covid_data,'Province_State','Washington')
plot_data(covid_data,'Province_State','California')
plot_data(covid_data,'Province_State','Florida')
plot_data(covid_data,'Country/Region','Sweden')
plot_data(covid_data,'Country/Region','Japan')
plot_data(covid_data,'Country/Region','Russia')
plot_data(covid_data,'Country/Region','Turkey')