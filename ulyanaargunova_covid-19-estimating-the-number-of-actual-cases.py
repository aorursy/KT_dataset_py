# run each cell by Shift+Enter

# daily_infected returns the number of infected in a given region under the assumptions 1-3. 



dayshift = 16



def estimation(timeseries, multiplier=100.0/4.0, shift=dayshift):      

    timeseries = multiplier * timeseries

    timeseries[:-shift] = timeseries[shift:]

    timeseries[-shift:] = 0

    return timeseries



import pandas as pd    

import numpy as np    

import matplotlib.pyplot as plt

from IPython.display import Image

global ax



%matplotlib inline 

plt.rcParams['figure.figsize'] = 14, 10





df = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv') 



countries = list(df['Country/Region'].unique())

def smooth(time_series,eps=0.2):    # this function smooths timeseries by three elements

    timeseries = np.zeros(len(time_series)+2)

    timeseries[1:-1] = np.array(time_series)

    timeseries[0], timeseries[-1] = timeseries[1], timeseries[-2]

    return pd.Series(eps*timeseries[:-2]+(1-2*eps)*timeseries[1:-1]+eps*timeseries[2:])



def smooth5(timeseries,eps=0.15):   # this function smooths timeseries by five elements

    timeseries = np.zeros(len(time_series)+4)

    timeseries[2:-2] = np.array(time_series)

    timeseries[0], timeseries[1], timeseries[-2], timeseries[-1] = timeseries[2], timeseries[2], timeseries[-3], timeseries[-3]

    return pd.Series(eps*timeseries[:-4]+(1/3-eps/3)*timeseries[1:-3]+(1/3-4*eps/3)*timeseries[2:-2]+(1/3-eps/3)*timeseries[3:-1]+eps*timeseries[4:])



def cum2daily(time_series):      # this function is an inverse of cummulative summation

    timeseries = np.zeros(len(time_series)+1)

    timeseries[1:] = np.array(time_series)

    timeseries[0] = timeseries[1]

    return pd.Series(timeseries[1:]-timeseries[:-1])



def show_with_dates(columns,dates, label):

    columns = pd.DataFrame(columns, columns=[label])

    columns['Date']=dates.reset_index()['Date'] 

    return columns.set_index('Date')



def show_est(data,vert_lines,d_rate=4.0,smooth_koef=0.0,shift_deaths_day=17,shift_confirm_day=5): 

    plt.figure(figsize=(14, 14), dpi=50)

    confirmed_per_day = estimation(cum2daily(smooth(data['Confirmed'],smooth_koef)),

                                   multiplier=4.0,shift=shift_confirm_day)

    ax = show_with_dates(confirmed_per_day[:-shift_confirm_day],

         data['Date'],'Symptomatic infections per day, estimation by confirmed').plot.line()



    infected_per_day = estimation(cum2daily(smooth(data['Deaths'],smooth_koef)),

                                  multiplier=100.00/d_rate,shift=shift_deaths_day)

    show_with_dates(infected_per_day[:-shift_deaths_day],

         data['Date'],'Symptomatic infections per day, estimation by deaths').plot.line(ax=ax)



    scale = max(confirmed_per_day)

    ax.xaxis.set_ticks(range(0,len(data['Date'])))

    ax.set_xticklabels(list(data['Date']), rotation=90)

    

    for day, y, text, color in vert_lines:

            plt.axvline(x=day,color=color)

            ax.text(day+0.5, scale * y, text,

                verticalalignment='bottom', 

                horizontalalignment='left',

                color='blue', fontsize=10)   

    plt.show()

    

    return

print('Current data set is "covid_19_clean_complete.csv" from kaggle.com by Devakumar KP')

print('data from ',df[:1]['Date'].to_string(index=False),' till ', df[-1:]['Date'].to_string(index=False))
country = 'Germany'

region_df = df[df['Country/Region']==country]

 

show_est(region_df,[[6, 0.1, ' the company Lufthansa plane suspended all flights to China (28 Jan)', 'black'],

                    [33, 0.2, 'the Light Building Trade Fair in Frankfurt was postponed until September (24 Feb)', 'black'],

                    [35, 0.3, 'North Rhine-Westphalia, Heinsberg initiated closure of schools, swimming pools, libraries and the town hall until 2 March (26 Feb)', 'black'],

                    [37, 0.35, 'Heinsberg extended closure of daycare facilities and schools to 6 March (28 Feb)', 'black'],

                    [39, 0.4, 'the number of confirmed infections almost doubled within one day (1 March)', 'black'],

                    [46, 0.45, 'the German Health Minister recommended cancelling events with more than 1000 attendees for the time being (8 March)', 'black'],

                    [47, 0.5, 'Germany reported the first deaths (9 March)', 'black'],

                    [51, 0.55, '14 of the 16 German federal states decided to close their schools and nurseries for the next few weeks (13 March)', 'black'],

                      [58, 0.6, 'Bavaria was the first state to declare a curfew (20 March)', 'black'],

                    [60, 0.65, 'the government and the federal states agreed for at least two weeks to forbid gatherings of more than two people (22 March)', 'black'],

                    [61, 0.7, 'the government decided on a financial aid package totaling around 750 billion (23 March)', 'black'],

                   ],d_rate=10.0,smooth_koef=0.0,shift_deaths_day=1,shift_confirm_day=1)

#plt.yscale('log')





# https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Germany