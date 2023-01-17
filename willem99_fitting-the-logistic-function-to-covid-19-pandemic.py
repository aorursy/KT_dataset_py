import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as pl
import matplotlib
import random

from scipy.optimize import curve_fit
df = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
df = df.rename(columns={"Date": "datetime"})
df["datetime"] = pd.to_datetime(df["datetime"],format='%Y-%m-%d')
df = df.sort_values("datetime")
df['Province/State'] = df['Province/State'].fillna('')
df.loc[df["Country/Region"]==df['Province/State'],'Province/State'] = ''
df["Country_Province"] = df["Country/Region"] + '-' + df['Province/State']
df["Country_Province"] = df["Country_Province"].map(lambda x: x.rstrip('-'))
df.head()
# Combine all data per country
table = pd.pivot_table(df, values=["Confirmed","Deaths"], index=["datetime"],
                    columns=["Country_Province"], aggfunc=sum)

total_deaths = pd.DataFrame(table['Deaths'].iloc[-1,:][table['Deaths'].iloc[-1,:]>60].sort_values(ascending=False))
total_deaths.head(20)
# Select top 15 countries based on the number of deaths
country_list = [j for j in total_deaths.index][:15]
random.shuffle(country_list)
print(country_list)
# Find time at which a certain number of deaths is found: 
time_at = {} # Time at number of deaths 
days = (table.index - table.index[0])
death_counts = [20,30,40,50,60]
for country in country_list:
    time_at[country] = {}
    for death_count in death_counts: 
        time_at[country][death_count] = np.interp(np.log(death_count),np.log(table['Deaths'][country][table['Deaths'][country]>0]),days.days[table['Deaths'][country]>0])
        #linear: time_at[country, death_count] = np.interp(death_count,table['Deaths'][country],days.days)
days_at_death = pd.DataFrame(time_at)
days_at_death.index.name = "Deaths"

# fitting functions
def f(t, K, P0, r):
    return  (K / (1 + ((K-P0)/P0)*np.exp(-r*t)))

import random

number_of_countries = 15

# generate random color strings
colors = ['#'+str(hex(random.randint(0,256*256*256)))[2:] for k in range(number_of_countries)]

# init main graph
fig = pl.figure(figsize=(13, 9))
ax = pl.axes()

# main graph captions
pl.suptitle("2020 COVID-19 pandemic", fontweight="bold")
pl.ylabel('Cases')
pl.xlabel('Days')

country_list #.remove('US')
for c, country in enumerate(country_list): 
    print(country)
    y_values = table['Deaths'][country].values
    x_values = pd.to_timedelta(table['Deaths'].index)/pd.offsets.Day(1) - 18283 - days_at_death[country][days_at_death.index == 60].values[0]
    #x_values = x_values - x_values[0]
    k = x_values[-1]
    initvals = [50, 0.0, 1.0]
    #y_values = y_values[x_values > -10]
    #x_values = x_values[x_values > -10]
    popt, pcov = curve_fit(f, x_values, y_values, p0= initvals , maxfev=300000)

    # main fitting plot
    xx = np.linspace(x_values[0], x_values[0]+150, 150)
    yy = f(xx, popt[0], popt[1], popt[2])
    K1, P01, r1 = popt[0], popt[1], popt[2]

    popt, pcov = curve_fit(f, x_values[:-1:], y_values[:-1:], p0= initvals , maxfev=300000)
    K2, P02, r2 = popt[0], popt[1], popt[2]
    yy2 = f(xx, popt[0], popt[1], popt[2])

    popt, pcov = curve_fit(f, x_values[:-2:], y_values[:-2:], p0= initvals , maxfev=300000)
    K3, P03, r3 = popt[0], popt[1], popt[2]
    yy3 = f(xx, popt[0], popt[1], popt[2])

    pl.semilogy(x_values, y_values,'o', color=colors[c], label=country)
    pl.semilogy(xx, yy, '-', color=colors[c])
    yl = np.min(np.vstack([yy, yy2, yy3]),axis=0)
    yu = np.max(np.vstack([yy, yy2, yy3]),axis=0)
    pl.fill_between(xx, yl, yu, color=colors[c], alpha=0.3)
    print(f"{country}")
    print(f"  Growth rate r:       {[r3, r2, r1]}")
    print(f"  Carrying capacity K: {[K3, K2, K1]}")
    print(f"  P0:                  {[P03, P02, P01]}")
    print(f"  Estimated deaths:    {[np.round(yy3[-1]), np.round(yy2[-1]), np.round(yy[-1])]}")
    print()
    
pl.xlim(xx[0], xx[-1])

ax.set_ylim([1,100000])
ax.set_yticks([1,3,10,30,100,300,1000,3000,10000,30000,100000])
ax.set_ylabel('Deaths')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True)

pl.legend(loc='best')
def moving_average(a, n=5):
    #http://stackoverflow.com/questions/14313510/ddg#14314054
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#pl.style.use('fivethirtyeight')
ll = [[10, 100000], [100, 2000000]]
for q, quantity in enumerate(['Deaths','Confirmed']):
    # init main graph
    #fig = pl.figure(figsize=(16, 7))
    fig, axes = pl.subplots(figsize=(16, 7), nrows=1, ncols=2, sharex=True, sharey=True)
    fig.suptitle("2020 COVID-19 pandemic", fontweight="bold")

    for c, country in enumerate(country_list): 
        ax = axes[c//8]
        # main graph captions
        ax.set_ylabel('New '+quantity)
        ax.set_xlabel('Total '+quantity)
        y_values = table[quantity][country].values
        x_values = pd.to_timedelta(table[quantity].index)/pd.offsets.Day(1)
        x_values = x_values - x_values[0]
        k = x_values[-1]
        
        y_values = moving_average(y_values)
        ax.loglog(y_values[:-1], np.diff(y_values),'.-', label=country) #  color=colors[c], 
  
        ax.set_xlim(ll[q])
        ax.legend(loc='best')  
    
    
    for k in range(2):
        ax = axes[k]
        #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True)
    
ll = [[-10, 100], [-10, 100]]
yl = [[-10, 300], [-10, 1000]]

for q, quantity in enumerate(['Deaths','Confirmed']):
    # init main graph
    #fig = pl.figure(figsize=(16, 7))
    fig, axes = pl.subplots(figsize=(16, 7), nrows=1, ncols=2, sharex=True, sharey=True)
    fig.suptitle("2020 COVID-19 pandemic", fontweight="bold")

    for c, country in enumerate(country_list): 
        ax = axes[c//8]
        # main graph captions
        ax.set_ylabel('New '+quantity)
        ax.set_xlabel('Days since 60 Deaths')
        y_values = table[quantity][country].values
        k = x_values[-1]
        x_values = pd.to_timedelta(table['Deaths'].index)/pd.offsets.Day(1) - 18283 - days_at_death[country][days_at_death.index == 60].values[0]
        
        y_values = moving_average(y_values)
        ax.plot(x_values[5:], np.diff(y_values),'.-', label=country) #  color=colors[c], 
  

        ax.set_xlim(ll[q])
        ax.set_ylim(yl[q])
        ax.legend(loc='best')  
        
    for k in range(2):

        ax = axes[k]
        #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True)
    
