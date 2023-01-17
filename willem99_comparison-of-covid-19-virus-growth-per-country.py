# Load libraries 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # operating system
import matplotlib.pyplot as plt # plotting
import matplotlib.ticker
import matplotlib

# Add plot settings
matplotlib.rcParams.update({'font.size': 16})
#plt.style.use('dark_background')
df = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
df = df.rename(columns={"Date": "datetime"})
df["datetime"] = pd.to_datetime(df["datetime"],format='%m/%d/%y')
df = df.sort_values("datetime")
df['Province/State'] = df['Province/State'].fillna('')
df.loc[df["Country/Region"]==df['Province/State'],'Province/State'] = ''
df["Country_Province"] = df["Country/Region"] + '-' + df['Province/State']
df["Country_Province"] = df["Country_Province"].map(lambda x: x.rstrip('-'))
df.tail()
# Combine all data per country
table = pd.pivot_table(df, values=["Confirmed","Deaths"], index=["datetime"],
                    columns=["Country_Province"], aggfunc=sum)

total_deaths = pd.DataFrame(table['Deaths'].iloc[-1,:][table['Deaths'].iloc[-1,:]>60].sort_values(ascending=False))
total_deaths.head(20)
# Select top 10 countries based on the number of deaths
country_list = [j for j in total_deaths.index][:10]

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
days_at_death
fig, axes = plt.subplots(2,1,figsize=(20,20))
axis_handles = axes.ravel()
for (j,death_count) in enumerate([30,60]):
    ax1 = axis_handles[j]
    for country in country_list: 
        ax1.semilogy(days.days - time_at[country][death_count], table['Deaths'][country],'.-')
    ax1.legend(loc='upper left')
    ax1.set_xlim([-10,50])
    ax1.set_ylim([1,100000])
    ax1.set_yticks([1,3,10,30,100,300,1000,3000,10000,30000,100000])
    ax1.set_xlabel(f'Days since {death_count} deaths')
    ax1.set_ylabel('Deaths')
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.grid(True)
plt.show()
fig, axes = plt.subplots(2,1,figsize=(20,20))
axis_handles = axes.ravel()
for (j,death_count) in enumerate([30,60]):
    ax1 = axis_handles[j]
    for country in country_list: 
        ax1.semilogy(days.days - time_at[country][death_count], table['Confirmed'][country],'.-')
    ax1.legend()
    ax1.set_xlim([-5,50])
    ax1.set_ylim([100,3000000])
    ax1.set_yticks([100,300,1000,3000,10000,30000,100000,300000,1000000,3000000])
    ax1.set_xlabel(f'Days since {death_count} deaths')
    ax1.set_ylabel('Confirmed')
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.grid(True)
plt.show()
fig, ax1 = plt.subplots(1,1,figsize=(20,10))
death_count = 60 #days
for country in country_list: 
    ax1.semilogy(days.days - time_at[country][death_count], table['Deaths'][country],'.-')
ax1.legend()
ax1.set_xlim([-5,60])
ax1.set_ylim([10,100000])
ax1.set_yticks([10,30,100,300,1000,3000,10000,30000,100000])
ax1.set_xlabel(f'Days since {death_count} deaths')
ax1.set_ylabel('Deaths')
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.grid(True)
plt.show()