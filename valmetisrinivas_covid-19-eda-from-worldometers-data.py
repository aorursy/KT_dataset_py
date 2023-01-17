import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import requests

from bs4 import BeautifulSoup

from urllib.request import Request, urlopen

import seaborn as sns

from pylab import rcParams

import re

%matplotlib inline

pd.options.display.float_format = '{:.2f}'.format
req = Request('https://www.worldometers.info/coronavirus/', headers={'User-Agent': 'Firefox/75.0'})

webpage = re.sub(r'<.*?>', lambda g: g.group(0).upper(), urlopen(req).read().decode('utf-8') )
#display(webpage)#printing shows the contents of webpage
tables = pd.read_html(webpage)

#display(tables) # we can print tables
df = tables[1]
df.info()

display(df.columns.tolist())
df.head(10)
df.tail(10)
# ignore this block.. for learning purpose

# df.columns.tolist() 

# display([(i, hex(ord(i))) for i in df.columns[13]])
df = df.rename(columns={'Country,Other': 'Country_or_Other','Serious,Critical': 'Serious_or_Critical','Tot\xa0Cases/1M pop':'Cases_per_1M_pop', 'Tests/  1M pop': 'Tests_per_1M_pop','Deaths/1M pop':'Deaths_per_1M_pop','Tests/ 1M pop':'Tests_per_1M_pop'})

df['NewCases'] = df['NewCases'].str.replace(',','')

df['NewDeaths'] = df['NewDeaths'].str.replace(',','')

# df['NewRecovered'] = df['NewRecovered'].str.replace('+','')

# df['NewRecovered'] = df['NewRecovered'].str.replace(',','')
df['NewCases'] = pd.to_numeric(df['NewCases']).fillna(0)

df['NewCases'] = df['NewCases'].astype(np.int64)

df['NewDeaths'] = pd.to_numeric(df['NewDeaths']).fillna(0)

df['NewDeaths'] = df['NewDeaths'].astype(np.int64)

# df['NewRecovered'] = pd.to_numeric(df['NewRecovered']).fillna(0)

# df['NewRecovered'] = df['NewRecovered'].astype(np.int64)

df['ActiveCases'] = df['ActiveCases'].fillna(0).astype(np.int64)

df['Serious_or_Critical'] = df['Serious_or_Critical'].fillna(0).astype(np.int64)

df['TotalDeaths'] = df['TotalDeaths'].fillna(0).astype(np.int64)

df['TotalRecovered'] = df['TotalRecovered'].fillna(0).astype(np.int64)

df['TotalTests'] = df['TotalTests'].fillna(0).astype(np.int64)

df.head(10)
df.info()

df.index
df1 = df.drop(df.index[0:8]).drop(df.index[-8:])

df2 = df1[['Country_or_Other','NewCases', 'NewDeaths']]

display(df2.head())

df2.tail()
cum_data = df1.drop(columns=['NewCases','NewDeaths'])

cum_data['Dead_to_Recovered'] = 100*cum_data['TotalDeaths']/cum_data['TotalRecovered']

cum_data = cum_data.sort_values('TotalCases', ascending=False)

cum_data['TotalCases_Percent'] = 100*cum_data['TotalCases']/cum_data['TotalCases'].sum()

cum_data['TotalDeaths_Percent'] = 100*cum_data['TotalDeaths']/cum_data['TotalDeaths'].sum()

cum_data['TotalRecovered_Percent'] = 100*cum_data['TotalRecovered']/cum_data['TotalRecovered'].sum()

cum_data['TotalActive_Percent'] = 100*cum_data['ActiveCases']/cum_data['ActiveCases'].sum()

cum_data['TotalTests_Percent'] = 100*cum_data['TotalTests']/cum_data['TotalTests'].sum()

display(cum_data.columns)

cum_data.head()
cum_data[['TotalCases','TotalDeaths','TotalRecovered', 'ActiveCases','Serious_or_Critical','TotalTests']].sum()
df4 = df2.copy()

df4['Date_Time'] = pd.to_datetime('today')+ pd.DateOffset(-1)#for yesterday

df4.head()
#df4.to_csv('xxxxxxxxxxxxxxxxxxxxxxxxxxx/worldometers_covid19_uptoMay2nd2020.csv', sep =',', index=False)
df5 = pd.read_csv('https://raw.githubusercontent.com/valmetisrinivas/Covid19_Worldometers/master/worldometers_covid19_uptoMay2nd2020.csv')

df5['Date_Time'] = pd.to_datetime(df5['Date_Time'])

df5.head()
if df5['Date_Time'].dt.date.max() < df4['Date_Time'].dt.date.min():

    daily_data = df4.append(df5)

    daily_data.to_csv('xxxxxxxxxxxxxxxxxx/worldometers_covid19_uptoMay2nd2020.csv', sep =',', index=False)

else:

    daily_data = df5.copy()



daily_data['Date'] = pd.to_datetime(daily_data['Date_Time'].dt.date)

display(daily_data.shape)

daily_data['NewCases'] = daily_data['NewCases'].fillna(0).astype(np.int64)

daily_data['NewDeaths'] = pd.to_numeric(daily_data['NewDeaths']).fillna(0).astype(np.int64)

daily_data['NewRecovered'] = pd.to_numeric(daily_data['NewRecovered']).fillna(0).astype(np.int64)
daily_data.sort_values(['Date', 'NewCases'], ascending=False).drop('Date_Time', axis=1).head(30)
daily_data.sort_values(['Date', 'NewDeaths'], ascending=False).drop('Date_Time', axis=1).head(30)
select_countries = cum_data['Country_or_Other'][0:30]

select_countries
select_cum = cum_data[cum_data['Country_or_Other'].isin(select_countries)]

display(select_cum.shape)

select_daily = daily_data[daily_data['Country_or_Other'].isin(select_countries)].drop(columns = 'Date_Time')

display(select_daily.shape)

display(select_cum.head(2))

display(select_daily.head(2))
select_cum[['Country_or_Other','TotalCases','TotalDeaths']]
select_cum_percents=select_cum[['Country_or_Other','TotalCases_Percent','TotalDeaths_Percent','TotalRecovered_Percent','TotalActive_Percent','TotalTests_Percent']]

select_cum_percents
sps=pd.melt(select_cum_percents, id_vars='Country_or_Other',value_name='Percentage', var_name='Type')

sps['Type'] = sps['Type'].str.replace("_Percent","")

display(sps.head())

display(sps[sps['Type']=='TotalDeaths'].sort_values('Percentage', ascending=False).head())

display(sps[sps['Type']=='TotalRecovered'].sort_values('Percentage', ascending=False).head())
sns.set()

c=sns.catplot(data=sps,x='Type',y='Percentage',col='Country_or_Other', kind='bar', col_wrap=6)

for axes in c.axes.flat:

    axes.set_xticklabels(axes.get_xticklabels(), rotation=45, horizontalalignment='right', size=20)

c.set_titles(size=20)

c.fig.suptitle('Percentage contribution for various case type global numbers - top 30 affected countries', y=1.02, size=25)

plt.show()
rcParams['figure.figsize'] = 15, 5

fig, ax = plt.subplots()



ax.get_yaxis().get_major_formatter().set_scientific(False)



# Add a bar for the total confimred cases column 

ax.bar("Confimred", cum_data['TotalCases'].sum())

plt.text(-.1, cum_data['TotalCases'].sum() + 50000, str(cum_data['TotalCases'].sum()),fontweight='bold')



# Add a bar for the total active cases column 

ax.bar("ActiveCases", cum_data['ActiveCases'].sum())

plt.text(-.1+1, cum_data['ActiveCases'].sum() + 50000, str(cum_data['ActiveCases'].sum()),fontweight='bold')



# Add a bar for the total recovered cases column 

ax.bar("Recovered", cum_data['TotalRecovered'].sum())

plt.text(-.1+2, cum_data['TotalRecovered'].sum() + 50000, str(cum_data['TotalRecovered'].sum()),fontweight='bold')



# Add a bar for the total deaths column 

ax.bar("Deaths", cum_data['TotalDeaths'].sum())

plt.text(-.1+3, cum_data['TotalDeaths'].sum() + 50000, str(cum_data['TotalDeaths'].sum()),fontweight='bold')



# Add a bar for the total critical cases column

ax.bar("Serious_or_Critical", cum_data['Serious_or_Critical'].sum())

plt.text(-.1+4, cum_data['Serious_or_Critical'].sum() + 50000, str(cum_data['Serious_or_Critical'].sum()),fontweight='bold')



# Label the y-axis

ax.set_ylabel("Total Numbers")



# Plot title

plt.title('Total numbers across the world')



plt.show()
fig, ax = plt.subplots()



# Add a bar for the total confimred cases column with mean/std

ax.bar("Confimred", cum_data['TotalCases'].mean(), yerr=cum_data['TotalCases'].std())



# Add a bar for the total active cases column with mean/std

ax.bar("ActiveCases", cum_data['ActiveCases'].mean(), yerr=cum_data['ActiveCases'].std())



# Add a bar for the total recovered cases column with mean/std

ax.bar("Recovered", cum_data['TotalRecovered'].mean(), yerr=cum_data['TotalRecovered'].std())



# Add a bar for the total deaths column with mean/std

ax.bar("Deaths", cum_data['TotalDeaths'].mean(), yerr=cum_data['TotalDeaths'].std())



# Add a bar for the total critical cases column with mean/std

ax.bar("Serious_or_Critical", cum_data['Serious_or_Critical'].mean(), yerr=cum_data['Serious_or_Critical'].std())



# Label the y-axis

ax.set_ylabel("Numbers")



# Plot title

plt.title('Average numbers with corresponding standard deviation')



plt.show()
fig, ax = plt.subplots()

rcParams['figure.figsize'] = 15, 5



test_data = cum_data.sort_values('Tests_per_1M_pop', ascending=False)

test_data = test_data.head(50).set_index('Country_or_Other').sort_values('Tests_per_1M_pop', ascending=False).fillna(0)



# Plot a bar-chart of tests conducted per million people as a function of country

ax.bar(test_data.index,test_data['Tests_per_1M_pop'])



# Set the x-axis tick labels to the country names

ax.set_xticklabels(test_data.index, rotation = 90)



# Set the y-axis label

ax.set_ylabel("Tests Conducted/ Million")



# Plot title

plt.title('Total tests conducted - top 50 countries')



plt.show()
fig, ax = plt.subplots()

ax.get_yaxis().get_major_formatter().set_scientific(False)



# Add a bar for the total tests conducted

ax.bar("Total", cum_data['TotalTests'].sum())



# Add a bar for the tests conducted column with mean/std

ax.bar("Average & Std deviation", cum_data['TotalTests'].mean(), yerr=cum_data['TotalTests'].std())



# Label the y-axis

ax.set_ylabel("Total Numbers")



# Plot title

plt.title('Total and average (with SD) number of tests conducted across the world')



plt.show()
select_cum1 = select_cum.sort_values('TotalCases', ascending=False).set_index('Country_or_Other').fillna(0)



rcParams['figure.figsize'] = 15, 5

fig, ax = plt.subplots()



# Plot a bar-chart of total confirmed cases as a function of country

ax.bar(select_cum1.index,select_cum1['TotalCases'])



# Set the x-axis tick labels to the country names

ax.set_xticklabels(select_cum1.index, rotation = 90)



# Set the y-axis label

ax.set_ylabel("Total Confirmed Cases")



# Plot title

plt.title('Total confirmed cases - top 30 hit countries')



plt.show()
fig, ax = plt.subplots()

rcParams['figure.figsize'] = 15, 5



# Plot a histogram of "Weight" for mens_rowing

ax.boxplot([select_cum1['ActiveCases'],select_cum1['TotalDeaths'],select_cum1['TotalRecovered']])



ax.set_ylabel("Number of cases")

# Add x-axis tick labels:

ax.set_xticklabels(['Active Cases', 'Total Deaths','Total Recovered'])



# Plot title

plt.title('Distribution of various category of cases - top 30 hit countries')



plt.show()
fig, ax = plt.subplots()

rcParams['figure.figsize'] = 15, 5

select_cum1 = select_cum1.sort_values('TotalDeaths', ascending=False).fillna(0)



# Plot a bar-chart of total deaths as a function of country

ax.bar(select_cum1.index,select_cum1['TotalDeaths'])



# Set the x-axis tick labels to the country names

ax.set_xticklabels(select_cum1.index, rotation = 90)



# Set the y-axis label

ax.set_ylabel("Total Deaths")



# Plot title

plt.title('Total deaths - top 30 hit countries')



plt.show()
fig, ax = plt.subplots()

rcParams['figure.figsize'] = 15, 5

select_cum1 = select_cum1.sort_values('TotalRecovered', ascending=False).fillna(0)



# Plot a bar-chart of total recovered cases as a function of country

ax.bar(select_cum1.index,select_cum1['TotalRecovered'])



# Set the x-axis tick labels to the country names

ax.set_xticklabels(select_cum1.index, rotation = 90)



# Set the y-axis label

ax.set_ylabel("Total Recovered Cases")



# Plot title

plt.title('Total recovered cases - top 30 hit countries')



plt.show()
fig, ax = plt.subplots()

rcParams['figure.figsize'] = 15, 5

ax.get_yaxis().get_major_formatter().set_scientific(False)



# Plot a histogram of "Weight" for mens_rowing

ax.boxplot([select_cum1['TotalTests'],select_cum1['TotalCases']])



ax.set_ylabel("Number")

# Add x-axis tick labels:

ax.set_xticklabels(['Total Tests', 'Total Confirmed'])



# Plot title

plt.title('Distribution of tests conducted vs confirmed cases - top 30 hit countries')



plt.show()
fig, ax = plt.subplots()

rcParams['figure.figsize'] = 15, 5



select_cum1 = select_cum1.sort_values('Tests_per_1M_pop', ascending=False).fillna(0)



# Plot a bar-chart of test conducted per a million of population as a function of country

ax.bar(select_cum1.index,select_cum1['Tests_per_1M_pop'])



# Set the x-axis tick labels to the country names

ax.set_xticklabels(select_cum1.index, rotation = 90)



# Set the y-axis label

ax.set_ylabel("Tests Conducted/ Million")



# Plot title

plt.title('Tests conducted per million - top 30 hit countries')



plt.show()
fig, ax = plt.subplots()

rcParams['figure.figsize'] = 15, 5

select_cum1 = select_cum1.sort_values('Tests_per_1M_pop', ascending=False).fillna(0)



# Plot a bar-chart for different parameters as a function of country

ax.bar(select_cum1.index,select_cum1['Tests_per_1M_pop'],label='Tests Conducted')

ax.bar(select_cum1.index,select_cum1['Cases_per_1M_pop'],bottom=select_cum1['Tests_per_1M_pop'],label='Confirmed')

ax.bar(select_cum1.index,select_cum1['Deaths_per_1M_pop'],bottom= select_cum1['Tests_per_1M_pop']+select_cum1['Cases_per_1M_pop'],label='Dead')



# Set the x-axis tick labels to the country names

ax.set_xticklabels(select_cum1.index, rotation = 90)



# Set the y-axis label

ax.set_ylabel("Number of cases")



# Plot title

plt.title('Total per Million Population')



plt.legend()



plt.show()
rcParams['figure.figsize'] = 15, 10

fig, ax = plt.subplots()



x= select_cum1['Deaths_per_1M_pop']

y= select_cum1['Cases_per_1M_pop']

jittered_y = y + (y*.1) * np.random.rand(len(y)) -0.05

jittered_x = x + (x*.1) * np.random.rand(len(x)) -0.05



# Add data: deaths per million, cases per million to data with tests per million as color

scatter=ax.scatter(jittered_x, jittered_y, c=select_cum1['Tests_per_1M_pop'], s=select_cum1['TotalRecovered']/select_cum1['TotalDeaths'],  cmap='Paired')



# Set the x-axis label to cnfirmed cases per Million

ax.set_ylabel('Confirmed cases / Million (log scale)')



# Set the y-axis label to Deaths per Million

ax.set_xlabel('Deaths/ Million')

for i, txt in enumerate(select_cum1.index):

    ax.annotate(txt, (jittered_x[i],jittered_y[i]))

plt.title('Severity of Covid-19 Impact - top 30 hit countries')

plt.yscale('log')

legend1 = ax.legend(*scatter.legend_elements(), loc="lower right", title="Tests_per_1Million")

ax.add_artist(legend1)

handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)

legend2 = ax.legend(handles, labels, loc="lower center", title="Recovered_to_Dead")

plt.show()
select_cum1 = select_cum.set_index('Country_or_Other').sort_values('Dead_to_Recovered',ascending=False)



rcParams['figure.figsize'] = 15, 5



fig, ax = plt.subplots()



# Plot a bar-chart of dead to recovered as a function of country

ax.bar(select_cum1.index,select_cum1['Dead_to_Recovered'])



# Set the x-axis tick labels to the country names

ax.set_xticklabels(select_cum1.index, rotation = 90)



# Set the y-axis label

ax.set_ylabel("% dead against recovered")



# Plot title



plt.title('Number people dead for 100 people recovered - top 30 hit countries')

plt.show()
daily_data1 = daily_data.set_index('Date').fillna(0).groupby('Date').sum()

display(daily_data1.tail())

daily_data2 = daily_data1.cumsum()

display(daily_data2.tail())
# Define a function called plot_timeseries

def plot_timeseries(axes, x, y, color, xlabel, ylabel):



  # Plot the inputs x,y in the provided color

  axes.plot(x, y, color=color)



  # Set the x-axis label

  axes.set_xlabel(xlabel)



  # Set the y-axis label

  axes.set_ylabel(ylabel, color=color)



  # Set the colors tick params for y-axis

  axes.tick_params('y', colors=color)
fig, ax = plt.subplots()

# Plot the new daily cases time-series in blue

plot_timeseries(ax, daily_data1.index, daily_data1['NewCases'], "blue", "Date" , "Number of confirmed cases")

plt.scatter(daily_data1.index, daily_data1['NewCases'], color='b')



# Create a twin Axes object that shares the x-axis

ax2 = ax.twinx()



# Plot the new daily deaths data in red

plot_timeseries(ax2, daily_data1.index, daily_data1['NewDeaths'], "red", "Date" , "Number of deaths")

plt.scatter(daily_data1.index, daily_data1['NewDeaths'], color='r')



plt.title('Daily confirmed cases & deaths')

plt.show()
fig, ax = plt.subplots()



# Plot the new cumulative cases time-series in blue

plot_timeseries(ax, daily_data2.index, daily_data2['NewCases']+3559352, "blue", "Date" , "Cumulative no. confirmed of cases")



# Create a twin Axes object that shares the x-axis

ax2 = ax.twinx()



# Plot the new cumulative deaths data in red

plot_timeseries(ax2, daily_data2.index, daily_data2['NewDeaths']+248525, "red", "Date" , "Cumulative no. of deaths")

plt.title('Cumulative confirmed cases & deaths')

plt.show()
fig, ax = plt.subplots()

# Create a twin Axes object that shares the x-axis

ax2 = ax.twinx()



# Plot the new cumulative cases time-series in green

plot_timeseries(ax, daily_data2.index, daily_data2['NewCases']+3559352, "green", "Date" , "Cumulative no. confirmed of cases")



# Plot the new cumulative deaths data in green

plot_timeseries(ax2, daily_data2.index, daily_data2['NewDeaths']+248525, "orange", "Date" , "Cumulative no. of deaths")



# Plot the new daily cases time-series in blue

plot_timeseries(ax, daily_data1.index, daily_data1['NewCases'], "blue", "Date" , "Confirmed cases")



# Plot the new daily deaths data in red

plot_timeseries(ax2, daily_data1.index, daily_data1['NewDeaths'], "red", "Date" , "Deaths")



plt.suptitle('Daily confirmed cases (Blue) & daily deaths (Red)')

plt.title('Cumulative confirmed cases (in Green) and deaths (in Orange)')



plt.show()
fig, ax = plt.subplots()

confirmed = select_daily.fillna(0).set_index(['Date', 'Country_or_Other']).NewCases

confirmed = confirmed.unstack()

ax.plot(confirmed, marker='*')

ax.set_ylabel('Daily new confirmed cases')

plt.legend(confirmed.columns,prop={'size': 10}, loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=8)

plt.title('Daily confirmed cases - Top 30 hit nations')

plt.show()
fig, ax = plt.subplots()

deaths = select_daily.fillna(0).set_index(['Date', 'Country_or_Other']).NewDeaths

deaths = deaths.unstack()

ax.plot(deaths, marker='*')

ax.set_ylabel('Daily new deaths')

plt.legend(deaths.columns,prop={'size': 10}, loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=8)

plt.title('Daily deaths - Top 30 hit nations')

plt.show()
sns.set()

sns.relplot(x='Date', y='NewCases', data=select_daily, kind='line', ci='sd', aspect=15/5, marker='o')

plt.xticks(rotation=90)

plt.title('Average number of confirmed cases across states on a given day with associated standard deviation')

plt.show()
sns.set()

sns.relplot(x='Date', y='NewDeaths', data=select_daily, kind='line', ci='sd', aspect=15/5, marker='o')

plt.xticks(rotation=90)

plt.title('Average number of confirmed cases across states on a given day with associated standard deviation')

plt.show()