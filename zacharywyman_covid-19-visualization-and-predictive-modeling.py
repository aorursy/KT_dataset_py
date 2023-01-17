# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import required libraries.

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import numpy as np

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import plotly.express as px

from datetime import datetime as dt

from datetime import timedelta

#from pytrends.request import TrendReq

from fbprophet import Prophet
# get data

data = pd.read_csv(r'/kaggle/input/covid19-visualization-and-analysis/covid.csv', error_bad_lines = False)



recovered = pd.read_csv(r'/kaggle/input/covid19-visualization-and-analysis/recovered.csv', error_bad_lines = False)

recovered.columns = ['date', 'Recovered']

recovered.index = pd.to_datetime(recovered['date'])

recovered = recovered['Recovered'].astype('int64')

recovered
# clean dataframes 

data.drop('Unnamed: 0',axis = 1, inplace = True)


cases = data[['date', 'state', 'cases', 'deaths']]
us_states = {

        'Alaska': 'AK',

        'Alabama': 'AL',

        'Arkansas': 'AR',

        'American Somoa': 'AS',

        'Arizona': 'AZ',

        'California': 'CA',

        'Colorado': 'CO',

        'Connecticut': 'CT',

        'District of Colombia': 'DC',

        'Delaware': 'DE',

        'Florida': 'FL',

        'Georgia': 'GA',

        'Hawaii': 'HI',

        'Iowa': 'IA',

        'Idaho': 'ID',

        'Illinois': 'IL',

        'Indiana': 'IN',

        'Kansas': 'KS',

        'Kentucky': 'KY',

        'Louisiana': 'LA',

        'Massachusetts': 'MA',

        'Maryland': 'MD',

        'Maine': 'ME',

        'Michigan': 'MI',

        'Minnesota': 'MN',

        'Missouri': 'MO',

        'Mississippi': 'MS',

        'Montana': 'MT',

        'National': 'NA',

        'North Carolina': 'NC',

        'North Dakota': 'ND',

        'Nebraska': 'NE',

        'New Hampshire': 'NH',

        'New Jersey': 'NJ',

        'New Mexico': 'NM',

        'Nevada': 'NV',

        'New York': 'NY',

        'Ohio': 'OH',

        'Oklahoma': 'OK',

        'Oregon': 'OR',

        'Pennsylvania': 'PA',

        'Puerto Rico': 'PR',

        'Rhode Island': 'RI',

        'South Carolina': 'SC',

        'South Dakota': 'SD',

        'Tennessee': 'TN',

        'Texas': 'TX',

        'Utah': 'UT',

        'Virginia': 'VA',

        'Virgin Islands': 'VI',

        'Vermont': 'VT',

        'Washington': 'WA',

        'Wisconsin': 'WI',

        'West Virginia': 'WV',

        'Wyoming': 'WY'

}
cases['abbrev'] = cases['state'].map(us_states).fillna(cases['state'])

cases


fig = px.choropleth(cases,

                   locations='abbrev',

                   color='cases',

                   hover_name = 'abbrev',

                   locationmode = 'USA-states',

                   animation_frame = 'date')



fig.update_layout(

title_text = 'Spread of Covid-19 in the United States',

title_x = 0.5,

geo_scope = 'usa',

geo=dict(

showframe = False,

showcoastlines = False))



fig.show()
pop_data = pd.read_csv(r'/kaggle/input/covid19-visualization-and-analysis/pop_data.csv')
#merge population to dataframe.

cases = cases.merge(pop_data,

          how = 'left',

          left_on = 'state',

          right_on = 'state')

cases
#creating percentage column in dataframe.

cases['percent'] = cases['cases'] / cases['population'] * 100
fig = px.choropleth(cases,

                   locations='abbrev',

                   color='percent',

                   hover_name = 'abbrev',

                   locationmode = 'USA-states',

                   animation_frame = 'date')



fig.update_layout(

title_text = 'Spread of Covid-19 in the United States (%)',

title_x = 0.5,

geo_scope = 'usa',

geo=dict(

showframe = False,

showcoastlines = False))



fig.show()
sorted_cases = cases.sort_values('cases', ascending=False)

state_max_cases = sorted_cases.drop_duplicates('state')

state_max_cases.drop('abbrev', axis=1, inplace=True)
#top 10 states with most cases.

cases_top10 = state_max_cases.nlargest(10, 'cases')



#top 10 states with highest severity of cases.

percent_top10 = state_max_cases.nlargest(10, 'percent')


fig, axes = plt.subplots(1, 2, sharex = False, sharey = False, figsize = (20,6))

fig.suptitle("Covid-19 Case Statistics", fontsize = 20)

axes[0].set_title('Top 10 States with Most Cases', fontsize = 14)

axes[0].set_xlabel('cases', fontsize = 12)

axes[0].set_ylabel('state', fontsize = 12)

axes[1].set_title('Top 10 States with Highest Severity', fontsize = 14)

axes[1].set_xlabel('percent', fontsize = 12)

axes[1].set_ylabel('state', fontsize = 12)



sns.barplot(ax = axes[0], data = cases_top10, 

            y = 'state', 

            x = 'cases')



sns.barplot(ax = axes[1], data = percent_top10, 

            y = 'state', 

            x = 'percent')
#add countplot.

fig, ax = plt.subplots(ncols=1, sharey = False, figsize = (10,20))

plt.xticks(rotation=90)

sns.barplot(y = state_max_cases['state'], x = state_max_cases['cases']).set_title('Statewide Covid-19 Cases')
#death rate.

percent_deaths = cases['deaths'] / cases['population'] * 100

cases['death rate'] = percent_deaths
#choro map

fig = px.choropleth(cases,

                   locations='abbrev',

                   color='deaths',

                   hover_name = 'abbrev',

                   locationmode = 'USA-states',

                   animation_frame = 'date')



fig.update_layout(

title_text = 'Covid-19 Deaths in the United States',

title_x = 0.5,

geo_scope = 'usa',

geo=dict(

showframe = False,

showcoastlines = False))



fig.show()
fig = px.choropleth(cases,

                   locations='abbrev',

                   color='death rate',

                   hover_name = 'abbrev',

                   locationmode = 'USA-states',

                   animation_frame = 'date')



fig.update_layout(

title_text = 'Covid-19 Death Rate in the United States',

title_x = 0.5,

geo_scope = 'usa',

geo=dict(

showframe = False,

showcoastlines = False))



fig.show()
sorted_deaths = cases.sort_values('deaths', ascending = False)

state_max_deaths = sorted_deaths.drop_duplicates('state')

state_max_deaths.drop('abbrev', axis=1, inplace=True)
#top 10 states with most deaths.

deaths_top10 = state_max_deaths.nlargest(10, 'deaths')



#top 10 states with highest death rate.

death_ratetop10 = state_max_deaths.nlargest(10, 'death rate')
fig, axes = plt.subplots(1, 2, sharex = False, sharey = False, figsize = (20,6))

fig.suptitle("Covid-19 Death Statistics", fontsize = 20)

axes[0].set_title('Top 10 States with Most Deaths', fontsize = 14)

axes[0].set_xlabel('cases', fontsize = 12)

axes[0].set_ylabel('state', fontsize = 12)

axes[1].set_title('Top 10 States with Highest Severity', fontsize = 14)

axes[1].set_xlabel('percent', fontsize = 12)

axes[1].set_ylabel('state', fontsize = 12)



sns.barplot(ax = axes[0], data = deaths_top10, 

            y = 'state', 

            x = 'deaths')



sns.barplot(ax = axes[1], data = death_ratetop10, 

            y = 'state', 

            x = 'death rate')
#add countplot. 

fig, ax = plt.subplots(ncols=1, sharey = False, figsize = (10,20))

plt.xticks(rotation=90)

sortedDeaths = state_max_cases.sort_values(by = ['deaths'], ascending = False)

sns.barplot(y = sortedDeaths['state'], x = sortedDeaths['deaths']).set_title('Statewide Covid-19 Deaths')


date_deaths = data[['date', 'deaths']]

deaths_by_date = date_deaths.groupby('date')['deaths'].sum()



date_cases = data[['date', 'cases']]

cases_by_date = date_cases.groupby('date')['cases'].sum()


deaths_by_date.index = pd.to_datetime(deaths_by_date.index)

months = mdates.MonthLocator()



#plot deaths over time.

fig, ax = plt.subplots(figsize = (16,8))

marker_style = dict(linewidth=2.5, linestyle = '-', marker = 'o', markersize = 3)

ax.plot(deaths_by_date, **marker_style)

plt.ylabel("Deaths", fontsize = 12)

plt.title('Covid-19 Deaths in the United States', fontsize = 14)



#format ticks

ax.xaxis.set_major_locator(months)

ax.grid(True)



plt.show()


cases_by_date.index = pd.to_datetime(cases_by_date.index)



#plot cases over time.

fig, ax = plt.subplots(figsize = (16,8))

marker_style = dict(linewidth=2.5, linestyle = '-', marker = 'o', markersize = 3)

ax.plot(cases_by_date, **marker_style)

plt.ylabel('Cases', fontsize = 12)

plt.title('Covid-19 Cases in the United States', fontsize = 14)



#formatting 

ax.xaxis.set_major_locator(months)

ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

ax.format_ydata = lambda x: '$%1.2f' % x  

ax.grid(True)



plt.show()
recovered_by_date = pd.DataFrame(recovered)

recovered_by_date.index = pd.to_datetime(recovered_by_date.index)



#plot cases over time.

fig, ax = plt.subplots(figsize = (16,8))

marker_style = dict(linewidth=2.5, linestyle = '-', marker = 'o', markersize = 3)

ax.plot(recovered_by_date, **marker_style)

plt.ylabel('Recovered', fontsize = 12)

plt.xlabel('Date', fontsize = 12)

plt.title('Covid-19 Recovered in the United States', fontsize = 14)



#formatting 

ax.xaxis.set_major_locator(months)

ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

ax.format_ydata = lambda x: '$%1.2f' % x  

ax.grid(True)



plt.show()
#create cleaned dataframe for plot.

recovered.index = pd.to_datetime(recovered.index)

plot_df = pd.DataFrame(cases_by_date)





labels = ['deaths', 'recovered']

dfs = [deaths_by_date, recovered]

i=0

for label in labels:

    plot_df[label] = pd.DataFrame(dfs[i])

    i+=1

plot_df
#plot.

fig, ax = plt.subplots(1, figsize = (16,8))



plt.plot(plot_df['cases'], label = 'cases', color = 'blue', linewidth = 2.5, marker = 'o', markersize = 2)

plt.plot(plot_df['recovered'], label = 'recovered', color = 'green', linewidth = 2.5, marker = 'o', markersize = 2)

plt.plot(plot_df['deaths'], label = 'deaths', color = 'red', linewidth = 2.5, marker = 'o', markersize = 2, alpha = 0.8)





#labels

plt.xlabel('Date', fontsize = 14)

plt.ylabel('Cases', fontsize = 14)

plt.title('Covid-19 United States Cases - Confirmed, Deaths, Recovered', fontsize = 20)

plt.legend()



#formatting

ax.xaxis.set_major_locator(months)

ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

ax.format_ydata = lambda x: '$%1.2f' % x  

ax.grid(True)

ax.patch.set_facecolor('grey') #chance background color if needed. 

ax.patch.set_alpha(0.2)


#states affected over time. 

states_affected = []

cases_date = cases[['cases', 'date','state']]

for i, row in cases_date.iterrows():

    if row['cases'] != 0:

        states_affected.append(row)

states_affected = pd.DataFrame(states_affected)
#first 10 States to contract covid-19. 

sorted_dates = states_affected.sort_values('date', ascending = True)

sorted_uniq_dates = sorted_dates.drop_duplicates('state')

sorted_uniq_dates.head(10)
#spread of Covid-19 to New States.

state_counts = []

for i in range(0, len(sorted_uniq_dates)):

    state_counts.append(i)



sorted_uniq_dates['count'] = state_counts



plt.figure(figsize = (16,8))

plt.scatter(x = sorted_uniq_dates['date'], y = sorted_uniq_dates['count'])

plt.plot(sorted_uniq_dates['date'], sorted_uniq_dates['count'], 'o--')

plt.xticks(rotation=90)

plt.title('Covid-19 Spread to New States', fontsize = 14)

plt.ylabel('States Afflicted', fontsize = 12)

plt.grid(True)
cases_by_date.index = pd.to_datetime(cases_by_date.index)



#covid-19 daily new confirmed cases.

difference = cases_by_date.diff()

difference = difference.fillna(0)

fig, ax = plt.subplots(figsize = (16,8))



marker_style = dict(linewidth=2.5, linestyle = '-', marker = 'o', markersize = 5)

ax.plot(difference, **marker_style)



#labels

plt.xlabel('Date', fontsize = 12)

plt.ylabel('Cases', fontsize = 12)

plt.title('Covid-19 Daily New Confirmed Cases')



#formatting

ax.xaxis.set_major_locator(months)

ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

ax.format_ydata = lambda x: '$%1.2f' % x  

ax.grid(True)
difference = deaths_by_date.diff()

difference = difference.fillna(0)

fig, ax = plt.subplots(figsize = (16,8))





marker_style = dict(linewidth=2.5, linestyle = '-', marker = 'o', markersize = 5)

ax.plot(difference, **marker_style)



#labels

plt.xlabel('Date', fontsize = 12)

plt.ylabel('Cases', fontsize = 12)

plt.title('Covid-19 Daily New Death Reports')



#formatting

ax.xaxis.set_major_locator(months)

ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

ax.format_ydata = lambda x: '$%1.2f' % x  

ax.grid(True)
#prepare data.

data = data.merge(pop_data,

          how = 'left',

          left_on = 'state',

          right_on = 'state')
democratic = ['Washington', 'Oregon', 'Nevada', 'California', 'Colorado', 'New Mexico', 'Illinois', 'Minnesota', 'Virginia',

             'Maine', 'New York', 'New Hampshire', 'Vermont', 'Massachusetts', 'Rhode Island', 'Connecticut', 'New Jersey',

             'Deleware', 'Maryland', 'Washington D.C.', 'Hawaii']



republican = ['Idaho', 'Montana', 'Utah', 'Arizona', 'Wyoming' ,'Texas', 'North Dakota', 'South Dakota', 'Nebraska',

             'Kansas', 'Oklahoma', 'Iowa', 'Missouri', 'Arkansas', 'Lousiana', 'Wisconsin' ,'Michigan', 'Indiana', 'Kentucky',

             'Tennessee', 'Alabama', 'Georgia', 'Florida', 'South Carolina', 'North Carolina', 'Ohio', 'West Virgina',

             'Pennsylvania', 'Alaska']



#initialize empty column.

data['political status'] = np.nan



#create dummy variables.

for i, state in enumerate(data['state']):

    if state in democratic:

        data.at[i,'political status'] = 0

    else:

        data.at[i, 'political status'] = 1
#percents

democratic_cases = data[data['political status'] == 0]['cases'].sum() 

republican_cases = data[data['political status'] == 1]['cases'].sum()



democratic_pop = data[data['political status'] == 0]['population'].sum()

republican_pop = data[data['political status'] == 1]['population'].sum()



democratic_case_percent = democratic_cases / democratic_pop * 100

republican_case_percent = republican_cases / republican_pop * 100



political_affiliation = pd.DataFrame({'Democratic': [democratic_case_percent],

                                     'Republican': [republican_case_percent]})

political_affiliation
#coronavirus testing. 

data_owid = pd.read_csv(r'/kaggle/input/covid19-visualization-and-analysis/owid-covid-data.csv')

united_states = data_owid['location'] == 'United States'

us_df = data_owid[united_states]

testing = us_df[['date', 'new_tests', 'total_tests', 'total_tests_per_thousand', 'new_tests_per_thousand', 'positive_rate', 'tests_per_case']]

testing = testing.fillna(0)
testing['date'] = pd.to_datetime(testing['date'])



fig, ax = plt.subplots(figsize = (16,8))





marker_style = dict(linewidth=2.5, linestyle = '-', marker = 'o', markersize = 5)

ax.plot(testing['date'], testing['total_tests'], **marker_style)



#Labels

plt.xlabel('Date', fontsize = 12)

plt.ylabel('Tests', fontsize = 12)

plt.title('Covid-19 Total Tests')



#Formatting

ax.xaxis.set_major_locator(months)

ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

ax.format_ydata = lambda x: '$%1.2f' % x  

ax.grid(True)
fig, ax = plt.subplots(figsize = (16,8))





marker_style = dict(linewidth=2.5, linestyle = '-', marker = 'o', markersize = 5)

ax.plot(testing['date'], testing['new_tests'], **marker_style)



#labels

plt.xlabel('Date', fontsize = 12)

plt.ylabel('Tests', fontsize = 12)

plt.title('Covid-19 Daily Tests')



#formatting

ax.xaxis.set_major_locator(months)

ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

ax.format_ydata = lambda x: '$%1.2f' % x  

ax.grid(True)
fig, ax = plt.subplots(figsize = (16,8))





marker_style = dict(linewidth=2.5, linestyle = '-', marker = 'o', markersize = 5)

ax.plot(testing['date'], testing['positive_rate'], **marker_style)



#labels

plt.xlabel('Date', fontsize = 12)

plt.ylabel('Positivity Rate', fontsize = 12)

plt.title('Covid-19 Positivity Rate')



#formatting

ax.xaxis.set_major_locator(months)

ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

ax.format_ydata = lambda x: '$%1.2f' % x  

ax.grid(True)
fig, ax = plt.subplots(figsize = (16,8))





marker_style = dict(linewidth=2.5, linestyle = '-', marker = 'o', markersize = 5)

ax.plot(testing['date'], testing['tests_per_case'], **marker_style)



#labels

plt.xlabel('Date', fontsize = 12)

plt.ylabel('Tests', fontsize = 12)

plt.title('Covid-19 Tests Per Case')



#formatting

ax.xaxis.set_major_locator(months)

ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

ax.format_ydata = lambda x: '$%1.2f' % x  

ax.grid(True)
fig, ax = plt.subplots(figsize = (16,8))





marker_style = dict(linewidth=2.5, linestyle = '-', marker = 'o', markersize = 5)

ax.plot(testing['date'], testing['total_tests_per_thousand'], **marker_style)



#labels

plt.xlabel('Date', fontsize = 12)

plt.ylabel('Tests (Per Thousand)', fontsize = 12)

plt.title('Covid-19 Total Tests (Per Thousand)')



#formatting

ax.xaxis.set_major_locator(months)

ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

ax.format_ydata = lambda x: '$%1.2f' % x  

ax.grid(True)
fig, ax = plt.subplots(figsize = (16,8))





marker_style = dict(linewidth=2.5, linestyle = '-', marker = 'o', markersize = 5)

ax.plot(testing['date'], testing['new_tests_per_thousand'], **marker_style)



#labels

plt.xlabel('Date', fontsize = 12)

plt.ylabel('Tests (Per Thousand)', fontsize = 12)

plt.title('Covid-19 New Tests (Per Thousand)')



#formatting

ax.xaxis.set_major_locator(months)

ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

ax.format_ydata = lambda x: '$%1.2f' % x  

ax.grid(True)
confirmed = pd.DataFrame(cases_by_date)



confirmed.tail()
#clean dataframe for usability with prophet.

confirmed.reset_index(level=0, inplace=True)

confirmed.columns = ['ds', 'y']
#setting up the model to predict 10 days ahead.

model = Prophet(interval_width = 0.95)

model.fit(confirmed)

future = model.make_future_dataframe(periods=10)
#predicting future forecast with date.

forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
pred_cases = forecast[['ds', 'yhat']]

#pred_cases['ds'] = pd.to_datetime(pred_cases['ds'])

pred_cases.set_index('ds', inplace=True)
#Prediction plot (cases) + 10 days. 



fig, ax = plt.subplots(figsize = (16,8))

marker_style = dict(linewidth=2.5, marker = 'o', markersize = 3)

ax.plot(cases_by_date, **marker_style, linestyle = '-', label = 'Actual')

ax.plot(pred_cases, **marker_style, linestyle = '-', label = 'Predicted', color = 'r', alpha =0.2)

plt.ylabel("Cases", fontsize = 12)

plt.xlabel('Date', fontsize = 12)

plt.title('Covid-19 Cases - Predicted vs. Actual', fontsize = 14)

ax.legend(loc='upper left')



#ax.patch.set_facecolor('grey')

#ax.patch.set_alpha(0.2)



#Format ticks

ax.xaxis.set_major_locator(months)

ax.grid(True)



plt.show()
conf_comp_plot = model.plot_components(forecast)
deaths = pd.DataFrame(deaths_by_date)

deaths.reset_index(level=0, inplace=True)

deaths.columns = ['ds', 'y']
model = Prophet(interval_width = 0.95)

model.fit(deaths)

future = model.make_future_dataframe(periods=10)
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
pred_deaths = forecast[['ds', 'yhat']]

#pred_cases['ds'] = pd.to_datetime(pred_cases['ds'])

pred_deaths.set_index('ds', inplace=True)
#Prediction plot (deaths) + 10 days. 



fig, ax = plt.subplots(figsize = (16,8))

marker_style = dict(linewidth=2.5, marker = 'o', markersize = 3)

ax.plot(deaths_by_date, **marker_style, linestyle = '-', label = 'Actual')

ax.plot(pred_deaths, **marker_style, linestyle = '-', label = 'Predicted', color = 'r', alpha =0.2)

plt.ylabel("Deaths", fontsize = 12)

plt.xlabel('Date', fontsize = 12)

plt.title('Covid-19 Deaths - Predicted vs. Actual', fontsize = 14)

ax.legend(loc='upper left')



#ax.patch.set_facecolor('grey')

#ax.patch.set_alpha(0.2)



#Format ticks

ax.xaxis.set_major_locator(months)

ax.grid(True)



plt.show()
deaths_comp_plot = model.plot_components(forecast)
recovered = pd.DataFrame(recovered)

recovered.reset_index(level=0, inplace=True)

recovered.columns = ['ds', 'y']
model = Prophet(interval_width = 0.95)

model.fit(recovered)

future = model.make_future_dataframe(periods=10)
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
pred_recovered = forecast[['ds', 'yhat']]

#pred_cases['ds'] = pd.to_datetime(pred_cases['ds'])

pred_recovered.set_index('ds', inplace=True)
#Prediction plot (deaths) + 10 days. 



fig, ax = plt.subplots(figsize = (16,8))

marker_style = dict(linewidth=2.5, marker = 'o', markersize = 3)

ax.plot(recovered_by_date, **marker_style, linestyle = '-', label = 'Actual')

ax.plot(pred_recovered, **marker_style, linestyle = '-', label = 'Predicted', color = 'r', alpha =0.2)

plt.ylabel("Recovered", fontsize = 12)

plt.xlabel('Date', fontsize = 12)

plt.title('Covid-19 Recovered - Predicted vs. Actual', fontsize = 14)

ax.legend(loc='upper left')



#ax.patch.set_facecolor('grey')

#ax.patch.set_alpha(0.2)



#Format ticks

ax.xaxis.set_major_locator(months)

ax.grid(True)



plt.show()
recovered_comp_plot = model.plot_components(forecast)