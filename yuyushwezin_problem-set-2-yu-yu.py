# math operations
from numpy import inf

# time operations
from datetime import timedelta

# for numerical analyiss
import numpy as np

# to store and process data in dataframe
import pandas as pd

# basic visualization package
import matplotlib.pyplot as plt

# for advanced visualization
import seaborn as sns; sns.set()

# advanced ploting
import seaborn as sns

# interactive visualization
import plotly.express as px
import plotly.graph_objs as go
# import plotly.figure_factory as ff
#from plotly.subplots import make_subplots

# for offline ploting
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)

# hide warnings
import warnings
warnings.filterwarnings('ignore')

# to interface with operating system
import os

# for trendlines
import statsmodels

# data manipulation
from datetime import datetime as dt
from scipy.stats.mstats import winsorize
cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801'
!ls ../input/corona-virus-report

!ls ../input/covid19-worldometer-snapshots-since-april-18
files = []

for dirname, _, filenames in os.walk('../input/econ-zip'):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))
        
files = sorted(files)
files



series = [pd.read_csv(f, na_values=['.']) for f in files]
series_name = ['btc', 'cpi', 'gold', 'snp', 'high_yield_bond', 'inv_grade_bond', 'moderna', 'employment', 'tesla_robinhood', 
               'trea_20y_bond', 'trea_10y_yield', 'tesla_stock', 'fed_bs', 'wti']
series_dict = dict(zip(series_name, series))
country_wise = pd.read_csv('../input/corona-virus-report/country_wise_latest.csv')
country_wise = country_wise.replace('', np.nan).fillna(0)

day_wise = pd.read_csv('../input/corona-virus-report/day_wise.csv')
day_wise['Date'] = pd.to_datetime(day_wise['Date'])


full_grouped = pd.read_csv('../input/corona-virus-report/full_grouped.csv')
full_grouped['Date'] = pd.to_datetime(full_grouped['Date'])

selected = full_grouped ['Country/Region'].str.contains('Singapore')

Singapore = full_grouped[selected]
Singapore.tail(10)

temp = Singapore.groupby('Date')['Confirmed'].sum().reset_index()
temp = temp.melt(id_vars="Date", value_vars=['Confirmed'],
                 var_name='Case', value_name='Count')
import plotly.express as px
fig = px.area(temp, x="Date", y="Count", color='Case', height=550, width=700,
             title='Cases over time', color_discrete_sequence = [rec, dth, act])
fig.show()


full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')
selected = full_table ['Country/Region'].str.contains('Singapore')
Singapore= full_table[selected]
Singapore.tail(1)

worldometer_data = pd.read_csv('../input/corona-virus-report/worldometer_data.csv')

worldometer_data = worldometer_data.replace('', np.nan).fillna(0)
worldometer_data['Case Positivity'] = round(worldometer_data['TotalCases']/worldometer_data['TotalTests'],2)
worldometer_data['Case Fatality'] = round(worldometer_data['TotalDeaths']/worldometer_data['TotalCases'],2)
worldometer_data[worldometer_data["Case Positivity"] == inf] = 0
worldometer_data ['Case Positivity Bin']= pd.qcut(worldometer_data['Case Positivity'], q=3, labels=["low", "medium", "high"])


worldometer_pop_struc = pd.read_csv('../input/covid19-worldometer-snapshots-since-april-18/population_structure_by_age_per_contry.csv')
worldometer_pop_struc = worldometer_pop_struc.fillna(0)
worldometer_data = worldometer_data.merge(worldometer_pop_struc,how='inner',left_on='Country/Region', right_on='Country')

worldometer_data = worldometer_data[worldometer_data["Country/Region"] != 0]

selected = worldometer_data ['Country/Region'].str.contains('Singapore')


Singapore = worldometer_data[selected]
Singapore.tail()



def gt_n(n):
    
    countries = full_grouped[full_grouped['Confirmed']<n]['Country/Region'].unique()
    # Filter countries that are in the unique set of countries with confirmed cases greater than N
    temp = full_grouped[full_grouped['Country/Region'].isin(countries)]
    temp = temp.groupby(['Country/Region', 'Date'])['Confirmed'].sum().reset_index()
       
    # Filter observations with confirmed cases more than N
    temp = temp[temp['Confirmed']<n]
    # print(temp.head())

    
    
    # Identify the start date when confirmed cases exceed N for each country
    min_date = temp.groupby('Country/Region')['Date'].min().reset_index()
    
    # Name the columns in the dataframe min_date
    min_date.columns = ['Country/Region', 'Min Date']
    # print(min_date.head())

    # Merge dataframe temp with dataframe min_date by 'Country/Region'
    from_nth_case = pd.merge(temp, min_date, how='inner',on='Country/Region')
    
    # Convert data type to datetime object
    from_nth_case['Date'] = pd.to_datetime(from_nth_case['Date'])
    from_nth_case['Min Date'] = pd.to_datetime(from_nth_case['Min Date'])
    
    # Create a variable that counts the number of days relative to the day when confirmed cases exceed N
    from_nth_case['N days'] = (from_nth_case['Date'] - from_nth_case['Min Date']).dt.days
    # print(from_nth_case.head())

    # Plot a line graph from dataframe from_nth_case with column 'N days' and 'Confirmed' mapped to x-axis and y-axis, respectively.
    # Distinguish each country by color (system-determined color)
    # str converts n integer into string and "'N days from '+ str(n) +' case'" is the title 
    fig = px.line(from_nth_case, x='N days', y='Confirmed', color='Country/Region', 
                  title='N days from '+ str(n) +' case', height=600)
    fig.show()
gt_n(50000)
!ls ../input/singapore-airline

import pandas as pd
singapore_airline= pd.read_csv('../input/singapore-airline/C6L.SI new.csv')
singapore_airline['Date'] = pd.to_datetime(singapore_airline['Date'])
singapore_airline['singapore_airline_return'] = singapore_airline['Adj Close'].pct_change()
singapore_airline.set_index('Date', inplace=True)

singapore_airline.info()

baseline = pd.merge(singapore_airline,Singapore, how='left', on='Date')
baseline.info()

baseline2020 = baseline[baseline['Date'] >= '2020-01-01']
baseline2020.info()
baseline2020['New cases'] = baseline2020['New cases'].fillna(0)

sns.jointplot(x = 'New cases', y = 'singapore_airline_return', data = baseline2020, kind='reg')
def plot_chart(series):
    fig = px.scatter(baseline[baseline[series].notnull()], x="Date", y=series, color='recession', width=1000)
    fig.update_traces(mode='markers', marker_size=4)
    fig.update_layout(title=series, xaxis_title="", yaxis_title="")
    fig.show()
baseline2020['singapore_airline_return'].describe()
baseline['singapore_airline_return'].describe()
print("singapore_airline historical daily returns from " + str(baseline[baseline['singapore_airline_return'].notnull()]['Date'].min().date()) + ' to '
       + str(baseline[baseline['singapore_airline_return'].notnull()]['Date'].max().date()))
fig = px.histogram(baseline, x="singapore_airline_return")
fig.show()
