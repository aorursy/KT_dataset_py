import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt
!git clone https://github.com/CSSEGISandData/COVID-19.git

!cd COVID-19
os.listdir('COVID-19/csse_covid_19_data/csse_covid_19_daily_reports')
data_covid19 = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/05-05-2020.csv').replace('US', 'United States')

data_covid19 = data_covid19.groupby('Country_Region').sum()

data_covid19
data_energy_consumption = pd.read_html('https://en.wikipedia.org/wiki/List_of_countries_by_electricity_consumption')[0]

data_energy_consumption.set_index('Country/Region', inplace=True)

data_energy_consumption
data = data_covid19.merge(data_energy_consumption, 'left', left_index=True, right_index=True)

data
data['Electricity consumption (kW·h/yr)'] = data['Electricity consumption (kW·h/yr)'].astype('float').fillna(0)

data['Electricity consumption (kW·h/yr)']
data = data.replace([np.Inf, -np.Inf, np.NaN], 0)

data
data_corr = data[['Confirmed', 'Deaths', 'Recovered', 'Electricity consumption (kW·h/yr)', 'Average electrical energy per capita (kWh per person per year)']].corr()

data_corr['Electricity consumption (kW·h/yr)']
data_corr['Electricity consumption (kW·h/yr)'][['Confirmed', 'Deaths', 'Recovered']].plot(kind='bar', title='Correlations between energy consumption and COVID-19 cases')

plt.ylabel('Correlations')

plt.show()
data_corr['Average electrical energy per capita (kWh per person per year)'][['Confirmed', 'Deaths', 'Recovered']].plot(kind='bar', title='Correlations between energy consumption per capita and COVID-19')

plt.ylabel('Correlations')

plt.show()
from scipy.stats.stats import pearsonr

from pydoc import help

pearsonr(data['Average electrical energy per capita (kWh per person per year)'], data['Confirmed'])
help(pearsonr)