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
opsd_daily = pd.read_csv("../input/germany-electricity-power-for-20062017/opsd_germany_daily.csv", index_col=0, parse_dates=True)
opsd_daily.index
opsd_daily.head(3)
opsd_daily.tail(3)
opsd_daily.dtypes
opsd_daily['Year'] = opsd_daily.index.year

opsd_daily['Month'] = opsd_daily.index.month

opsd_daily['Weekday Name'] = opsd_daily.index.weekday_name
opsd_daily.sample(20,random_state=0)
opsd_daily.loc['2017-08-10']
opsd_daily.loc['2014-01-20':'2014-01-22']
opsd_daily.loc['2012-02']
import matplotlib.pyplot as plt
import seaborn as sns

# Use seaborn style defaults and set the default figure size

sns.set(rc={'figure.figsize':(11, 4)})
opsd_daily['Consumption'].plot(linewidth=0.5);
cols_plot = ['Consumption', 'Solar', 'Wind']

axes = opsd_daily[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)

for ax in axes:

    ax.set_ylabel('Daily Totals (GWh)')
ax = opsd_daily.loc['2017', 'Consumption'].plot()

ax.set_ylabel('Daily Consumption (GWh)');
ax = opsd_daily.loc['2017-01':'2017-02', 'Consumption'].plot(marker='o', linestyle='-')

ax.set_ylabel('Daily Consumption (GWh)');
ax = opsd_daily.loc['2012', 'Consumption'].plot()

ax.set_ylabel('Daily Consumption (GWh)');
import matplotlib.dates as mdates
fig, ax = plt.subplots()

ax.plot(opsd_daily.loc['2017-01':'2017-02', 'Consumption'], marker='o', linestyle='-')

ax.set_ylabel('Daily Consumption (GWh)')

ax.set_title('Jan-Feb 2017 Electricity Consumption')

# Set x-axis major ticks to weekly interval, on Mondays

ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))

# Format x-tick labels as 3-letter month name and day number

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'));
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

for name, ax in zip(['Consumption', 'Solar', 'Wind'], axes):

    sns.boxplot(data=opsd_daily, x='Month', y=name, ax=ax)

ax.set_ylabel('GWh')

ax.set_title(name)

# Remove the automatic x-axis label from all but the bottom subplot

if ax != axes[-1]:

    ax.set_xlabel('')
sns.boxplot(data=opsd_daily, x='Weekday Name', y='Consumption');
pd.date_range('1998-03-10', '1998-03-15', freq='D')
pd.date_range('2004-09-20', periods=8, freq='H')
opsd_daily.index
# To select an arbitrary sequence of date/time values from a pandas time series,

# we need to use a DatetimeIndex, rather than simply a list of date/time strings

times_sample = pd.to_datetime(['2013-02-03', '2013-02-06', '2013-02-08'])

# Select the specified dates and just the Consumption column

consum_sample = opsd_daily.loc[times_sample, ['Consumption']].copy()

consum_sample
consum_sample.index
# Convert the data to daily frequency, without filling any missings

consum_freq = consum_sample.asfreq('D')

# Create a column with missings forward filled

consum_freq['Consumption - Forward Fill'] = consum_sample.asfreq('D', method='ffill')

consum_freq
# Convert the data to daily frequency, without filling any missings

consum_freq = consum_sample.asfreq('D')

# Create a column with missings forward filled

consum_freq['Consumption - Forward Fill'] = consum_sample.asfreq('D', method='bfill')

consum_freq
Resampling
# Specify the data columns we want to include (i.e. exclude Year, Month, Weekday Name)

data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']

# Resample to weekly frequency, aggregating with mean

opsd_weekly_mean = opsd_daily[data_columns].resample('W').mean()

opsd_weekly_mean.head(3)
print(opsd_daily.shape[0])

print(opsd_weekly_mean.shape[0])
# Start and end of the date range to extract

start, end = '2017-01', '2017-06'

# Plot daily and weekly resampled time series together

fig, ax = plt.subplots()

ax.plot(opsd_daily.loc[start:end, 'Solar'],

marker='.', linestyle='-', linewidth=0.5, label='Daily')

ax.plot(opsd_weekly_mean.loc[start:end, 'Solar'],

marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')

ax.set_ylabel('Solar Production (GWh)')

ax.legend();
# Compute the monthly sums, setting the value to NaN for any month which has

# fewer than 28 days of data

opsd_monthly = opsd_daily[data_columns].resample('M').sum(min_count=28)

opsd_monthly.head(3)
fig, ax = plt.subplots()

ax.plot(opsd_monthly['Consumption'], color='black', label='Consumption')

opsd_monthly[['Wind', 'Solar']].plot.area(ax=ax, linewidth=0)

ax.xaxis.set_major_locator(mdates.YearLocator())

ax.legend()

ax.set_ylabel('Monthly Total (GWh)');