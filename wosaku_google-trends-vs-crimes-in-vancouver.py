# Import data manipulation packages

import numpy as np

import pandas as pd



# Import data visualization packages

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Importing the data

googletrend = pd.read_csv('../input/googletrend.csv', index_col='Month')
# Taking a look at the first entries

googletrend.head()
# Checking index and data types

googletrend.info()
# Importing CSV file

crimes = pd.read_csv('../input/crime.csv')

crimes.head()
# Creating a date column from the date parts

crimes['DATE'] = pd.to_datetime({'year':crimes['YEAR'], 'month':crimes['MONTH'], 'day':crimes['DAY']})



# Change the index to the colum 'DATE'

crimes.index = pd.DatetimeIndex(crimes['DATE'])
# The crime data starts from 2003, but our Google data starts from 2004 and ends in 2017-06. 

# Let's remove 2003 from our crime data and 2017-07.

crimes = crimes[(crimes['DATE'] > '2003-12-31') & (crimes['DATE'] < '2017-07-01') ]



# The crime data lists all individual crimes. 

# We need to group it by month to compare it to the Google trend.

crimes_month = pd.DataFrame(crimes.resample('M').size()) 
crimes_month.info()
# Just renaming the column...

crimes_month.columns = ['Total']



# Taking a look at the data

crimes_month.head()
# Dividing the total number of crimes by the maximum value and round them

crimes_month['Crime Index'] = (crimes_month['Total']/crimes_month['Total']

                               .max()*100).astype(int)
crime_trend = pd.concat([crimes_month['Crime Index'],googletrend], axis =1)

crime_trend.head()
crime_trend.plot(figsize=(12,6), linewidth=3)

plt.title('Crime Index and Google Trends', fontsize=16)

plt.tick_params(labelsize=14)

plt.legend(prop={'size':14});
# Now let's use a 6 months window

crime_trend_rolling6 = crime_trend.rolling(window=6).mean().dropna()
# Plot

crime_trend_rolling6.plot(figsize=(8,4), linewidth=3)

plt.title('Crime Index and Google Trends - Moving Average', fontsize=16)

plt.tick_params(labelsize=14)

plt.legend(prop={'size':14});
# Using .shift(-5) to lag the search index

crime_trend_rolling6_shifted = (pd.concat([crime_trend_rolling6['Crime Index'],

                                             crime_trend_rolling6['Search Index']

                                             .shift(-5)], axis=1))



crime_trend_rolling6_shifted.columns = ['Crime Index','Search Index (shifted)']



# Let's focus on 2010 on

crime_trend_rolling6_shifted = crime_trend_rolling6_shifted[crime_trend_rolling6_shifted.index >=

                                                            '2010-01-01']
# Plot

crime_trend_rolling6_shifted.plot(figsize=(8,4), linewidth=3)

plt.title('Crime Index and Google Trends (Shifted) - Moving Average', fontsize=16)

plt.tick_params(labelsize=14)

plt.legend(prop={'size':14});
# Let's check the corrleation

crime_trend_rolling6_shifted.corr()