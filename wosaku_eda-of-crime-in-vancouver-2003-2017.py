# Import data manipulation packages

import numpy as np

import pandas as pd



# Import data visualization packages

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Import CSV file as a pandas df (Data Frame)

df = pd.read_csv('../input/crime.csv')



# Take a look at the first entries

df.head()
# Dropping column. Use axis = 1 to indicate columns and inplace = True to 'commit' the transaction

df.drop(['MINUTE'], axis = 1, inplace=True)
# Let's take a look into our data to check for missing values and data types

df.info()
# As HOUR is a float data type, I'm filling with a dummy value of '99'. For others, filling with 'N/A'

df['HOUR'].fillna(99, inplace = True)

df['NEIGHBOURHOOD'].fillna('N/A', inplace = True)

df['HUNDRED_BLOCK'].fillna('N/A', inplace = True)
# Use pandas function to_datetime to convert it to a datetime data type

df['DATE'] = pd.to_datetime({'year':df['YEAR'], 'month':df['MONTH'], 'day':df['DAY']})
# Let's use padas dt.dayofweek (Monday=0 to Sunday=6) and add it as a column called 'DAY_OF_WEEK'

df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
# Change the index to the colum 'DATE'

df.index = pd.DatetimeIndex(df['DATE'])
# Filtering the data to exclude month of July 2017

df = df[df['DATE'] < '2017-07-01']
# Using pandas value_counts function to aggregate types

df['TYPE'].value_counts().sort_index()
# Create a function to categorize types, using an 'if' statement.

def category(crime_type):

    if 'Theft' in crime_type:

        return 'Theft'

    elif 'Break' in crime_type:

        return 'Break and Enter'

    elif 'Collision' in crime_type:

        return 'Vehicle Collision'

    else:

        return 'Others'
# Apply the function and add it as CATEGORY column

df['CATEGORY'] = df['TYPE'].apply(category)
vehicle_collision = df[df['CATEGORY'] == 'Vehicle Collision']

crimes = df[df['CATEGORY'] != 'Vehicle Collision']
# Using resample('D') to group it by day and size() to return the count

plt.figure(figsize=(15,6))

plt.title('Distribution of Crimes per day', fontsize=16)

plt.tick_params(labelsize=14)

sns.distplot(crimes.resample('D').size(), bins=60);
# Using idxmax() to find out the index of the max value

crimes.resample('D').size().idxmax()
# Create a Upper Control Limit (UCL) and a Lower Control Limit (LCL) without the outlier

crimes_daily = pd.DataFrame(crimes[crimes['DATE'] != '2011-06-15'].resample('D').size())

crimes_daily['MEAN'] = crimes[crimes['DATE'] != '2011-06-15'].resample('D').size().mean()

crimes_daily['STD'] = crimes[crimes['DATE'] != '2011-06-15'].resample('D').size().std()

UCL = crimes_daily['MEAN'] + 3 * crimes_daily['STD']

LCL = crimes_daily['MEAN'] - 3 * crimes_daily['STD']



# Plot Total crimes per day, UCL, LCL, Moving-average

plt.figure(figsize=(15,6))

crimes.resample('D').size().plot(label='Crimes per day')

UCL.plot(color='red', ls='--', linewidth=1.5, label='UCL')

LCL.plot(color='red', ls='--', linewidth=1.5, label='LCL')

crimes_daily['MEAN'].plot(color='red', linewidth=2, label='Average')

plt.title('Total crimes per day', fontsize=16)

plt.xlabel('Day')

plt.ylabel('Number of crimes')

plt.tick_params(labelsize=14)

plt.legend(prop={'size':16});
# Find out how many crimes by getting the length

len(crimes['2011-06-15'])
# Check how many crimes per type

crimes['2011-06-15']['TYPE'].value_counts().head(5)
# Check how many crimes per neighbourhood

crimes['2011-06-15']['NEIGHBOURHOOD'].value_counts().head(5)
# Check how many crimes per hour

crimes['2011-06-15']['HOUR'].value_counts().head(5)
# Create a pivot table with day and month; another that counts the number of years that each day had; and the average. 

crimes_pivot_table = crimes[(crimes['DATE'] != '2011-06-15')].pivot_table(values='YEAR', index='DAY', columns='MONTH', aggfunc=len)

crimes_pivot_table_year_count = crimes[(crimes['DATE'] != '2011-06-15')].pivot_table(values='YEAR', index='DAY', columns='MONTH', aggfunc=lambda x: len(x.unique()))

crimes_average = crimes_pivot_table/crimes_pivot_table_year_count

crimes_average.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']



# Using seaborn heatmap

plt.figure(figsize=(7,9))

plt.title('Average Number of Crime per Day and Month', fontsize=14)

sns.heatmap(crimes_average.round(), cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt=".0f");
# Using resample 'M' and rolling window 12

plt.figure(figsize=(15,6))

crimes.resample('M').size().plot(label='Total per month')

crimes.resample('M').size().rolling(window=12).mean().plot(color='red', linewidth=5, label='12-months Moving Average')



plt.title('Crimes per month', fontsize=16)

plt.xlabel('')

plt.legend(prop={'size':16})

plt.tick_params(labelsize=16);
# Using pivot_table to groub by date and category, resample 'M' and rolling window 12

crimes.pivot_table(values='TYPE', index='DATE', columns='CATEGORY', aggfunc=len).resample('M').sum().rolling(window=12).mean().plot(figsize=(15,6), linewidth=4)

plt.title('Moving Average of Crimes per month by Category', fontsize=16)

plt.xlabel('')

plt.legend(prop={'size':16})

plt.tick_params(labelsize=16);
# Create a pivot table with month and category. 

crimes_pivot_table = crimes.pivot_table(values='TYPE', index='CATEGORY', columns='MONTH', aggfunc=len)



# To compare categories, I'm scaling each category by diving by the max value of each one

crimes_scaled = pd.DataFrame(crimes_pivot_table.iloc[0] / crimes_pivot_table.iloc[0].max())



# Using a for loop to scale others

for i in [2,1]:

    crimes_scaled[crimes_pivot_table.index[i]] =  pd.DataFrame(crimes_pivot_table.iloc[i] / crimes_pivot_table.iloc[i].max())

                    

# Using seaborn heatmap

plt.figure(figsize=(4,4))

plt.title('Month and Category heatmap', fontsize=14)

plt.tick_params(labelsize=12)

sns.heatmap(crimes_scaled, cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False);
# Create a pivot table with day of the week and category. 

crimes_pivot_table = crimes.pivot_table(values='TYPE', index='CATEGORY', columns='DAY_OF_WEEK', aggfunc=len)



# To compare categories, I'm scaling each category by diving by the max value of each one

crimes_scaled = pd.DataFrame(crimes_pivot_table.iloc[0] / crimes_pivot_table.iloc[0].max())



# Using a for loop to scale row

for i in [2,1]:

    crimes_scaled[crimes_pivot_table.index[i]] = crimes_pivot_table.iloc[i] / crimes_pivot_table.iloc[i].max()

                    

crimes_scaled.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']



# Using seaborn heatmap

plt.figure(figsize=(4,4))

plt.title('Day of the Week and Category heatmap', fontsize=14)

plt.tick_params(labelsize=12)

sns.heatmap(crimes_scaled, cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False);
# Create a pivot table with hour and category. 

crimes_pivot_table = crimes.pivot_table(values='TYPE', index='CATEGORY', columns='HOUR', aggfunc=len)



# To compare categories, I'm scaling each category by diving by the max value of each one

crimes_scaled = pd.DataFrame(crimes_pivot_table.iloc[0] / crimes_pivot_table.iloc[0].max())



# Using a for loop to scale row

for i in [2,1]:

    crimes_scaled[crimes_pivot_table.index[i]] =  pd.DataFrame(crimes_pivot_table.iloc[i] / crimes_pivot_table.iloc[i].max())

                    

# Using seaborn heatmap

plt.figure(figsize=(5,5))

plt.title('Hour and Category heatmap', fontsize=14)

plt.tick_params(labelsize=12)

sns.heatmap(crimes_scaled, cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False);
# Create a pivot table with hour and day of week. 

crimes_pivot_table = crimes[crimes['HOUR'] != 99].pivot_table(values='TYPE', index='DAY_OF_WEEK', columns='HOUR', aggfunc=len)



# To compare categories, I'm scaling each category by diving by the max value of each one

crimes_scaled = pd.DataFrame(crimes_pivot_table.loc[0] / crimes_pivot_table.loc[0].max())



# Using a for loop to scale each day

for i in [1,2,3,4,5,6]:

    crimes_scaled[i] = crimes_pivot_table.loc[i] / crimes_pivot_table.loc[i].max()



# Rename days of week

crimes_scaled.columns = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']



# Using seaborn heatmap

plt.figure(figsize=(6,6))

plt.title('Hour and Day of the Week heatmap', fontsize=14)

plt.tick_params(labelsize=12)

sns.heatmap(crimes_scaled, cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False);
# Let's check what types of theft we have and how many

crimes[crimes['CATEGORY'] == 'Theft']['TYPE'].value_counts()
# Initiate the figure and define size

plt.figure(1)

plt.figure(figsize=(15,8))



# Using a for loop to plot each type of crime with a moving average

i = 221

for crime_type in crimes[crimes['CATEGORY'] == 'Theft']['TYPE'].unique():    

    plt.subplot(i);

    crimes[crimes['TYPE'] == crime_type].resample('M').size().plot(label='Total per month')

    crimes[crimes['TYPE'] == crime_type].resample('M').size().rolling(window=12).mean().plot(color='red', linewidth=5, label='12-months Moving Average')

    plt.title(crime_type, fontsize=14)

    plt.xlabel('')

    plt.legend(prop={'size':12})

    plt.tick_params(labelsize=12)

    i = i + 1