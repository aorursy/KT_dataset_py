# Importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Importing the dataset

df = pd.read_csv(r'../input/all_energy_statistics.csv')



# Lists of countries and years covered.

labels = list(range(1990,2015))

countries = ['United States', 'United Kingdom', 'Germany', 'Japan', 'India', 'China',

             'Italy', 'France', 'Russia']

# Display first few rows of the dataframe

df.head()
# Display energy types in this dataset.

transaction_types = df.groupby('commodity_transaction').size()

# Display only the transaction types with large amounts of data

transaction_types = transaction_types[transaction_types > 2000]

# Display head of transaction_types

transaction_types.head()

#transaction_types.tail()
# Short description of the dataframe

df.describe()
# Crude oil energy supply

crude_oil_supply = df[df['commodity_transaction'].isin(['Conventional crude oil - total energy supply'])]

crude_oil_supply = crude_oil_supply[crude_oil_supply['country_or_area'].isin(countries)]

ax = sns.pointplot(x='year', y='quantity', hue='country_or_area', data=crude_oil_supply, palette='Set2')

ax.set_xticklabels(labels=labels, rotation=45)
# The previous plot was a bit small, so I've increased the size in this one

# plot

sns.set_style('ticks')

fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(11.7, 8.27)

sns.pointplot(x='year', y='quantity', hue='country_or_area', data=crude_oil_supply, ax=ax, palette='Set2')

ax.set_xticklabels(labels=labels, rotation=45)

sns.despine()
# Now we can fit some linear regression lines on China and USA and predict

# When the USA's energy supply will be overtaken

# Import libraries

from sklearn.linear_model import LinearRegression



# USA's data

US_oil_supply = crude_oil_supply[crude_oil_supply['country_or_area'].isin(['United States'])]

X_US = US_oil_supply['year']

X_US = X_US.reshape(-1, 1)

y_US = US_oil_supply['quantity']

y_US = y_US.reshape(-1, 1)

# USA linear model

us_regressor = LinearRegression()

us_regressor.fit(X_US, y_US)



# China's data

CN_oil_supply = crude_oil_supply[crude_oil_supply['country_or_area'].isin(['China'])]

X_CN = CN_oil_supply['year']

X_CN = X_CN.reshape(-1, 1)

y_CN = CN_oil_supply['quantity']

y_CN = y_CN.reshape(-1, 1)

# China linear model

cn_regressor = LinearRegression()

cn_regressor.fit(X_CN, y_CN)

# Forecasting

forecast_years = list(range(1990, 2040))

forecast_years = np.array(forecast_years)

forecast_years = forecast_years.reshape(-1, 1)



# Plotting

plt.scatter(US_oil_supply['year'], US_oil_supply['quantity'])

plt.plot(forecast_years, us_regressor.predict(forecast_years))

plt.scatter(CN_oil_supply['year'], CN_oil_supply['quantity'], color='r')

plt.plot(forecast_years, cn_regressor.predict(forecast_years), color='r')

plt.xlabel('Year')

plt.ylabel('Quantity')

plt.title('Crude oil supply in USA and China')

# The above model doesn't fit what would likely be the trajectory for China, I think it would be more

# Logistic, we can try that next.