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
# Import appropriate libraries

import pandas as pd

import numpy as np

from pandas import Series, DataFrame

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')

%matplotlib inline
# Read dataset 

#df = pd.read_csv('train.csv',parse_dates=['Date']) # From local

df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv') # from Kaggle

df.head()
# Check Data types

df.dtypes
# Check if any duplicates

duplicate_rows_df = df[df.duplicated()]

print("number of duplicate rows: ", duplicate_rows_df.shape)
df.count
# If any null values (as per dataset only Province/State shows null)

df.isnull().sum()
# Rename both the columns (Province/State to State & Country/Region to Country to avoid confusion and errors)

df.rename(columns = {'Province/State':'State', 'Country/Region':'Country'}, inplace = True)
# drop State as there are so many null values

df1 = df.drop(columns = {'State'})

df1.head()
# Groupby Country/ confirmesCases and Fatalities

countries = df.groupby('Country')['ConfirmedCases','Fatalities'].sum().sort_values(by='Country',ascending=False)

countries
# Plot to check ConfirmedCases and Fatalities

countries.plot(legend=True,figsize=(15,5))

plt.show()
# Graph to show top confirmed cases by top countries

top_countries = countries.index[:10]

df_top_countries = df[df['Country'].isin(top_countries)]

ax = sns.scatterplot(x="Country", y="ConfirmedCases", size=10, data=df)
df1.head()
# Plot to show Confirmed cases

df1['ConfirmedCases'].plot(legend=True,figsize=(15,5))

plt.show()
# Plot to show Fatalities

df1['Fatalities'].plot(legend=True,figsize=(10,4))

plt.show()
# Plot to check Confirmed cases by Country

df1.ConfirmedCases.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))

plt.title("Covid19 ConfirmedCases")

plt.ylabel("ConfirmedCases")

plt.xlabel("Ratio")
# Plotting a scatter plot between Confirmed Cases vs Fatalities

fig, ax = plt.subplots(figsize=(10,6))

ax.scatter(df1['ConfirmedCases'], df['Fatalities'])

ax.set_xlabel('ConfirmedCases')

ax.set_ylabel('Fatalities')

plt.show()
# Finally Pairplot to check for all Fatalities vs all the parameters

for i in range(0, len(df1.columns), 5):

    sns.pairplot(df, x_vars=df.columns[i:i+5],

                y_vars=['Fatalities'])