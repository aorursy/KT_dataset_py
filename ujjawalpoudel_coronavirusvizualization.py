# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import date

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Reading the dataset

data = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

deathData = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_deaths.csv")

confirmedData = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_confirmed.csv")

recoveredData = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_recovered.csv")

data.head()
# Change String type column into timestamp in data dataframe

print ("Here we have Date Column as ",type(data['Date'][0]),", that's why we change it into timestamp")

data['Date'] = pd.to_datetime(data['Date'])

data.head()
# No need of Sno colum, so delete that particular column

data.drop('Sno',axis=1,inplace=True)

data.head()
# Countries affected

print(data['Country'].unique())
# Mainland China and China are same country that's why change Mainland China into China

data.replace(to_replace='Mainland China',value='China', inplace=True)

print("\nTotal countries affected by virus: ",len(data['Country'].unique()))
# To find out current status of worldwide, we have to know latest updated data



# First we have to find out last date, which is find out by data['Date'][-1]

# And change that timestamp into string for further process

d = data['Date'][-1:].astype('str')



# Now extract year,month and day using split function and convert each resulted data into integer

year = int(d.values[0].split('-')[0])

month = int(d.values[0].split('-')[1])

day = int(d.values[0].split('-')[2].split()[0])



# Generate latest datetime(i.e. Last Date) in pandas series type

latestDate = pd.Timestamp(date(year,month,day))



# Get only data of latest update

data_latest = data[data['Date'] > latestDate]

data_latest.head()
print("Top 5 Countries were affected most:\n")



topFive = data_latest.groupby(['Country']).sum().nlargest(5,['Confirmed'])  

print (topFive)
# Top five countries affected from Corona Virus

plt.figure(figsize=(12,6))

plt.title("Top most 5 countries were affected by Coronavirus")

sns.barplot(x=topFive.index,y='Confirmed',data=topFive)

plt.show()
# Top five state that are affected to corona virus

plt.figure(figsize=(15,6))

plt.title('Province/State which reported more than 1000 Confirmed case')

sns.barplot(x='Province/State', y='Confirmed', data =data_latest[data_latest['Confirmed']>1000])