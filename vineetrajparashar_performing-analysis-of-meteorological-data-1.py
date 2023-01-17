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

%matplotlib inline

import matplotlib.pyplot as plt 
path = '../input/weather-dataset/weatherHistory.csv'

df=pd.read_csv(path)
df.sample(5)
df.shape
df.dtypes
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)

df['Formatted Date']
df.describe()
df = df.set_index("Formatted Date")

df.head(2)
data_columns = ['Apparent Temperature (C)', 'Humidity']

df_monthly_mean = df[data_columns].resample('MS').mean()

df_monthly_mean.head()
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

plt.figure(figsize=(14,6))

plt.title("Variation in Apparent Temperature and Humidity with time")

sns.lineplot(data=df_monthly_mean)
df1 = df_monthly_mean[df_monthly_mean.index.month==4]

print(df1)

df1.dtypes
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(18,7))

ax.plot(df1.loc['2006-04-01':'2016-04-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)')

ax.plot(df1.loc['2006-04-01':'2016-04-01', 'Humidity'], marker='o', linestyle='-',label='Humidity')

#ax.set_xticks(['04-01-2006','04-01-2007','04-01-2008','04-01-2009','04-01-2010','04-01-2011','04-01-2012','04-01-2013','04-01-2014','04-01-2015','04-01-2016'])

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %m %Y'))

ax.legend(loc ='center right')

ax.set_xlabel('Month of April')
sns.lmplot(x='Apparent Temperature (C)',y='Humidity',data=df_monthly_mean)

plt.show()
corr = df_monthly_mean.corr()

sns.heatmap(corr)
sns.distplot(df.Humidity,color='red')
sns.relplot(data=df,x="Apparent Temperature (C)",y="Humidity",color="purple",hue="Summary")