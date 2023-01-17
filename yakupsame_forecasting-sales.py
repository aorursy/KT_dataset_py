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
train = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')
store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
train.info()
train.describe()
train.hist(bins=30,figsize = (20,20))
train['Open'].value_counts()
train = train[train['Open']==1]
train.drop(['Open'],axis=1,inplace=True)
train.describe()
store.isnull().sum()
sns.heatmap(store.isnull(),yticklabels = False, cmap = 'Blues',cbar = False)
store.head(10)
# When Promo2 column is 0 then related fields are NaN so we can change them to 0 as well.
store[store['CompetitionDistance'].isnull()]
store[store['CompetitionOpenSinceMonth'].isnull()]
store[store['CompetitionOpenSinceYear'].isnull()]
null_columns = ['Promo2SinceWeek','Promo2SinceYear','PromoInterval','CompetitionOpenSinceMonth','CompetitionOpenSinceYear']
for i in null_columns:
    store[i].fillna(0,inplace=True)
store.isnull().sum()
store['CompetitionDistance'].fillna(store['CompetitionDistance'].mean(),inplace=True)
sns.heatmap(store.isnull(),yticklabels = False, cmap = 'Blues',cbar = False)
store.hist(bins = 30,figsize=(20,20),color='red')
df = pd.merge(store,train,how = 'inner',on = 'Store')
df
correlation = df.corr()['Sales'].sort_values()
correlation
correlations = df.corr()
f,ax =plt.subplots(figsize = (20,20))
sns.heatmap(correlations,annot=True)
df.head()
df['Year'] = pd.DatetimeIndex(df['Date']).year
df['Month'] = pd.DatetimeIndex(df['Date']).month
df['Day'] = pd.DatetimeIndex(df['Date']).day
df.drop(['Date'],axis=1,inplace=True)
df.head()
axis = df.groupby('Month')[['Sales']].mean().plot(figsize = (12,5),color = 'r' , marker = 'o')
axis.set_title('Average sales per month')
plt.figure()
axis = df.groupby('Month')[['Customers']].mean().plot(figsize = (12,5),color = 'b' , marker = 'o')
axis.set_title('Average customers per month')
plt.figure()
## Minimum number of sales are on 24th of a month and generally maximum is at beginning and at the end of the month.
axis = df.groupby('Day')[['Sales']].mean().plot(figsize = (15,5),color = 'g' , marker = 'o')
axis.set_title('Average sales per day')
plt.figure()
# Minimum number of customers are on 24th of a month and generally maximum is at beginning and at the end of the month.
axis = df.groupby('Day')[['Customers']].mean().plot(figsize = (15,5),color = 'm' , marker = 'o')
axis.set_title('Average customers per day')
plt.figure()
df.head()
# Most sales are on Sunday and Monday
#Minimum sales is on Saturday
axis = df.groupby('DayOfWeek')[['Sales']].mean().plot(figsize = (10,5),color = 'y' , marker = 'o')
axis.set_title('Average sales per day of a week')
plt.figure()
# Customers visit most on Sunday
# Customers visit stores less on Saturday
axis = df.groupby('DayOfWeek')[['Customers']].mean().plot(figsize = (10,5),color = 'k' , marker = 'o')
axis.set_title('Average customers per day of a week')
plt.figure()
plt.figure(figsize = (10,10))

plt.subplot(211)
sns.barplot(x = 'Promo' , y = 'Sales' , data = df)
plt.subplot(212)
sns.barplot(x = 'Promo' , y = 'Customers' , data = df)
#On average, customers visit and sales are higher during promo. 
plt.figure(figsize = (15,15))

plt.subplot(211)
sns.boxplot(x = 'Promo' , y = 'Sales' ,hue = 'StoreType' ,data = df)
plt.subplot(212)
sns.boxplot(x = 'Promo' , y = 'Customers' ,hue = 'StoreType' ,data = df)