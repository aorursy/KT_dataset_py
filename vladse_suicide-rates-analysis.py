# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
# first look

data.head()
data.info()
data.isnull().sum()
# we'll drop 'HDI for year'(too many missing values) and 'country-year' column (redundant info)
data.drop(['HDI for year','country-year'],axis =1, inplace=True)
data.isnull().sum()
data.describe()
plt.figure(figsize=(20,10))
sns.barplot(x='country',y='suicides_no',data=data.groupby('country')['suicides_no'].sum().sort_values(ascending=False).head().reset_index())
plt.ylabel('Total suicides')
plt.xlabel('Country')
plt.title('Countries with the most suicides from 1985 to 2016')
sns.despine()
top_5 = data.groupby('country')['suicides_no'].sum().sort_values(ascending=False).head().reset_index()['country'].values.tolist()
plt.figure(figsize=(20,10))
sns.barplot(x='sex',y='suicides_no',data=data.groupby('sex')['suicides_no'].sum().sort_values(ascending=False).head().reset_index())
plt.ylabel('Total suicides')
plt.xlabel('Genre')
plt.title('Distribution by sex from 1985 to 2016')
sns.despine()
plt.figure(figsize=(20,10))
sns.barplot(x='age',y='suicides_no',data=data.groupby('age')['suicides_no'].sum().sort_values(ascending=False).head().reset_index())
plt.ylabel('Total suicides')
plt.xlabel('Age group')
plt.title('Distribution by age group from 1986 to 2016')
sns.despine()
plt.figure(figsize=(20,10))
sns.lineplot(x='year',y='suicides_no',data=data.groupby('year')['suicides_no'].sum().sort_values(ascending=False).reset_index())
plt.ylabel('Total suicides')
plt.xlabel('Year')
plt.title('Evolution of total suicides by year')
sns.despine()

plt.figure(figsize=(20,10))
sns.lineplot(x='year',y='suicides_no',hue='country', data=data[data['country'].isin(top_5)].groupby(['country','year'])['suicides_no'].sum().reset_index())
sns.despine()
plt.title('Evolution of the sucide rates by year for the 5 countries with most suicides');

d = pd.merge(data.groupby(['country','year'])['gdp_per_capita ($)'].mean(),data.groupby(['country','year'])['suicides_no'].sum(), on=['country','year']).reset_index()

d.head()
d[['gdp_per_capita ($)','suicides_no']].corr()
#cannot use sns.lmplot here, there seem to be too many variables in hue for the regression to work

plt.figure(figsize=(20,10))
sns.scatterplot(x='gdp_per_capita ($)',y='suicides_no',hue='country', data=d)
plt.legend('')
sns.despine()

plt.figure(figsize=(20,10))
sns.scatterplot(x='gdp_per_capita ($)',y='suicides_no',hue='country', data=d[d['gdp_per_capita ($)']>65000],s=50)
plt.ylim(0,60000)
sns.despine()

d[d['gdp_per_capita ($)']>65000][['gdp_per_capita ($)','suicides_no']].corr()

d_100k = pd.merge(data.groupby(['country','year'])['gdp_per_capita ($)'].mean(),data.groupby(['country','year'])['suicides/100k pop'].sum(), on=['country','year']).reset_index()

plt.figure(figsize=(20,10))
sns.scatterplot(x='gdp_per_capita ($)',y='suicides/100k pop',hue='country', data=d_100k)
plt.legend('')
sns.despine()
d_100k[['gdp_per_capita ($)','suicides/100k pop']].corr()
happy = pd.read_csv('../input/share-of-people-who-say-they-are-happy/share-of-people-who-say-they-are-happy.csv')
happy.head()
happy.columns
# data cleaning (dropping the 'code' column) and creating a new data-frame for us to analyse
happy.drop('Code',axis=1, inplace=True)
happy = happy.rename(columns={'Entity':'country','Year':'year',' (%)':'happy'})
happy.loc[happy['country']=='Russia','country'] = "Russian Federation"
new_d = pd.merge(d,happy, on=['country','year'])
new_d[['suicides_no','happy']].corr()
most_suicides = data.groupby('country')['suicides_no'].sum().sort_values(ascending=False).head().reset_index().country.tolist()
plt.figure(figsize=(20,10))
sns.scatterplot(x='happy',y='suicides_no',hue='country', data=new_d[new_d['country'].isin(most_suicides)],s=50)
plt.legend('')
sns.despine()

new_d[['gdp_per_capita ($)','happy']].corr()
#it appeas so, more or less... let's plot it

plt.figure(figsize=(20,10))
sns.scatterplot(x='happy',y='gdp_per_capita ($)',hue='country', data=new_d)
plt.legend('')
sns.despine()

