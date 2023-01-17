# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/data.csv')
df.info()
columns = ['Value','ID','Age','Nationality','Overall','Potential','Wage','Preferred Foot']

data = df[columns]
data = data.dropna()
data['Preferred Foot'].unique()
def foot(x):

    if x=='Left':

        return 0

    else:

        return 1

data['Pre Foot'] = data['Preferred Foot'].apply(foot)
data['Pre Foot'].head()
def value(x):

    if 'M' in x:

        return float(x[1:-1])*1000

    elif 'K' in x:

        return float(x[1:-1])

    else:

        return 0

data['value'] = data['Value'].apply(value)
data['wage'] = data['Wage'].apply(value)
data.describe()
sns.distplot(data.value)

print(data[data.value>10000]['value'].count())
sns.pairplot(data.drop('ID',axis=1))
plt.figure(figsize=(10,8))

corr = data.drop('ID',1).corr()

sns.heatmap(corr,annot=True)
sns.countplot(data['Preferred Foot'])

print(data['Pre Foot'].sum()/data['Pre Foot'].count())
sns.violinplot(y = 'value',x='Preferred Foot',data=data[data.value<10000])

plt.figure()

sns.stripplot(x='Preferred Foot',y='value',data=data)
plt.figure(figsize=(30,8))

sns.countplot(data['Nationality'])

plt.xticks(rotation=90)

print(data['Nationality'].value_counts().sort_values(ascending=False).index[:5].values)
sns.countplot(x='Nationality',order=['England','Germany','Spain','Argentina','France'],data=data)

plt.figure()

sns.violinplot(x='Nationality',y='value',data=data[data['value']<10000],order=['England','Germany','Spain','Argentina','France'])

plt.title('value<10000')

plt.figure()

sns.violinplot(x='Nationality',y='value',data=data[data['value']<5000],order=['England','Germany','Spain','Argentina','France'])

plt.title('value<50000')

plt.figure()

sns.violinplot(x='Nationality',y='value',data=data[data['value']<2000],order=['England','Germany','Spain','Argentina','France'])

plt.title('value<20000')
sns.distplot(data['wage'])
sns.regplot(x='wage',y='value',data=data,marker='.')
sns.distplot(data['Age'],kde=False,color='deepskyblue')
sns.regplot(x='Age',y='value',data=data,fit_reg=False,color='darkgrey')
sns.boxplot(y='Overall',data=data,width=0.4,color='lime')
def log(x):

    return np.log10(x)

test=data[['Overall','value']].copy()

test['logvalue'] = test['value'].apply(log)
sns.lmplot('Overall','logvalue',data=test)
test.corr()
sns.boxplot(y = 'Potential',data=data,width=0.4)
data.plot.scatter('Potential','value',logy=True)

plt.ylim(1,10**6)