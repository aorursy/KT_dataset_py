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
df = pd.read_csv('../input/telecom_churn.csv')

df.head()

df.columns

df.info()

df.sort_values(by='Total day charge', ascending=False).head()

df['Churn'] = df['Churn'].astype('int64')

df['Churn'].value_counts()
df.loc[0:5, 'State':'Area code']
df.iloc[0:5, 4:6]
df.iloc[-1:]
df[df['State'].apply(lambda state:state[0] == 'W')].head()
d = {'No' : False, 'Yes' : True}

df['International plan'] = df['International plan'].map(d)

df.head()
df = df.replace({'Voice mail plan': d})

df.head()
columns_to_show = ['Total day minutes', 'Total eve minutes', 'Total night minutes']

df.groupby(['Churn'])[columns_to_show].describe(percentiles=[])
pd.crosstab(df['Churn'], df['International plan'])
total_calls = df['Total day calls'] + df['Total eve calls'] + df['Total night calls'] + df['Total intl calls']

df.insert(loc=len(df.columns), column='Total calls', value=total_calls) 

df.head()
df['Total charge'] = df['Total day charge'] + df['Total eve charge'] + df['Total night charge'] + df['Total intl charge']

df.head()
# Delete columns

df.drop(['Total calls', 'Total charge'], axis=1, inplace=True)

# Delete rows

df.drop([1, 2]).head()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams['figure.figsize'] = (8, 6)

sns.countplot(x='International plan', hue='Churn', data=df);
pd.crosstab(df['Churn'], df['Customer service calls'], margins=True)

sns.countplot(x='Customer service calls', hue='Churn', data=df)
df['Many_service_calls'] = (df['Customer service calls'] > 3).astype('int')

df.head()
pd.crosstab(df['Many_service_calls'], df['Churn'], margins=True)

sns.countplot(x='Many_service_calls', hue='Churn', data=df);
pd.crosstab(df['Many_service_calls'] & df['International plan'], df['Churn'])
sns.countplot(x='International plan', hue='Churn', data=df);