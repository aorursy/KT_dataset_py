# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/raw_data.csv")

df.head()
df.shape
df = df.drop(['Source_1','Source_2','Source_3','State Patient Number','Estimated Onset Date','Status Change Date'],axis=1)

df.head()
df.shape
df.dtypes
df.describe()
df['Age Bracket'].plot.hist(bins=50)
df['Date Announced'].value_counts().plot(kind='bar')
(df['Gender'].value_counts()/len(df['Gender'])*100).plot(kind='bar')
df['Detected State'].value_counts()
df['Detected State'].value_counts().plot(kind='bar')
## Lets look for which districts are most affected in Maharashtra



temp_df=df.loc[df['Detected State']=='Maharashtra']

temp_df
temp_df['Notes'].value_counts()
temp_df['Detected District'].value_counts().plot(kind='bar')
## Lets look for which city are most affected in Maharashtra



temp_df['Detected City'].value_counts().plot(kind='bar')
## Lets look for which districts are most affected in Karnataka



temp2_df=df.loc[df['Detected State']=='Karnataka']

temp2_df
temp2_df['Detected District'].value_counts().plot(kind='bar')
## Current Status of the patients



df['Current Status'].value_counts()
df['Current Status'].value_counts().plot(kind='bar')
df['Nationality'].value_counts()
df['Nationality'].value_counts().plot.bar()
df['Notes'].value_counts()
temp3_df = df.loc[(df['Nationality']=='India') & ((df['Notes']=='Travelled from Dubai') | (df['Notes']=='Travelled from UK'))]

temp3_df
temp3_df.shape
temp4_df = df.loc[df['Nationality']=='India']

temp4_df['Notes'].value_counts()
temp4_df['Current Status'].value_counts()
temp4_df['Current Status'].value_counts().plot.bar()
temp5_df = df.loc[df['Current Status']=='Deceased']

temp5_df['Age Bracket'].value_counts().plot.bar()
temp6_df = df.loc[df['Current Status']=='Recovered']

temp6_df['Age Bracket'].value_counts().plot.bar()
temp7_df= df[['Age Bracket','Notes','Current Status']]

temp7_df
males = df[df['Gender']=='M']

females = df[df['Gender']=='F']



m=males['Current Status'].value_counts()/len(males)*100

f=females['Current Status'].value_counts()/len(females)*100

m, f
m.plot(kind='bar')
f.plot(kind='bar')
df.describe()
df.corr()
df.isnull().sum()
tem = ['Date Announced','Detected City','Gender','Detected District','Detected State','Current Status','Nationality']



for i in tem:

    print('--------------********-------------')

    print(df[i].value_counts())