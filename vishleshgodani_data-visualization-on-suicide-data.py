# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
df.info()
df.head()
alpha =0.7

plt.figure(figsize=(10,25))

sns.barplot(x='suicides_no', y='country',data=df)

plt.xlabel('suicide count')
plt.figure(figsize=(16,7))

sns.countplot(x='sex',data=df)
plt.figure(figsize=(16,7))

sns.barplot(x='sex',y='suicides_no',hue='age',data=df)
plt.figure(figsize=(16,7))

sns.barplot(x='generation',y='suicides_no',hue='sex',data=df)

plt.figure(figsize=(16,7))

sns.barplot(x='sex',y='suicides_no',hue='generation',data=df)
sns.catplot('sex','suicides_no',hue='age',col='year',data=df,kind='bar',col_wrap=3)
age_5 = df.loc[df.loc[:, 'age']=='5-14 years',:]

age_15 = df.loc[df.loc[:, 'age']=='35-54 years',:]

age_25 = df.loc[df.loc[:, 'age']=='75+ years',:]

age_35 = df.loc[df.loc[:, 'age']=='25-34 years',:]

age_55 = df.loc[df.loc[:, 'age']=='55-34 years',:]

age_75 = df.loc[df.loc[:, 'age']=='15-24 years',:]
age2=df['age']=='15-24 years'
age2.head()
sns.lineplot(x='year',y='suicides_no',hue='age',data=df)
male=df['sex']=='male'

female=df['sex']=='female'
male.head()
sns.lineplot(x='year',y='suicides_no',hue='sex',data=df)