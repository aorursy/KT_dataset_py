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
!pip install calmap

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import calmap

from pandas_profiling import ProfileReport

df=pd.read_csv('/kaggle/input/supermarket-sales/supermarket_sales - Sheet1.csv')
df.head()
df.columns
df.dtypes
df['Date']=pd.to_datetime(df['Date'])

df.set_index('Date',inplace=True)

df.describe()
sns.distplot(df['Rating'])

plt.axvline(x=np.mean(df['Rating']),c='red',ls='--',label='mean')

plt.axvline(x=np.percentile(df['Rating'],25),c='green',ls='--',label='25-75th percentile')

plt.axvline(x=np.percentile(df['Rating'],75),c='green',ls='--')



df.hist(figsize=(12,12))
sns.countplot(df['Branch'])
df['Branch'].value_counts()
sns.countplot(df['Payment'])
sns.regplot(df['Rating'],df['gross income'])
sns.boxplot(x=df['Branch'],y=df['gross income'])
sns.boxplot(x=df['Gender'],y=df['gross income'])
df.groupby(df.index).mean()
sns.lineplot(x=df.groupby(df.index).mean().index,

            y=df.groupby(df.index).mean()['gross income'])
sns.pairplot(df)
df.duplicated()
df.drop_duplicates(inplace=True)
df.isna().sum()
df.fillna(df.mean(),inplace=True)
dataset=pd.read_csv('/kaggle/input/supermarket-sales/supermarket_sales - Sheet1.csv')

prof=ProfileReport(dataset,title='Profiling Report')

prof.to_widgets()
df.corr()
np.corrcoef(df['gross income'],df['Rating'])
sns.heatmap(round(df.corr(),3),annot=True)