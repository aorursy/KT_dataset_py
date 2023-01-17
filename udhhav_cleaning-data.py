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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns
df = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
df.head()
df.info()
df.shape
df.isnull().sum()
df.drop(['Unnamed: 0'], axis=1,inplace=True)
df.columns
df['Job Title'].value_counts()
df.dropna(inplace=True)

df.shape
df['Salary Estimate'].head()
df['Salary Estimate'].str.split('-')
new=df["Salary Estimate"].str.split(" ", n = 1, expand = True)
sal_range=new[0].str.split('-',n=1,expand=True)
df['Lower'] = sal_range[0]

df['Upper'] = sal_range[1]

df.head()
df.drop('Salary Estimate',axis=1,inplace=True)

df.head()
df.replace(-1,np.nan,inplace=True)
df.replace(-1.0,np.nan,inplace=True)

df.replace('-1',np.nan,inplace=True)
df.head()
df.isnull().sum()
na_features = [features for features in df.columns if df[features].isnull().sum()>=1]

na_features
for feature in na_features:

    print(feature, np.round(df[feature].isnull().mean(),4),'% Missing Values.')
df['Lower']=df['Lower'].str.replace('K','000')

df['Lower']=df['Lower'].str.replace('$','')

df['Upper']=df['Upper'].str.replace('K','000')

df['Upper']=df['Upper'].str.replace('$','')
df.head()
df['Upper'].unique()
df['Lower'].replace('',0,inplace=True)

df['Lower']=df['Lower'].astype('int')
df['Upper'].replace('',0,inplace=True)

df['Upper']=df['Upper'].astype('int')
df.info()
df['Average_Sal'] = (df['Lower']+df['Upper'])/2
df.head()
st = df['Location'].str.split(',',n=1,expand=True)
df['State'] = st[1]
df.head()
plt.figure(figsize=(10,5))

sns.countplot(df['State'])