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
df = pd.read_csv("../input/nyc-jobs.csv")

df.head(3)
df.info()
df.describe()
df['Business Title'].value_counts().head()
import re

df['degree'] = df['Minimum Qual Requirements'].str.extract(r'(degree)')

df['degree'] = df['degree'].map({'degree':1,np.nan:0})

df['degree'].value_counts()
sns.jointplot(x='degree', y='Salary Range From', data=df[['degree','Salary Range From']])
df.groupby('degree')['Salary Range From'].mean()
df['experience'] = df['Minimum Qual Requirements'].str.extract(r'(\w+)\syears.+experience')

years = {'two':2,

        'three':3,

        'four':4,

        'five':5,

        'six':6,

        '4':4,

        'eight':8,

        'seven':7,

        'ten':10,

        '10':10,

        '3':3,

        '5':5,

        '2':2

}

df['experience'] = df['experience'].str.lower().map(years)

df['experience'].value_counts()



#Seems it takes at least two years to gain any worthy experience
import seaborn as sns

sns.distplot(df['experience'].dropna())
#Only the slightest notable difference in starting salary from years of experiance.

sns.jointplot(x='experience', y='Salary Range From', data=df[['experience','Salary Range From']])