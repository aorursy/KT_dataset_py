# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))

df =pd.read_csv('../input/rural_urban.csv')

# Any results you write to the current directory are saved as output.

df.head(10)
df = df.drop(df.index[:7])

df.head()
df.groupby('area')['transgender'].agg('sum').sort_values(ascending=False).head(10).plot(kind='bar')
df.groupby('area')['average annual growth rate'].agg('sum').sort_values(ascending=False).head(10).plot(kind='bar')
df1 =df.loc[df['female'] >= df['male']]

df1['no of women greater']=df1['female'] - df1['male']

df1.groupby('area')['no of women greater'].agg('sum').sort_values(ascending=False).head(10).plot(kind='bar')