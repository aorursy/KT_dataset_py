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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/unemployment.csv')
data.head()
data.info()
data['Number'].describe()
plt.figure(figsize=(20,15))

sns.barplot(data=data,y='Number',x='Year',hue='Gender',estimator=sum)
plt.figure(figsize=(20,15))

sns.barplot(data=data[data['District Name']!='No consta'],y='Number',x='District Name',hue='Year',estimator=sum)
plt.figure(figsize=(20,15))

sns.barplot(data=data,y='Number',x='Year',hue='Month',estimator=sum)
plt.figure(figsize=(20,15))

sns.barplot(data=data,y='Number',x='Year',hue='Demand_occupation',estimator=sum)
data[data['Year']==2013].groupby(['District Name','Neighborhood Name'])['Number'].sum().sort_values()
data[data['Year']==2014].groupby(['District Name','Neighborhood Name'])['Number'].sum().sort_values()
data[data['Year']==2015].groupby(['District Name','Neighborhood Name'])['Number'].sum().sort_values()
data[data['Year']==2016].groupby(['District Name','Neighborhood Name'])['Number'].sum().sort_values()
data[data['Year']==2017].groupby(['District Name','Neighborhood Name'])['Number'].sum().sort_values()
data[data['District Name']!='No consta'].groupby(['Year','District Name','Neighborhood Name'])['Number'].sum().sort_values()