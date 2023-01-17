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
df1=pd.read_csv("../input/ontario/ontario-public-sector-salary-2019.csv")
df2 = pd.read_csv("../input/ontario/ontario-public-sector-salary-2018.csv")
df1.head()
df2.head()
# checking null values

print(df1.isnull().sum())
# checking null values

print(df2.isnull().sum())
# renaming the column

df2 = df2.rename(columns = {'Salary Paid':'Salary','Taxable Benefits':'Benefits','Calendar Year':'Year'})
df2
# renaming the column

df1 = df1.rename(columns = {'Salary Paid':'Salary','Taxable Benefits':'Benefits','Calendar Year':'Year'})
df1
df1.dtypes
# cleaning the data

df1 = df1.dropna()
import re

df1['Salary'] = df1['Salary'].apply(lambda x : float(re.sub(',','',x)[1:]))
df1['Benefits'] = df1['Benefits'].apply(lambda x : float(re.sub(',','',x)[1:]))

df1
# cleaning the data

df2 = df2.dropna()
import re

df2['Salary'] = df2['Salary'].apply(lambda x : float(re.sub(',','',x)[1:]))
df2['Benefits'] = df2['Benefits'].apply(lambda x : float(re.sub(',','',x)[1:]))

df2
# heat map

import matplotlib.pyplot as plt
import seaborn as sns
correlation = df1.corr()
plt.figure(figsize = (12 , 12))
sns.heatmap(correlation)
import numpy as np 
import plotly 
import plotly.graph_objects as go 

x = list(df1['Sector'].unique())
y = list(df1['Salary'].groupby(df1['Sector']).max())
fig = plt.figure(figsize = (15, 5)) 
  
# creating the bar plot 
plt.bar(x, y, color ='orange',  
        width = 0.6) 
  
plt.xlabel("Sector") 
plt.ylabel("Max salary") 
plt.xticks(rotation=90)
plt.title("Max salary of each sector") 
plt.show() 
import plotly.express as px

m = list(df2['Sector'].unique())
n = list(df2['Salary'].groupby(df2['Sector']).max())
fig = plt.figure(figsize = (15, 5)) 
  
# creating the bar plot 
plt.bar(m, n, color ='pink',  
        width = 0.6) 
  
plt.xlabel("Sector") 
plt.ylabel("Max salary") 
plt.xticks(rotation=90)
plt.title("Max salary of each sector") 
plt.show() 
# plot

my_circle=plt.Circle( (0,0), 0.9, color='white')
plt.pie(df1['Job Title'].value_counts()[:10].values, labels = df1['Job Title'].value_counts()[:10].index)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

# plot

my_circle=plt.Circle( (0,0), 0.9, color='white')
plt.pie(df2['Job Title'].value_counts()[:10].values, labels = df2['Job Title'].value_counts()[:10].index)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()