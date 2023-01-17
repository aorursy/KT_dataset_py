# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# http://stackoverflow.com/questions/31755900/python-3-xs-dictionary-view-objects-and-matplotlib

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

from collections import Counter

from mpl_toolkits.basemap import Basemap

from matplotlib.patches import Polygon

from matplotlib.colors import LinearSegmentedColormap

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/SPD_Reports.csv')
df.head()
df.shape
dates = pd.to_datetime(df['Report Date'],dayfirst=True, errors='coerce')



## day

day = dates.dropna().map(lambda x:x.day)

day_counter = Counter(day)

day_keys = np.array(list(day_counter.keys()))

day_values = np.array(list(day_counter.values())).astype(float)



## month

month = dates.dropna().map(lambda x:x.month)

month_counter = Counter(month)

month_keys = np.array(list(month_counter.keys()))

month_values = np.array(list(month_counter.values())).astype(float)
fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x = day_keys, y = day_values,ax=ax,color='#81d4fa')

plt.title('Day Counter',fontsize=15)

plt.ylabel('Counter',fontsize=15)

plt.xlabel('Day',fontsize=15)
fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x = month_keys,y=month_values,ax=ax,color='#81d4fa')

plt.title('Month Counter',fontsize=15)

plt.xlabel('Month',fontsize=15)

plt.ylabel('Counter',fontsize=15)
# date process

df['Date'] = df['Report Date'].map(lambda x:str(x).split('T')[0])

df.head()
year =  dates.dropna().map(lambda x:x.year)

year_counter = Counter(year)

year_keys = np.array(list(year_counter.keys()))

year_values = np.array(list(year_counter.values())).astype(float)
fig,ax = plt.subplots(figsize=(8,6))

plt.title('Year',fontsize=15)

sns.barplot(x=year_keys,y=year_values,ax=ax,color='#81d4fa')

plt.ylabel('Counter',fontsize=15)

plt.xlabel('Year',fontsize=15)

ticks = plt.setp(ax.get_xticklabels(),fontsize=12,rotation=90)
df['Year']  = df['Report Date'].map(lambda x:str(x)[:4])

df['Month'] = df['Report Date'].map(lambda x:str(x)[5:7])

df.head()
df_copy = df.dropna(axis=0)

table_count = pd.pivot_table(df_copy,index=['Month'],

                             columns=['Year'],

                             values=['Offense Type'],

                             aggfunc='count')

table_count.head()
fig,ax = plt.subplots(figsize=(8,6))

plt.title('Count vs year-month',fontsize=15)

sns.heatmap(table_count['Offense Type'],annot=False,vmin=0,ax=ax)
table_count = pd.pivot_table(df_copy,index = ['District'],

                             columns = ['Year'],values=['Offense Type'],

                             aggfunc='count')



fig,ax = plt.subplots(figsize=(8,6))

sns.heatmap(table_count['Offense Type'],annot=False,ax=ax,vmin=0)

plt.title('District vs Year',fontsize=15)
offense_type = Counter(df['Offense Type'].dropna().tolist()).most_common(10)

offense_type_indexs = [offense[0] for offense in offense_type]

offense_type_values = [offense[1] for offense in offense_type]



fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x = offense_type_indexs,y = offense_type_values,color='#81d4fa',ax=ax)

plt.title('Top Ten Offense Type',fontsize=15)

plt.xlabel('Offense Type',fontsize=15)

plt.ylabel('Count',fontsize=15)

ticks = plt.setp(ax.get_xticklabels(),fontsize=10,rotation=90)
table_count = pd.pivot_table(df_copy[df_copy['Offense Type'].isin(offense_type_indexs)],

                             index=['Year'],columns=['Offense Type'],

                             values=['District'],aggfunc='count')

#table_count.head()

fig,ax = plt.subplots(figsize=(8,6))

sns.heatmap(table_count['District'],annot=False,ax=ax,vmin=0)

plt.title('Offense Type and distrcit vs Year',fontsize=15)