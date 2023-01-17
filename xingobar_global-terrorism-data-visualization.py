# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/globalterrorismdb.csv',encoding="latin1")
df.head()
year_count = df['eventid.1'].value_counts()

#year_count = year_count.sort_values(ascending=False)

year = np.array(list(year_count.index))

year_val = np.array(list(year_count.values))

fig,ax = plt.subplots(figsize=(8,6))

ax = sns.barplot(x=year,y=year_val,ax=ax,color='black')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)

plt.title('Global Terrorism per year')

plt.ylabel('Total Terrorism')

plt.xlabel('Year')
df['country_txt'].unique()
df.columns.tolist()
df['attacktype1_txt'].unique()
pivot = pd.pivot_table(data=df,index='eventid.1',

                       columns='attacktype1_txt',

                       values='attacktype1',

                       aggfunc='count')

fig,ax = plt.subplots(figsize=(8,6))

sns.heatmap(pivot,vmin=0,annot=False,ax=ax,cmap='Reds',linewidth=.5)

plt.title('Attack type per year')

plt.xlabel('Attack Type')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)

plt.ylabel('Year')
nkill_df = df[['eventid.1','nkill']].groupby(['eventid.1'])['nkill'].agg(['sum'])

nkill_df = nkill_df.reset_index()

fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(data=nkill_df,x='eventid.1',y='sum',color='black',ax=ax)

plt.title('Number of kill per year')

plt.xlabel('year')

plt.ylabel('Count')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)
df['targtype1_txt'].unique()

pivot = pd.pivot_table(data=df,index='eventid.1',

                      columns='targtype1_txt',

                      values='targtype1',

                      aggfunc='count')

fig,ax = plt.subplots(figsize=(8,6))

sns.heatmap(pivot,vmin=0,ax=ax,annot=False,linewidth=.5,cmap='Reds')

plt.title('Target Type per year')

plt.xlabel('Target')

plt.ylabel('Year')
df['success'].unique()

target_success = df[['eventid.1','targtype1','targtype1_txt','success']]

target_success = target_success.groupby(['targtype1_txt','success'])['success'].agg(['count'])

target_success = target_success.reset_index()

fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(data=target_success,x='targtype1_txt',y='count',hue='success')

plt.xlabel('Target Type')

plt.ylabel('Count')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)

plt.title('target type vs success')
df['region_txt'].unique()