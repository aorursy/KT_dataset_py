import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import numpy as np

sns.set(style="whitegrid")

warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv',encoding='iso-8859-1')

df.head()
group_state = df.groupby('state')['number'].agg('sum').reset_index().sort_values(ascending=False,by='number')

fig,ax = plt.subplots(figsize=(12,8))

ax = sns.barplot(data=group_state,x='state',y='number')

ax.set(xlabel='State',ylabel='Total Num. of Fires',title='Total Fires by State 1998-2017')

plt.xticks(rotation='vertical')

plt.show()
state_year = df.groupby(['year','state'])['number'].agg('sum').reset_index()

fig,ax = plt.subplots(figsize=(12,8))

ax= sns.lineplot(ax=ax,data=state_year[state_year.state == 'Mato Grosso'],x='year',y='number')

ax.set(xlabel='Year',ylabel='Total Num. of Fires',

       title='Total Num. of Fires by Year in Mato Grosso 1998-2017')

plt.xlim(1998,2017)

plt.xticks(np.arange(1998, 2018, 1),fontsize=12)

plt.show()
month_map = {'Janeiro':1,'Fevereiro':2,'Março':3,'Abril':4,'Maio':5,

             'Junho':6,'Julho':7,'Agosto':8,'Setembro':9,'Outubro':10,'Novembro':11,'Dezembro':12}

state_month = df.groupby(['month','state'])['number'].agg('sum').reset_index()

state_month.month = state_month.month.map(month_map).astype('category')

fig,ax = plt.subplots(figsize=(12,8))

ax= sns.lineplot(ax=ax,data=state_month[state_month.state == 'Mato Grosso'],x='month',y='number')

ax.set(xlabel='Month',ylabel='Total Num. of Fires',title='Total Num. of Fires by Month in Mato Grosso 1998-2017')



plt.show()
fig,ax = plt.subplots(figsize=(12,8))

ax= sns.lineplot(ax=ax,data=state_year,x='year',y='number',hue='state')

ax.set(xlabel='Year',ylabel='Total Num. of Fires',title='Total Num. of Fires by Year and State 1998-2017')

plt.show()
fig,ax = plt.subplots(figsize=(12,8))

ax= sns.lineplot(ax=ax,data=state_year[state_year.year >= 2015],x='year',y='number',hue='state')

ax.set(xlabel='Year',ylabel='Total Num. of Fires',title='Total Num. of Fires by Year and State 2015-2017')

plt.show()
df.date = pd.to_datetime(df.date)

group_year = df.groupby(df.year)['number'].agg('sum').reset_index()

fig,ax = plt.subplots(figsize=(12,8))

ax= sns.lineplot(ax=ax,data=group_year,x='year',y='number')

ax.set(xlabel='Year',ylabel='Total Num. of Fires',title='Total Num. of Fires by Year')

plt.show()