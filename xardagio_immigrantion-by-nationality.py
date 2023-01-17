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

from plotly import __version__

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()

%matplotlib inline
data_imm_nat = pd.read_csv('../input/immigrants_by_nationality.csv')
data_imm_nat.head()
sum(data_imm_nat['Number'])
data_imm_nat.info()
data_imm_nat['Nationality'].unique()
data_imm_nat['District Name'].unique()
data_imm_nat['Neighborhood Name'].unique()
data_imm_nat['Year'].unique()
data = dict(type='choropleth',

           locations=data_imm_nat['Nationality'].unique(),

           locationmode='country names',

           z=data_imm_nat.set_index(['Nationality'])['Number'].sum(level='Nationality'),

           colorscale='Jet',

           colorbar={'title':'Number'})



layout = dict(title='Number of immigration to Barcelona 2015-2017',

             geo=dict(showframe=False,projection={'type':'mercator'}))



choromap = go.Figure(data=[data],layout=layout)

iplot(choromap,validate=False)
filtered_data = data_imm_nat[data_imm_nat['Nationality']!='Spain']



data = dict(type='choropleth',

           locations=filtered_data['Nationality'].unique(),

           locationmode='country names',

           z=filtered_data.set_index(['Nationality'])['Number'].sum(level='Nationality'),

           colorscale='Jet',

           colorbar={'title':'Number'})



layout = dict(title='Number of abroad immigration to Barcelona 2015-2017',

             geo=dict(showframe=False,projection={'type':'mercator'}))



choromap = go.Figure(data=[data],layout=layout)

iplot(choromap,validate=False)
filtered_data = data_imm_nat[(data_imm_nat['Nationality']!='Spain') & (data_imm_nat['Year']==2015)]



data = dict(type='choropleth',

           locations=filtered_data['Nationality'].unique(),

           locationmode='country names',

           z=filtered_data.set_index(['Nationality'])['Number'].sum(level='Nationality'),

           colorscale='Jet',

           colorbar={'title':'Number'})



layout = dict(title='Number of abroad immigration to Barcelona 2015',

             geo=dict(showframe=False,projection={'type':'mercator'}))



choromap = go.Figure(data=[data],layout=layout)

iplot(choromap,validate=False)
filtered_data = data_imm_nat[(data_imm_nat['Nationality']!='Spain') & (data_imm_nat['Year']==2016)]



data = dict(type='choropleth',

           locations=filtered_data['Nationality'].unique(),

           locationmode='country names',

           z=filtered_data.set_index(['Nationality'])['Number'].sum(level='Nationality'),

           colorscale='Jet',

           colorbar={'title':'Number'})



layout = dict(title='Number of abroad immigration to Barcelona 2016',

             geo=dict(showframe=False,projection={'type':'mercator'}))



choromap = go.Figure(data=[data],layout=layout)

iplot(choromap,validate=False)
filtered_data = data_imm_nat[(data_imm_nat['Nationality']!='Spain') & (data_imm_nat['Year']==2017)]



data = dict(type='choropleth',

           locations=filtered_data['Nationality'].unique(),

           locationmode='country names',

           z=filtered_data.set_index(['Nationality'])['Number'].sum(level='Nationality'),

           colorscale='Jet',

           colorbar={'title':'Number'})



layout = dict(title='Number of abroad immigration to Barcelona 2017',

             geo=dict(showframe=False,projection={'type':'mercator'}))



choromap = go.Figure(data=[data],layout=layout)

iplot(choromap,validate=False)
x = data_imm_nat[(data_imm_nat['Year']==2015)].set_index(['Nationality'])['Number'].sum(level='Nationality')

y = data_imm_nat[(data_imm_nat['Year']==2016)].set_index(['Nationality'])['Number'].sum(level='Nationality')

z = data_imm_nat[(data_imm_nat['Year']==2017)].set_index(['Nationality'])['Number'].sum(level='Nationality')

sum_imm = pd.DataFrame()

sum_imm['2015'] = x

sum_imm['2016'] = y

sum_imm['2017'] = z

sum_imm = sum_imm.fillna(0).astype(int)
sum_imm_to1000 = sum_imm[sum_imm['2015']>1000]
sum_imm_to1000.head()
sum_imm_to1000_abroad = sum_imm_to1000.drop('Spain')
sum_imm_to1000_abroad.head()
layout = dict(title='Number of abroad immigration to Barcelona 2015-2017 - more than 1000 people',

             geo=dict(showframe=False))

sum_imm_to1000_abroad.iplot(kind='bar',layout=layout)
plt.figure(figsize=(25,15))

plt.title('Migrations to Barcelona by districts')

sns.barplot(data=data_imm_nat[data_imm_nat['District Name']!='No consta'], 

           y='District Name', 

           x='Number', 

           estimator=sum,

           hue='Year')
e = data_imm_nat[data_imm_nat['Nationality']!='Spain'].set_index(['District Name','Nationality'])

e = e.sort_index()['Number']

data_imm_dist = pd.DataFrame(e.sum(level=('District Name','Nationality')))
data_imm_dist_top100 = data_imm_dist[data_imm_dist['Number']>100]

data_imm_dist_top100 = data_imm_dist_top100.sort_values(by='Number',ascending=False)

data_imm_dist_top100 = data_imm_dist_top100.sort_index(level=0,ascending=[True])
data_imm_dist_top1000 = data_imm_dist[data_imm_dist['Number']>1000]

data_imm_dist_top1000 = data_imm_dist_top1000.sort_values(by='Number',ascending=False)

data_imm_dist_top1000 = data_imm_dist_top1000.sort_index(level=0,ascending=[True])
data_imm_dist_top100
data_imm_dist_top1000
data_imm_dist_all = data_imm_dist.sort_values(by='Number',ascending=False)

data_imm_dist_all = data_imm_dist_all.sort_index(level=0,ascending=[True])
data_imm_dist_all
layout = dict(title='Number of abroad immigration to Barcelona 2015-2017 by districts and nationalities - more than 1000 people',

             geo=dict(showframe=False))

data_imm_dist_top1000.iplot(kind='bar',layout=layout)
plt.figure(figsize=(80,20))

plt.title('Migrations to Barcelona by districts and nationalities')

sns.barplot(data=data_imm_dist_top1000,x=data_imm_dist_top1000.index,y='Number')