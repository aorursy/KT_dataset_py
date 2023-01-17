import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.plotly as py

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()
suicide = pd.read_csv("../input/master.csv")
suicide.head()
suicide.info()
suicide.describe()
suicide.set_index('country',inplace=True)

suicide.head()
sns.heatmap(suicide.isnull(),yticklabels=False,cbar=False,cmap='viridis')
suicide.drop('HDI for year',axis=1,inplace=True)

suicide.head(2)
sns.pairplot(suicide,hue = 'sex', palette = 'gist_rainbow')
sns.heatmap(suicide.corr(), cmap='rainbow', annot=True)
sns.set_style('whitegrid')

plt.figure(figsize=(11,8))

sns.barplot(x='age', y='suicides_no', hue='sex', data=suicide.sort_values('age'), palette='gist_ncar')

plt.title("Suicide counts with respective to age", fontsize = 20)

plt.xlabel("Age", fontsize = 12)

plt.ylabel("Suicide counts", fontsize = 12)
suicide_no_sort = suicide.groupby('country').sum().sort_values('suicides_no',ascending=False)

suicide_no_sort = suicide_no_sort[suicide_no_sort.suicides_no > 1000]

plt.figure(figsize=(20,75))

sns.barplot(y=suicide_no_sort.index, x='suicides_no', data=suicide_no_sort, palette='tab20c')

plt.title("Countries with suicide counts more than 1K", fontsize = 20)

plt.xlabel("Country", fontsize = 12)

plt.ylabel("Suicide counts", fontsize = 12)
suicide_byyear = suicide.groupby('year').sum()
plt.figure(figsize=(15,6))

sns.lineplot(x=suicide_byyear.index, y='suicides_no', data=suicide_byyear, color='red', marker='o')

plt.title("Suicide counts across the globe over the year", fontsize = 20)

plt.xlabel("Year", fontsize = 12)

plt.ylabel("Suicide counts", fontsize = 12)
suicides_per100k = suicide.groupby('country-year').sum().sort_values('suicides/100k pop', ascending=False)

suicides_per100k = suicides_per100k[suicides_per100k['suicides/100k pop'] > 500]
suicides_per100k.head()
plt.figure(figsize=(25,30))

sns.barplot(y=suicides_per100k.index, x='suicides/100k pop', data = suicides_per100k)

plt.title("Countries with most suicide counts in specific years(suicides/100K population)", fontsize = 20)

plt.xlabel("suicides/100K population", fontsize = 12)

plt.ylabel("Country/Year", fontsize = 12)
suicide.iplot(kind='pie',labels='generation',values='suicides/100k pop', pull=0.1,

         colorscale = 'paired', textposition='outside', textinfo='value+percent',title ="Suicide percent per Generation")
mean_suicide = suicide.groupby('country').mean()

mean_suicide.head()
data = dict(type = 'choropleth',

            colorscale = 'Reds',

            locations = mean_suicide.index,

            locationmode = 'country names',

            z = mean_suicide['suicides_no'],

            text = mean_suicide.index,

            colorbar = {'title':'Mean suicide counts'})



layout = dict(title = 'Suicide counts across the Globe',

               geo = dict(showframe = False, projection = {'type':'natural earth'}))



choromap = go.Figure([data],layout)

iplot(choromap)

suicide_uk = suicide[suicide.index=='United Kingdom'][['year','age','suicides_no']]
plt.figure(figsize=(15,13))

sns.lineplot(x='year',y='suicides_no',data=suicide_uk,hue='age',marker='o')

plt.title("UK's suicide trend over the years", fontsize = 20)

plt.xlabel("Year", fontsize = 12)

plt.ylabel("Suicides count", fontsize = 12)
sns.jointplot(x='gdp_per_capita ($)',y='suicides_no',data=suicide[suicide.index=='United Kingdom'],

              kind='hex',color ='purple').set_axis_labels("GDP/capita ($)", "Suicides count")
plt.figure(figsize=(15,4))

sns.lmplot(x='year',y='suicides/100k pop',palette='prism',hue='sex',data=suicide[suicide.index=='United Kingdom'])

plt.title("UK's suicide trend over the years", fontsize = 17)

plt.xlabel("Year", fontsize = 12)

plt.ylabel("Suicides/100K", fontsize = 12)