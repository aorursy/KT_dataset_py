# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import plotly.graph_objects as go

import seaborn as sns

import matplotlib.pyplot as plt

import plotly

import plotly.express as pc

import holoviews as hv

from holoviews import opts

hv.extension('bokeh','matplotlib')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

data2020 = pd.read_csv('/kaggle/input/world-happiness-report/2020.csv')

data2019 = pd.read_csv('/kaggle/input/world-happiness-report/2019.csv')

data2018 = pd.read_csv('/kaggle/input/world-happiness-report/2018.csv')

data2017 = pd.read_csv('/kaggle/input/world-happiness-report/2017.csv')

data2016 = pd.read_csv('/kaggle/input/world-happiness-report/2016.csv')

data2015 = pd.read_csv('/kaggle/input/world-happiness-report/2015.csv')
data2015['year']=2015

data2016['year']=2016

data2017['year']=2017

data2018['year']=2018

data2019['year']=2019

data2020['year']=2020
data2017.rename(columns={'Economy  GDP per Capita':'Economy (GDP per Capita)'},inplace=True)

data2018.rename(columns={'Country or region':'Country','Score':'Happiness Score','GDP per Capita':'Economy (GDP per Capita)'},inplace=True)

data2019.rename(columns={'Country or region':'Country','Score':'Happiness Score','Economy  GDP per Capita':'Economy (GDP per Capita)'},inplace=True)

data2020.rename(columns={'Country name':'Country','Ladder score':'Happiness Score','Regional indicator':'Region','Logged GDP per capita':'Economy (GDP per capita)'},inplace=True)
data2015.set_index('Country',inplace=True)

data2016.set_index('Country',inplace=True)

data2017.set_index('Country',inplace=True)

data2018.set_index('Country',inplace=True)

data2019.set_index('Country',inplace=True)

data2020.set_index('Country',inplace=True)
for c in data2017.index:

    if c in data2016.index:

        data2017.loc[c,'Region'] = data2016.loc[c,'Region']

    elif(c.find('China')!=-1):

        data2017.loc[c,'Region'] = 'Eastern Asia'

    else:

        data2017.loc[c,'Region'] = 'Middle East and Northern Africa'

        

        
data2015.drop(columns='Standard Error',axis=1,inplace=True)

data2016.drop(columns=['Lower Confidence Interval','Upper Confidence Interval'],axis=1,inplace=True)

data2017.drop(columns=['Whisker.low','Whisker.high'],axis=1,inplace=True)

data2020.drop(columns=['Standard error of ladder score', 'upperwhisker', 'lowerwhisker'],axis=1,inplace=True)

data2017.columns = [i.replace('.',' ') for i in data2017.columns]
dataCombined = (data2015.append(data2016)).append(data2017)

dataCombined = (dataCombined.append(data2018)).append(data2019)

dataCombined = dataCombined.append(data2020)
dataCombined.fillna(value=0,inplace=True)
dataCombined.reset_index()
l = dataCombined.reset_index().columns

cols = l[2:11].append(l[12:]) #Exclude Country,Region and Year

dataFinal = pd.pivot_table(dataCombined,values=cols,index=['Country','Region','year'],aggfunc=np.mean)
dataFinal.reset_index(inplace=True)
dataFinal.sort_values(by='year',ascending=True,inplace=True)


fig = pc.scatter_geo(dataFinal,

    locations = dataFinal['Country'],

    locationmode = "country names",

    hover_data = ['Country','Happiness Score','year'],

    size='Happiness Score',

    color='Happiness Score',animation_frame='year',width=1200,height=800)



fig.update_layout(title = dict(text="World Happiness Report(2015-2020)",y=0.9,x=0.5,xanchor='center',yanchor='top'),geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'),margin=dict(l=20, r=50, t=110, b=20),font=dict(size=15,color='Blue'))





fig.show()
dataFinal['year_rank'] = dataFinal.groupby('year')['Happiness Score'].rank(ascending=False)

dataFinal.sort_values(['year', 'year_rank'],inplace=True)
d1 = dataFinal[(dataFinal['year']==2015) & (dataFinal['year_rank']<11)]

d2 = dataFinal[(dataFinal['year']==2016) & (dataFinal['year_rank']<11)]

d3 = dataFinal[(dataFinal['year']==2017) & (dataFinal['year_rank']<11)]

d4 = dataFinal[(dataFinal['year']==2018) & (dataFinal['year_rank']<11)]

d5 = dataFinal[(dataFinal['year']==2019) & (dataFinal['year_rank']<11)]

d6 = dataFinal[(dataFinal['year']==2020) & (dataFinal['year_rank']<11)]

d1 = d1[['Country','Happiness Score','year','year_rank']].sort_values(by='year_rank')

d2 = d2[['Country','Happiness Score','year','year_rank']].sort_values(by='year_rank')

d3 = d3[['Country','Happiness Score','year','year_rank']].sort_values(by='year_rank')

d4 = d4[['Country','Happiness Score','year','year_rank']].sort_values(by='year_rank')

d5 = d5[['Country','Happiness Score','year','year_rank']].sort_values(by='year_rank')

d6 = d6[['Country','Happiness Score','year','year_rank']].sort_values(by='year_rank')

top10 = d1.append(d2).append(d3)

top10 = top10.append(d4).append(d5).append(d6)
b1=dataFinal[(dataFinal['year']==2015)].tail(10)

b2=dataFinal[(dataFinal['year']==2016)].tail(10)

b3=dataFinal[(dataFinal['year']==2017)].tail(10)

b4=dataFinal[(dataFinal['year']==2018)].tail(10)

b5=dataFinal[(dataFinal['year']==2019)].tail(10)

b6=dataFinal[(dataFinal['year']==2020)].tail(10)



b1 = b1[['Country','Happiness Score','year','year_rank']].sort_values(by='year_rank')

b2 = b2[['Country','Happiness Score','year','year_rank']].sort_values(by='year_rank')

b3 = b3[['Country','Happiness Score','year','year_rank']].sort_values(by='year_rank')

b4 = b4[['Country','Happiness Score','year','year_rank']].sort_values(by='year_rank')

b5 = b5[['Country','Happiness Score','year','year_rank']].sort_values(by='year_rank')

b6 = b6[['Country','Happiness Score','year','year_rank']].sort_values(by='year_rank')
bot10 = b1.append(b2).append(b3)

bot10 = bot10.append(b4).append(b5).append(b6)
key_dimensions   = [('year', 'Year'), ('Country', 'Country')]

value_dimensions = [('Happiness Score', 'Happiness Score')]

macro = hv.Table(top10, key_dimensions, value_dimensions)

bars = macro.to.bars(['year', 'Country'], 'Happiness Score', [])

bars.opts(

    opts.Bars(color=hv.Cycle('Category20'), show_legend=False, stacked=True, 

              tools=['hover'], width=600,height=600, xrotation=90,title="Happy Countries Scores"))
key_dimensions   = [('year', 'Year'), ('Country', 'Country')]

value_dimensions = [('Happiness Score', 'Happiness Score')]

macro = hv.Table(bot10, key_dimensions, value_dimensions)

bars = macro.to.bars(['year', 'Country'], 'Happiness Score', [])

bars.opts(

    opts.Bars(color=hv.Cycle('Category20'), show_legend=False, stacked=True, 

              tools=['hover'], width=600, xrotation=90,title='Low Happiness Scores'))