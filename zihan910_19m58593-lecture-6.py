import numpy as np 

import matplotlib.pyplot as plt 

import pandas as pd

np.set_printoptions(threshold=np.inf)



selected_country = 'Italy'

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)

#select the dataframe that contains Italy

df = df[df['Country/Region']==selected_country]

df = df.groupby('ObservationDate').sum()

print(df)
df['Daily Confrimed'] = df['Confirmed'].diff()

df['Daily Deaths'] = df['Deaths'].diff()

df['Daily Recovered'] = df['Recovered'].diff()



df['Daily Confrimed'].plot()

df['Daily Deaths'].plot()

df['Daily Recovered'].plot()
df1 = df#[['Daily Confrimed']]

#Replace non value with zeros

df1 = df1.fillna(0.)

styled_object = df1.style.background_gradient(cmap='jet').highlight_max('Daily Confrimed').set_caption('Daily Summaries')

display(styled_object)

#export the output to an html file with option to write

f = open('table_IDNumber.html','w')

f.write(styled_object.render())
#Create an interactive table for Italy daily confirmed and death cases

from plotly.offline import iplot

import plotly.graph_objs as go



daily_confirmed_object = go.Scatter(x=df.index,y=df['Daily Confrimed'].values,name='Daily confirmed')

daily_deaths_object = go.Scatter(x=df.index,y=df['Daily Deaths'].values,name='Daily deaths')



layout_object = go.Layout(title='Italy daily cases 19M58593',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object],layout=layout_object)



iplot(fig)

print(df)
#Reading the data file to create the country ranking of the confirmed cases

data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data.head()
#Reduce the columns to include only useful information for the ranking

data.index=data['ObservationDate']

data = data.drop(['SNo','ObservationDate'],axis=1)

data.head()
data_italy = data[data['Country/Region']=='Italy']

data_italy.tail()
#Ranking by 'Confirmed' case in Italy

latest = data[data.index=='06/02/2020']

latest = latest.groupby('Country/Region').sum()

latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 



#Italy's Ranking

print('Ranking of Italy is: ', latest[latest['Country/Region']=='Italy'].index.values[0]+1)
