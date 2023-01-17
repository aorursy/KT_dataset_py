import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



selected_country = 'Italy'



df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", header = 0)

df = df[df['Country/Region'] == selected_country]

df = df.groupby('ObservationDate').sum()



df['daily_confirmed'] = df['Confirmed'].diff()

df['daily_deaths'] = df['Deaths'].diff()

df['daily_recovered'] = df['Recovered'].diff()
from plotly.offline import iplot

import plotly.graph_objs as go





daily_confirmed_object = go.Scatter(x = df.index, y = df['daily_confirmed'].values, name = 'Daily confirmed')

daily_deaths_object = go.Scatter(x = df.index, y = df['daily_deaths'].values, name = 'Daily deaths')

daily_recovered_object = go.Scatter(x = df.index, y = df['daily_recovered'].values, name = 'Daily recovered')



layout = go.Layout(title = "Covid-19 in Italy", xaxis = dict(title = 'Date'), yaxis = dict(title = 'Case Number') )

figure = go.Figure(data = [daily_confirmed_object, daily_deaths_object, daily_recovered_object], layout = layout)

iplot(figure)



figure.write_html("plot.html")
df = df.fillna(0.)

styled_object = df.style.background_gradient(cmap = 'gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')

display(styled_object)

file = open('table.html', 'w')

file.write(styled_object.render())
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", header = 0)

df = df[df['ObservationDate'] == '06/07/2020']

df = df.groupby(['Country/Region']).sum().sort_values(by='Confirmed',ascending=False).reset_index()

print(df[df['Country/Region'] == 'Italy'].index.values[0] + 1)