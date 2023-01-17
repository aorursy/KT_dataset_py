import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold=np.inf)



selected_country = 'South Korea'

cv = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)

#print(cv.columns)

cv = cv[cv['Country/Region']==selected_country]

cv = cv.groupby('ObservationDate').sum()

print(cv)
cv['Daily_confirmed'] = cv['Confirmed'].diff()

cv['Daily_deaths'] = cv['Deaths'].diff()

cv['Daily_recovered'] = cv['Recovered'].diff()

cv['Daily_confirmed'].plot()

cv['Daily_recovered'].plot()

plt.show()
from plotly.offline import iplot

import plotly.graph_objs as go



Daily_confirmed_object = go.Scatter(x=cv.index, y=cv['Daily_confirmed'].values, name='Daily_confirmed')

Daily_death_object = go.Scatter(x=cv.index, y=cv['Daily_deaths'].values, name='Daily_deaths')



layout_object = go.Layout(title='South Korea daily cases 19M58481', xaxis=dict(title='Date'), yaxis=dict(title='Number of People'))

fig = go.Figure(data=[Daily_confirmed_object, Daily_death_object], layout=layout_object)

iplot(fig)
cv1 = cv

cv1 = cv1.fillna(0.)





styled_object1 = cv1.style.background_gradient(cmap='gist_ncar').highlight_max('Daily_confirmed').set_caption('Daily Summaries')

display(styled_object1)
cv = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')



date_rank = cv

date_rank = date_rank.groupby('Country/Region').sum()

date_rank = date_rank.sort_values(by='Confirmed',ascending=False).reset_index()



print('Ranking of South Korea: ', date_rank[date_rank['Country/Region']=='South Korea'].index.values[0]+1)