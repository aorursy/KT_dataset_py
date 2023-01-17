import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

df.Date = pd.to_datetime(df.Date, format='%d/%m/%y')

df.head()
# Plotting using plotly

import plotly.express as px



dfDate = df.groupby(['Date'])['Cured', 'Confirmed', 'Deaths'].sum().reset_index()

dfDate.Confirmed.pct_change() * 100 



fig = px.bar(data_frame=df.sort_values('Confirmed', ascending=False),

             x='Date', y='Confirmed',

             color='State/UnionTerritory',

             title='# of Confirmed cases by State sorted by Date'

            )

fig.show()





fig = px.bar(data_frame=df.sort_values('Deaths', ascending=False),

             x='Date', y='Deaths',

             color='State/UnionTerritory',

             title='# of Deaths by State sorted by Date'

            )

fig.show()





fig = px.bar(data_frame=df.sort_values('Cured', ascending=False),

             x='Date', y='Cured',

             color='State/UnionTerritory',

             title='# of Cured by State sorted by Date'

            )

fig.show()
dfDate = df.groupby(['Date'])['Cured', 'Confirmed', 'Deaths'].sum().reset_index()



dfDate['NewCasesRoi'] = dfDate.Confirmed.pct_change() * 100 

dfDate['NewDeathsRoi'] = dfDate.Deaths.pct_change() * 100 

dfDate['NewCuredRoi'] = dfDate.Cured.pct_change() * 100 



# Rate of increase in the same chart for all 3 parameters

# fig1 = px.bar(data_frame=dfDate, x='Date', y='NewCasesRoi', hover_data=['Confirmed'], color_discrete_sequence=['blue'])

# fig2 = px.bar(data_frame=dfDate, x='Date', y='NewDeathsRoi', hover_data=['Deaths'], color_discrete_sequence=['red'])

# fig3 = px.bar(data_frame=dfDate, x='Date', y='NewCuredRoi', hover_data=['Cured'], color_discrete_sequence=['green'])

# # # fig1.add_trace(px.line(data_frame=dfDate, x='Date', y='NewDeathsRoi', hover_data=['Deaths']))

# fig1.add_trace(fig2.data[0])

# fig1.add_trace(fig3.data[0])

# fig1.show()



dfDate = dfDate.melt(id_vars=['Date', 'Deaths', 'Cured', 'Confirmed'])

fig = px.bar(data_frame=dfDate, x='Date', y='value', facet_row='variable', color='variable')

fig.update_yaxes(matches=None)

# fig.for_each_trace(lambda t: t.update(name=t.name.split("=")[1]))

fig.show()
StateWiseDf = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')

StateWiseDf.head(5)
from wordcloud import WordCloud

words = ','.join(StateWiseDf.notes.dropna().astype(str).values)

words = words.replace('awaited', '').replace('Awaited', '').replace('Details', '').replace('Travelled', '')



# print(words)

import matplotlib.pyplot as plt

wordcloud = WordCloud(width = 1600, height = 800, 

                background_color ='white', 

                stopwords = ',', 

                min_font_size = 10

            ).generate(words)

plt.figure(figsize = (20, 10), facecolor = None) 

plt.imshow(wordcloud)