import numpy as np 

import pandas as pd 

import datetime

import plotly.graph_objects as go

from ipywidgets import widgets
df = pd.read_csv("/kaggle/input/golden-globe-awards/golden_globe_awards.csv")
df.head()
# creating a column comprised of year or award, nominee, and category with html formatting.

df['list_of_awards'] = "<br>"+df.year_award.apply(str) + ": "  + "(" + df.nominee + ") "+ df.category

df.head()
# count number of wins and losses for each film

i = pd.crosstab(index=[df['win'], df['film']],columns=[df['win']]).reset_index()



# put all wins and loss details into a single column

i['details'] = df.groupby(['win','film'])['list_of_awards'].apply(list).values



# change the details into a list. Getting rid of any duplicate values won't work for lists as they aren't hashable in a dataframe.

i.details = i.details.apply(lambda x: str(x).strip('[]'))



# split details into two columns (winners/losers)

i['details_win'] = i.details

i['details_lost'] = i.details

i.loc[i.win, 'details_lost'] = ""

i.loc[~i.win, 'details_win'] = ""



total_wins_losses = i.groupby('film')[[False,True]].sum().reset_index()

total_wins_losses['details_win'] = i.groupby('film')['details_win'].max().values

total_wins_losses['details_lost'] = i.groupby('film')['details_lost'].max().values

total_wins_losses['year'] = df.groupby('film')['year_award'].max().values

total_wins_losses['nominations'] = total_wins_losses[False] + total_wins_losses[True]



# split shows and films up based on number of years 'film' has been nominated.

years_nominated = df.groupby(['year_award','film'])['year_film'].count().reset_index()

years_nominated = years_nominated.groupby(['film'])['year_award'].count().reset_index()

films = years_nominated[years_nominated.year_award == 1].film.values

shows = years_nominated[~(years_nominated.year_award == 1)].film.values
# slider to filter dates of 'film'

year = widgets.IntSlider(

    value=total_wins_losses.year.min(),

    min=total_wins_losses.year.min(),

    max=total_wins_losses.year.max(),

    step=1.0,

    description='Year:',

    continuous_update=False

)



# dropdown box

textbox = widgets.Dropdown(

    description='Film or TV Show:',

    value='Films',

    options=['TV Shows','Films']

)

show_or_film = {'TV Shows': shows, 'Films': films}



# data shown when plot is first initialised

df2 = total_wins_losses[(total_wins_losses.film.isin(show_or_film[textbox.value])) & 

                      (total_wins_losses.year >= year.value)].sort_values(by=[True,False], ascending=False)[:15]



# assigning an empty figure widget with two traces

trace1 = go.Bar(x=df2.film[:15], y=df2[True], name='Won', hovertext=df2.details_win)

trace2 = go.Bar(x=df2.film[:15], y=df2[False], name='Lost', hovertext=df2.details_lost)

g = go.FigureWidget(data=[trace1, trace2],

                    layout=go.Layout(

                        title=str(textbox.value +" with most wins since " + str(year.value)),

                        barmode='stack'

                    ))
# updating the data whenever a widget is changed.

def response(change):

    

    df2 = total_wins_losses[(total_wins_losses.film.isin(show_or_film[textbox.value])) & 

                      (total_wins_losses.year >= year.value)].sort_values(by=[True,False], ascending=False)[:15]

    

    with g.batch_update():

        g.data[0].x = df2.film

        g.data[1].x = df2.film

        

        g.data[0].y = df2[True]

        g.data[1].y = df2[False]

        

        g.data[0].hovertext = df2.details_win

        g.data[1].hovertext = df2.details_lost

        

        g.layout.barmode = 'stack'

        g.layout.xaxis.title = textbox.value

        g.layout.title = str(textbox.value +" with most wins since " + str(year.value))



year.observe(response, names="value")

textbox.observe(response, names="value")
container = widgets.HBox([year, textbox])

widgets.VBox([container, g])