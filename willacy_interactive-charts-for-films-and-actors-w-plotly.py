import numpy as np 

import pandas as pd 

import datetime

import plotly.graph_objects as go

from ipywidgets import widgets
df = pd.read_csv("/kaggle/input/screen-actors-guild-awards/screen_actor_guild_awards.csv")

df.head()
df.shape
ensemble = df[df.category.apply(lambda x: True if 'ENSEMBLE' in x or x == 'CAST IN A MOTION PICTURE' else False)]

df = df[~(df.category.apply(lambda x: True if 'ENSEMBLE' in x or x == 'CAST IN A MOTION PICTURE' else False))]

ensemble = ensemble.groupby(['year','category','show'], as_index=False).first()

ensemble['full_name'] = 'ENSEMBLE'

df = pd.concat([df, ensemble], sort=False)
df= df.loc[(df.index != 5756) & (df.index != 5758),:].copy()

df['year'] = df['year'].apply(lambda x: str(x)[:4])

df['year'] = df['year'].astype('float64')
df.year = df.year.astype('int32')
df['description'] = "<br>"+df.year.apply(lambda x: str(x)[:4]) + ": "+ df.full_name.apply(lambda x: "" if pd.isnull(x) else "("+x+") ")+ df.category



df.head()
df['won_desc'] = df[df['won'] == True].groupby(['year'])['description'].apply(list)

df['won_desc'] = df[df['won'] == True].groupby(['year'])['description'].apply(list)



i = pd.crosstab(index=[df['won'], df['show']],columns=[df['won']]).reset_index()

i['won_desc'] = df.groupby(['won','show'])['description'].apply(list).values

i['lost_desc'] = df.groupby(['won','show'])['description'].apply(list).values



i['won_desc'] = i['won_desc'].apply(lambda x: str(x).strip('[]'))

i['lost_desc'] = i['lost_desc'].apply(lambda x: str(x).strip('[]'))



total_wins_losses = i.groupby('show', as_index=False)[[False,True]].sum()

total_wins_losses['won_desc'] = i.groupby('show')['won_desc'].max().values

total_wins_losses['lost_desc'] = i.groupby('show')['lost_desc'].max().values

total_wins_losses['year'] = df.groupby('show')['year'].first().values.astype('int32')

total_wins_losses['nominations'] = total_wins_losses[False] + total_wins_losses[True]



years_nominated = df.groupby(['year','show'])['category'].count().reset_index()

years_nominated = years_nominated.groupby(['show'])['category'].count().reset_index()

films = years_nominated[years_nominated.category == 1].show.values

shows = years_nominated[~(years_nominated.category == 1)].show.values
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

df2 = total_wins_losses[(total_wins_losses.show.isin(show_or_film[textbox.value])) & 

                      (total_wins_losses.year >= year.value)].sort_values(by=[True,False], ascending=False)[:15]



# assigning an empty figure widget with two traces

trace1 = go.Bar(x=df2.show[:15], y=df2[True], name='Won', hovertext=df2.won_desc)

trace2 = go.Bar(x=df2.show[:15], y=df2[False], name='Lost', hovertext=df2.lost_desc)

g = go.FigureWidget(data=[trace1, trace2],

                    layout=go.Layout(

                        title=str(textbox.value +" with most wins since " + str(year.value)),

                        barmode='stack'

                    ))
# updating the data whenever a widget is changed.

def response(change):

    

    df2 = total_wins_losses[(total_wins_losses.show.isin(show_or_film[textbox.value])) & 

                      (total_wins_losses.year >= year.value)].sort_values(by=[True,False], ascending=False)[:15]

    

    with g.batch_update():

        g.data[0].x = df2.show

        g.data[1].x = df2.show

        

        g.data[0].y = df2[True]

        g.data[1].y = df2[False]

        

        g.data[0].hovertext = df2.won_desc

        g.data[1].hovertext = df2.lost_desc

        

        g.layout.barmode = 'stack'

        g.layout.xaxis.title = textbox.value

        g.layout.title = str(textbox.value +" with most wins since " + str(year.value))



year.observe(response, names="value")

textbox.observe(response, names="value")
container = widgets.HBox([year, textbox])

widgets.VBox([container, g])
df_name = pd.read_csv("/kaggle/input/screen-actors-guild-awards/screen_actor_guild_awards.csv")



df_name= df_name.loc[(df_name.year != ' ESQ.') & (df_name.index != 5757) & (df_name.index != 5758),:].copy()

df_name['year'] = df_name['year'].apply(lambda x: str(x)[:4])

df_name['year'] = df_name['year'].astype('float64')



df_name = df_name[~(df_name.full_name.isna())]

df_name['description'] = "<br>"+df_name.year.apply(lambda x: str(x)[:4]) + ": "+ df_name.category

df = df_name



df['won_desc'] = df[df['won'] == True].groupby(['year'])['description'].apply(list)

df['won_desc'] = df[df['won'] == True].groupby(['year'])['description'].apply(list)



i = pd.crosstab(index=[df['won'], df['full_name']],columns=[df['won']]).reset_index()

i['won_desc'] = df.groupby(['won','full_name'])['description'].apply(list).values

i['lost_desc'] = df.groupby(['won','full_name'])['description'].apply(list).values



i['won_desc'] = i['won_desc'].apply(lambda x: str(x).strip('[]'))

i['lost_desc'] = i['lost_desc'].apply(lambda x: str(x).strip('[]'))



total_wins_losses = i.groupby('full_name', as_index=False)[[False,True]].sum()

total_wins_losses['won_desc'] = i.groupby('full_name')['won_desc'].max().values

total_wins_losses['lost_desc'] = i.groupby('full_name')['lost_desc'].max().values

total_wins_losses['year'] = df.groupby('full_name')['year'].first().values.astype('int32')

total_wins_losses['nominations'] = total_wins_losses[False] + total_wins_losses[True]



years_nominated = df.groupby(['year','full_name'])['category'].count().reset_index()

years_nominated = years_nominated.groupby(['full_name'])['category'].count().reset_index()

films = years_nominated[years_nominated.category == 1].full_name.values

shows = years_nominated[~(years_nominated.category == 1)].full_name.values
year = widgets.IntSlider(

    value=total_wins_losses.year.min(),

    min=total_wins_losses.year.min(),

    max=total_wins_losses.year.max(),

    step=1.0,

    description='Year:',

    continuous_update=False

)



# data shown when plot is first initialised

df2 = total_wins_losses[(total_wins_losses.full_name.isin(show_or_film[textbox.value])) & 

                      (total_wins_losses.year >= year.value)].sort_values(by=[True,False], ascending=False)[:15]



# assigning an empty figure widget with two traces

trace1 = go.Bar(x=df2.full_name[:15], y=df2[True], name='Won', hovertext=df2.won_desc)

trace2 = go.Bar(x=df2.full_name[:15], y=df2[False], name='Lost', hovertext=df2.lost_desc)

g = go.FigureWidget(data=[trace1, trace2], 

                    layout=go.Layout(

                        title=str("Actor with most wins since " + str(year.value)),

                        barmode='stack'

                        

                    )).update_yaxes(dtick=1)

# updating the data whenever a widget is changed.

def response(change):

    

    df2 = total_wins_losses[(total_wins_losses.year >= year.value)].sort_values(by=[True,False], ascending=False)[:15]

    

    with g.batch_update():

        g.data[0].x = df2.full_name

        g.data[1].x = df2.full_name

        

        g.data[0].y = df2[True]

        g.data[1].y = df2[False]

        

        g.data[0].hovertext = df2.won_desc

        g.data[1].hovertext = df2.lost_desc

        

        g.layout.barmode = 'stack'

        g.layout.title = str("Actor with most wins since " + str(year.value))



year.observe(response, names="value")

textbox.observe(response, names="value")
container = widgets.HBox([year])

widgets.VBox([container, g])