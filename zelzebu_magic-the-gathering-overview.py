#Packages

import pandas as pd

import numpy as np

from numpy.random import random

from math import ceil

from pandas.compat import StringIO

from pandas.io.common import urlopen

from IPython.display import display, display_pretty, Javascript, HTML

from matplotlib.path import Path

from matplotlib.spines import Spine

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

from plotly.tools import FigureFactory as FF

from plotly import tools

import plotly

import plotly.plotly as py

import plotly.graph_objs as go

import seaborn as sns





init_notebook_mode()



# Aux variables

colorIdentity_map = {'B': 'Black', 'G': 'Green', 'R': 'Red', 'U': 'Blue', 'W': 'White'}

keeps = ['name', 'colorIdentity', 'colors', 'type', 'types', 'cmc', 'power', 'toughness', 'legalities']
raw = pd.read_json('../input/AllSets-x.json')
# Data fusion

mtg = []

for col in raw.columns.values:

    release = pd.DataFrame(raw[col]['cards'])

    release = release.loc[:, keeps]

    release['releaseName'] = raw[col]['name']

    release['releaseDate'] = raw[col]['releaseDate']

    mtg.append(release)

mtg = pd.concat(mtg)



del release, raw



# Combine colorIdentity and colors

mtg.loc[(mtg.colors.isnull()) & (mtg.colorIdentity.notnull()), 'colors'] = mtg.loc[(mtg.colors.isnull()) & (mtg.colorIdentity.notnull()), 'colorIdentity'].apply(lambda x: [colorIdentity_map[i] for i in x])

mtg['colorsCount'] = 0

mtg.loc[mtg.colors.notnull(), 'colorsCount'] = mtg.colors[mtg.colors.notnull()].apply(len)

mtg.loc[mtg.colors.isnull(), 'colors'] = ['Colorless']

mtg['colorsStr'] = mtg.colors.apply(lambda x: ''.join(x))



# Include colorless and multi-color.

mtg['manaColors'] = mtg['colorsStr']

mtg.loc[mtg.colorsCount>1, 'manaColors'] = 'Multi'



# Set Date type for the releaseDate

format = '%Y-%m-%d'

mtg['releaseDate'] = pd.to_datetime(mtg['releaseDate'], format=format)
# Remove promo cards that aren't used in normal play

mtg_nulls = mtg.loc[mtg.legalities.isnull()]

mtg = mtg.loc[~mtg.legalities.isnull()]



# Remove cards that are banned in any game type

mtg = mtg.loc[mtg.legalities.apply(lambda x: sum(['Banned' in i.values() for i in x])) == 0]

mtg = pd.concat([mtg, mtg_nulls])

mtg.drop('legalities', axis=1, inplace=True)

del mtg_nulls



# Remove tokens without types

mtg = mtg.loc[~mtg.types.apply(lambda x: isinstance(x, float))]



# Transform types to str

mtg.loc[mtg.types.notnull(), 'typesCount'] = mtg.types[mtg.types.notnull()].apply(len)

mtg['typesStr'] = mtg.types.apply(lambda x: ''.join(x))



# Power and toughness that depends on board state or mana cannot be resolved

mtg[['power', 'toughness']] = mtg[['power', 'toughness']].apply(lambda x: pd.to_numeric(x, errors='coerce'))



# Include colorless and multi-color.

mtg['manaColors'] = mtg['colorsStr']

mtg.loc[mtg.colorsCount>1, 'manaColors'] = 'Multi'



# Remove 'Gleemax' and other cards with more than 90 cmc 

mtg = mtg[(mtg.cmc < 90) | (mtg.cmc.isnull())]



# Remove 'Big Furry Monster' and other cards with more than 90 of power and toughness

mtg = mtg[(mtg.power < 90) | (mtg.typesStr != 'Creature')]

mtg = mtg[(mtg.toughness < 90) | (mtg.typesStr != 'Creature')]



# Remove 'Spinal Parasite' and other cards whose power and toughness depends on the number of lands used to cast it

mtg = mtg[(mtg.power > 0) | (mtg.typesStr != 'Creature')]

mtg = mtg[(mtg.toughness > 0) | (mtg.typesStr != 'Creature')]

          

# Remove the duplicated cards

duplicated = mtg[mtg.duplicated(['name'])]

mtg = mtg.drop_duplicates(['name'], keep='first')



# Recode the card type 'Eaturecray' (Atinlay Igpay), which means 'Creature' on Pig-latin

mtg['typesStr'] = mtg['typesStr'].replace('Eaturecray', 'Creature')



cards_recoded_absolutes=(len(mtg[mtg.typesStr=='Vanguard']) + len(mtg[mtg.typesStr=='Scheme']) + len(mtg[mtg.typesStr=='Plane']) + len(mtg[mtg.typesStr=='Phenomenon']) + len(mtg[mtg.typesStr=='Conspiracy']))

cards_recoded_relatives=str(round((((float(len(mtg[mtg.typesStr=='Vanguard']) + len(mtg[mtg.typesStr=='Scheme']) + len(mtg[mtg.typesStr=='Plane']) + len(mtg[mtg.typesStr=='Phenomenon']) + len(mtg[mtg.typesStr=='Conspiracy']))) / float(len(mtg))) * 100), 2))+'%'



# Recode some special card types to 'Other types'

mtg = mtg.replace(['Vanguard', 'Scheme', 'Plane', 'Phenomenon', 'Conspiracy'], 'Other types')



# Transform the multi-choice variable 'types' to a 7-item dichotomized variable

mtg.loc[mtg.types.notnull(), 'typesCount'] = mtg.types[mtg.types.notnull()].apply(len)

mono_types = mtg[mtg.typesCount==1]

mono_types = np.sort(mono_types.typesStr.unique()).tolist()

for types in mono_types:

    mtg[types] = mtg.types.apply(lambda x: types in x)

    

#Transform the multi-choice variable 'colors' to a 5-item dichotomized variable

mono_colors = np.sort(mtg.colorsStr[mtg.colorsCount==1].unique()).tolist()

for color in mono_colors:

    mtg[color] = mtg.colors.apply(lambda x: color in x)
# Get the data

cards_over_time = pd.pivot_table(mtg, values='name',index='releaseDate', aggfunc=len)

cards_over_time.fillna(0, inplace=True)

cards_over_time = cards_over_time.sort_index()



#Create a trace

trace = go.Scatter(x=cards_over_time.index,

                   y=cards_over_time.values)



# Create the range slider

data = [trace]

layout = dict(

    title="Number of new (unique) cards over time",

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label='1m',

                     step='month',

                     stepmode='backward'),

                dict(count=6,

                     label='6m',

                     step='month',

                     stepmode='backward'),

                dict(count=1,

                    label='YTD',

                    step='year',

                    stepmode='todate'),

                dict(count=1,

                    label='1y',

                    step='year',

                    stepmode='backward'),

                dict(step='all')

            ])

        ),

        rangeslider=dict(),

        type='date'

    )

)



# Plot the data

fig = dict(data=data, layout=layout)

plotly.offline.iplot(fig)
# Create the list

total_color_freqs=[]



# Get the data for the colors

for i in mono_colors:

    total_color_freqs.append(str(round(float(len(mtg[mtg.colorsStr==i])/len(mtg)*100), 2))+'%')



# Get the data for the colorless cards

total_color_freqs.append(str(round(float(len(mtg[mtg.colors=='Colorless'])/len(mtg)*100), 2))+'%')



# Get the data for the multicolor cards

total_color_freqs.append(str(round(float(len(mtg[mtg.manaColors=='Multi'])/len(mtg)*100), 2))+'%')



#Tidy the data

total_color_freqs = pd.DataFrame(total_color_freqs)

total_color_freqs=total_color_freqs.transpose()

total_color_freqs.columns=['Black', 'Blue', 'Green', 'Red', 'White', 'Colorless', 'Multicolor']



total_color_freqs
#Create a dataframe with one column for each number of colors

freqs_mono=['monocolor',str(len(mtg[mtg.colorsCount==1]))]

freqs_bi=['bicolor',str(len(mtg[mtg.colorsCount==2]))]

freqs_tri=['tricolor',str(len(mtg[mtg.colorsCount==3]))]

freqs_cuatri=['tetracolor',str(len(mtg[mtg.colorsCount==4]))]

freqs=[freqs_mono, freqs_bi, freqs_tri, freqs_cuatri]



#Get the data for each number of colors

count=0

for a in range(1,5):   

    for i in mono_colors:      

        freqs_raw = pd.value_counts(mtg[i][mtg.colorsCount==a].values.flatten())

        freqs_True=freqs_raw[True]

        freqs[count].append(str(round(float(freqs_True*100)/sum(freqs_raw),2))+'%')

    count=count+1



#Create the dataframe

color_freqs = pd.DataFrame(freqs)

color_freqs.columns=['How many colors?', 'Base size', 'Black', 'Blue', 'Green', 'Red', 'White']

color_freqs=color_freqs.set_index('How many colors?')

del color_freqs.index.name



color_freqs
colors = np.sort(mtg.manaColors.unique()).tolist()

plotly_colors = ['rgb(100,100,100)', 'rgb(70,160,240)', 'rgb(175, 175, 175)', 'rgb(100,200,25)', 'rgb(150, 100, 150)', 'rgb(250,70,25)', 'rgb(225,225,175)']



# get counts

piv = pd.pivot_table(mtg, values='name', index='manaColors', columns='releaseDate', aggfunc=len)

piv.fillna(0, inplace=True)



traces = [go.Scatter(

    x = piv.columns.tolist(),

    y = piv[piv.index==color].iloc[0].tolist(),

    mode = 'lines',

    line = dict(color=plotly_colors[i]),

    connectgaps = True,

    name = color

) for i, color in enumerate(colors)]



layout = dict(

    title='Number of new (unique) cards by type over time',

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label='1m',

                     step='month',

                     stepmode='backward'),

                dict(count=6,

                     label='6m',

                     step='month',

                     stepmode='backward'),

                dict(count=1,

                    label='YTD',

                    step='year',

                    stepmode='todate'),

                dict(count=1,

                    label='1y',

                    step='year',

                    stepmode='backward'),

                dict(step='all')

            ])

        ),

        rangeslider=dict(),

        type='date'

    )

)



fig = go.Figure(data=traces, layout=layout)

iplot(fig, filename='Colors Over Time')
mtg['typesCount'] = 1

mtg.loc[mtg.types.notnull(), 'typesCount'] = mtg.types[mtg.types.notnull()].apply(len)

mtg.loc[mtg.typesCount>1, 'typesStr'] = 'Multi'

types_piv = pd.pivot_table(mtg, values='name',index='manaColors', columns='typesStr', aggfunc=len)

with sns.axes_style("white"):

    ax = sns.heatmap(types_piv)
plotly_colors=['rgb(225,225,175)', 'rgb(70,160,240)', 'rgb(100,100,100)', 'rgb(250,70,25)', 'rgb(100,200,25)', 'rgb(175, 175, 175)', 'rgb(150, 100, 150)']



traces=[]



for i in range(0,len(mtg['manaColors'].unique())):

    traces.append(go.Box(

        name=mtg['manaColors'].unique()[i],

        y=mtg.cmc[mtg['manaColors']==mtg.manaColors.unique()[i]],

        x=mtg['manaColors'].unique()[i],

        fillcolor=plotly_colors[i],

        boxmean=True,

        marker=dict(

            size=2,

            color='black',

        ),

        line=dict(width=1),

    ))



layout = go.Layout(

    yaxis=dict(

        title='Converted Mana Cost',

        zeroline=False

    ),

    showlegend=False

    )



fig = go.Figure(data=traces, layout=layout)

plotly.offline.iplot(fig)
traces=[]



for i in range(0,len(mtg['typesStr'].unique())):

    traces.append(go.Box(

        name=mtg['typesStr'].unique()[i],

        y=mtg.cmc[mtg['typesStr']==mtg.typesStr.unique()[i]],

        x=mtg['typesStr'].unique()[i],

        boxmean=True,

        fillcolor='rgb(70,160,240)',

        marker=dict(

            size=2,

            color='black',

        ),

        line=dict(width=1),

    ))



layout = go.Layout(

    yaxis=dict(

        title='Converted Mana Cost',

        zeroline=False

    ),

    showlegend=False

    )



fig = go.Figure(data=traces, layout=layout)

plotly.offline.iplot(fig)
#First I'm going to isolate the creature cards

creatures = mtg.loc[

    (mtg.type.str.contains('Creature', case=False))    

    & (mtg.cmc.notnull())

    & (mtg.power.notnull())

    & (mtg.toughness.notnull())

    & (0 <= mtg.cmc) & (mtg.cmc < 16)

    & (0 <= mtg.power) & (mtg.power < 16)

    & (0 <= mtg.toughness) & (mtg.toughness < 16)    

    , ['name', 'cmc', 'power', 'toughness', 'releaseName', 'releaseDate', 'manaColors']

]



creatures['cmc'] = round(creatures['cmc'])

creatures['power'] = round(creatures['power'])

creatures['toughness'] = round(creatures['toughness'])



#Count the frequencies of the power points

from collections import Counter

c = Counter(creatures.power)



#set the colors

colors = np.sort(mtg.manaColors.unique()).tolist()

plotly_colors = ['rgb(100,100,100)', 'rgb(70,160,240)', 'rgb(175, 175, 175)', 'rgb(100,200,25)', 'rgb(150, 100, 150)', 'rgb(250,70,25)', 'rgb(225,225,175)']



#create the traces

traces=[go.Scatter(

    x = list(c.keys()),

    y = list(Counter(creatures.power[creatures.manaColors == color]).values()),

    mode = 'lines',

    line = dict(color=plotly_colors[i]),

    connectgaps = True,

    name = color) for i, color in enumerate(colors)]



layout = go.Layout(

    xaxis=dict(

        title='Power',

    ))



#plot the results

fig = go.Figure(data=traces, layout=layout)

plotly.offline.iplot(fig)
from collections import Counter

c = Counter(creatures.toughness)



colors = np.sort(mtg.manaColors.unique()).tolist()

plotly_colors = ['rgb(100,100,100)', 'rgb(70,160,240)', 'rgb(175, 175, 175)', 'rgb(100,200,25)', 'rgb(150, 100, 150)', 'rgb(250,70,25)', 'rgb(225,225,175)']



traces=[go.Scatter(

    x = list(c.keys()),

    y = list(Counter(creatures.toughness[creatures.manaColors == color]).values()),

    mode = 'lines',

    line = dict(color=plotly_colors[i]),

    connectgaps = True,

    name = color) for i, color in enumerate(colors)]



layout = go.Layout(

    xaxis=dict(

        title='Toughness',

    ))



fig = go.Figure(data=traces, layout=layout)

plotly.offline.iplot(fig)
colors = mtg.manaColors.unique().tolist()

plotly_colors = ['rgb(225,225,175)', 'rgb(70,160,240)', 'rgb(100,100,100)', 'rgb(250,70,25)', 'rgb(100,200,25)', 'rgb(175, 175, 175)' , 'rgb(150, 100, 150)']



traces=[]



for i in range(0,len(colors)):

    cards=creatures[creatures.manaColors==colors[i]]

    traces.append(go.Scatter(

        x=cards.power,

        y=cards.toughness,

        mode='markers',

        name = colors[i],

        marker = dict(

            size = 10,

            opacity= 0.3,

            color = plotly_colors[i],

            )

        )

    )

    

traces.append(go.Scatter(

    x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],

    y=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],

    mode='lines',

    name='Perfect proportion',

    line = dict(

        color = ('rgb(102, 102, 102)'),

        width = 2,

        dash = 'dot'),

    ))



layout= go.Layout(

    title= 'Battle Potential',

    hovermode= 'closest',

    xaxis= dict(

        title= 'Power',

        ticklen= 5,

        zeroline= False,

        gridwidth= 1,

    ),

    yaxis=dict(

        title= 'Toughness',

        ticklen= 5,

        gridwidth= 1,

    ),

    annotations=[

        dict(

            x=14,

            y=4,

            xref='x',

            yref='y',

            showarrow=False,

            text='Attacking creatures'),

        dict(

            x=4,

            y=14,

            xref='x',

            yref='y',

            showarrow=False,

            text='Defending creatures',

        ),

    ],

    updatemenus=list([

        dict(

            x=-0.05,

            y=1,

            yanchor='top',

            buttons=list([

                dict(

                    args=['visible', [True, True, True, True, True, True, True, True]],

                    label='All',

                    method='restyle'

                ),

                dict(

                    args=['visible', [True, False, False, False, False, False, False, True]],

                    label='White',

                    method='restyle'

                ),

                dict(

                    args=['visible', [False, True, False, False, False, False, False, True]],

                    label='Blue',

                    method='restyle'

                ),

                dict(

                    args=['visible', [False, False, True, False, False, False, False, True]],

                    label='Black',

                    method='restyle'

                ),

                dict(

                    args=['visible', [False, False, False, True, False, False, False, True]],

                    label='Red',

                    method='restyle'

                ),

                dict(

                    args=['visible', [False, False, False, False, True, False, False, True]],

                    label='Green',

                    method='restyle'

                ),

                dict(

                    args=['visible', [False, False, False, False, False, True, False, True]],

                    label='Colorless',

                    method='restyle'

                ),

                dict(

                    args=['visible', [False, False, False, False, False, False, True, True]],

                    label='Multicolor',

                    method='restyle'

                ),

            ]),

        ),

    ]),

)



fig= go.Figure(data=traces, layout=layout)

plotly.offline.iplot(fig)