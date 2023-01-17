import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True) 

import plotly.graph_objs as go
data = pd.read_csv('../input/AviationData.csv', sep=',', header=0, encoding = 'iso-8859-1')



#UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8f in position 4: invalid start byte
data.isnull().sum()
data.head()

data_usage = data.iloc[:,3:13]
data_usage["Event.Date"] = pd.to_datetime(data_usage["Event.Date"])

data_usage.isna().sum()
data_usage.dropna(axis=0,inplace=True)

data_usage.index = data_usage["Event.Date"]

data_usage.drop("Event.Date",axis=1,inplace=True)
data_usage.index = pd.to_datetime(data_usage.index)

data_usage.head()
data_usage["year"] = data_usage.index
data_usage["year"] = data_usage["year"].astype(str)
for i,value in enumerate(data_usage["year"]):

    data_usage["year"][i] = data_usage["year"][i][0:4]
data_usage.year = data_usage.year.astype(int)
data_usage = data_usage.sort_values("year")

newdata = data_usage

newdata["Longitude2"] = data_usage["Longitude"]

newdata = newdata.groupby("Longitude2").first()

newdata = newdata.sort_values("year")
years = [str(each) for each in list(newdata["year"].unique())]  # str unique years

# make list of types

types = ['Substantial', 'Destroyed', 'Minor']

custom_colors = {

    'Substantial': 'rgb(189, 2, 21)',

    'Destroyed': 'rgb(52, 7, 250)',

    'Minor': 'rgb(99, 110, 250)'

}

# make figure

figure = {

    'data': [],

    'layout': {},

    'frames': []

}



figure['layout']['geo'] = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,

              countrywidth=1, 

              landcolor = 'rgb(217, 217, 217)',

              subunitwidth=1,

              showlakes = True,

              lakecolor = 'rgb(255, 255, 255)',

              countrycolor="rgb(5, 5, 5)")

figure['layout']['hovermode'] = 'closest'

figure['layout']['sliders'] = {

    'args': [

        'transition', {

            'duration': 400,

            'easing': 'cubic-in-out'

        }

    ],

    'initialValue': '1982',

    'plotlycommand': 'animate',

    'values': years,

    'visible': True

}

figure['layout']['updatemenus'] = [

    {

        'buttons': [

            {

                'args': [None, {'frame': {'duration': 500, 'redraw': False},

                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],

                'label': 'Play',

                'method': 'animate'

            },

            {

                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',

                'transition': {'duration': 0}}],

                'label': 'Pause',

                'method': 'animate'

            }

        ],

        'direction': 'left',

        'pad': {'r': 10, 't': 87},

        'showactive': False,

        'type': 'buttons',

        'x': 0.1,

        'xanchor': 'right',

        'y': 0,

        'yanchor': 'top'

    }

]



sliders_dict = {

    'active': 0,

    'yanchor': 'top',

    'xanchor': 'left',

    'currentvalue': {

        'font': {'size': 20},

        'prefix': 'Year:',

        'visible': True,

        'xanchor': 'right'

    },

    'transition': {'duration': 300, 'easing': 'cubic-in-out'},

    'pad': {'b': 10, 't': 50},

    'len': 0.9,

    'x': 0.1,

    'y': 0,

    'steps': []

}



# make data

year = 1982

for ty in types:

    dataset_by_year = newdata[newdata["year"] == year]

    dataset_by_year_and_cont = dataset_by_year[dataset_by_year["Aircraft.Damage"] == ty]

    

    data_dict = dict(

    type='scattergeo',

    lon = newdata['Longitude'],

    lat = newdata['Latitude'],

    hoverinfo = 'text',

    text = ty,

    mode = 'markers',

    marker=dict(

        sizemode = 'area',

        sizeref = 1,

        size= 10 ,

        line = dict(width=1,color = "white"),

        color = custom_colors[ty],

        opacity = 0.7),

)

    figure['data'].append(data_dict)

    

# make frames

for year in years:

    frame = {'data': [], 'name': str(year)}

    for ty in types:

        dataset_by_year = newdata[newdata["year"] == int(year)]

        dataset_by_year_and_cont = dataset_by_year[dataset_by_year["Aircraft.Damage"] == ty]



        data_dict = dict(

                type='scattergeo',

                lon = dataset_by_year_and_cont['Longitude'],

                lat = dataset_by_year_and_cont['Latitude'],

                hoverinfo = 'text',

                text = ty,

                mode = 'markers',

                marker=dict(

                    sizemode = 'area',

                    sizeref = 1,

                    size= 10 ,

                    line = dict(width=1,color = "white"),

                    color = custom_colors[ty],

                    opacity = 0.7),

                name = ty

            )

        frame['data'].append(data_dict)



    figure['frames'].append(frame)

    slider_step = {'args': [

        [year],

        {'frame': {'duration': 300, 'redraw': False},

         'mode': 'immediate',

       'transition': {'duration': 300}}

     ],

     'label': year,

     'method': 'animate'}

    sliders_dict['steps'].append(slider_step)





figure["layout"]["autosize"]= True

figure["layout"]["title"] = "Aviation Accident"       



figure['layout']['sliders'] = [sliders_dict]



iplot(figure)