import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

from IPython.display import display, HTML

init_notebook_mode(connected=True)
terror_data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1',usecols=[0, 1, 2, 3, 7, 8, 10, 13, 14, 35, 82, 98, 101])

terror_data['nkill'] = terror_data['nkill'].fillna(0).astype(int)

terror_data['nwound'] = terror_data['nwound'].fillna(0).astype(int)

terror_data = terror_data[np.isfinite(terror_data.latitude)]

# terror_data.head()
## Total Casualties in Global Terrorist Attacks

# prepare list

years=[]

for year in terror_data["iyear"].values:

    if year not in years:

        years.append(year)

years.sort()



Fatalities = []

Injuries = []

Counts=[]

for year in years:

    dataset_by_year = terror_data[terror_data['iyear'] == year]

    Count = dataset_by_year.iyear.count()

    Counts.append(Count)

    fatality = dataset_by_year['nkill'].sum()

    Fatalities.append(fatality)

    injury = dataset_by_year['nwound'].sum()

    Injuries.append(injury)



# make figure

trace1 = go.Bar(

    x=years,

    y=Injuries,

    name='Injuries',

    marker=dict(color='#FFD7E9')

)

trace2 = go.Bar(

    x=years,

    y=Fatalities,

    name='Fatalities',

    marker=dict(color='#EB89B5')

)

trace3 = go.Scatter(

    x=years,

    y=Counts,

    name='Events'

)



data = [trace1, trace2, trace3]

layout = go.Layout(

    title='Total Casualties in Global Terrorist Attacks (1970 - 2016)',

    xaxis = dict(title = 'Year'),

    yaxis = dict(title = 'Count'),

    barmode='group',

    bargap=0.2,

    bargroupgap=0.1

)



figure = dict(data = data, layout = layout);

iplot(figure)
## Total Casualties by Weapon Type in Global Terrorist Attacks

# prepare list

weapon_categories=[]

for weapon in terror_data["weaptype1_txt"].values:

    if weapon not in weapon_categories:

        weapon_categories.append(weapon)



count_total = terror_data.weaptype1_txt.count()

weapon_count = []

weapon_injury = []

weapon_fatality = []

for weapon in weapon_categories:

    dataset_by_weap = terror_data[terror_data['weaptype1_txt'] == weapon]

    weap_count = dataset_by_weap.weaptype1_txt.count()

    weapon_count.append(weap_count)

    weap_fatality = dataset_by_weap['nkill'].sum()

    weapon_fatality.append(weap_fatality)

    weap_injury = dataset_by_weap['nwound'].sum()

    weapon_injury.append(weap_injury)



# make figure

trace1 = go.Bar(

    x=weapon_categories,

    y=weapon_injury,

    name='Injuries',

    marker=dict(color='#FFD7E9')

)

trace2 = go.Bar(

    x=weapon_categories,

    y=weapon_fatality,

    name='Fatalities',

    marker=dict(color='#EB89B5')

)

trace3 = go.Scatter(

    x=weapon_categories,

    y=weapon_count,

    name='Events'

)



data = [trace1, trace2, trace3]

layout = go.Layout(

    title='Total Casualties by Weapon Type in Global Terrorist Attacks (1970 - 2016)',

    xaxis = dict(title = 'Weapon Type'),

    yaxis = dict(title = 'Count'),

    barmode='group',

    bargap=0.2,

    bargroupgap=0.1

)



figure = dict(data = data, layout = layout);

iplot(figure)
## Average Casualties by Weapon Type in Global Terrorist Attacks

# prepare list

ave_weapon_injury=[a/b for a,b in zip(weapon_injury,weapon_count)]

ave_weapon_fatality=[a/b for a,b in zip(weapon_fatality,weapon_count)]

ave_weapon_injury=np.round(ave_weapon_injury,2)

ave_weapon_fatality=np.round(ave_weapon_fatality,2)



# make figure

trace1 = go.Bar(

    x=weapon_categories,

    y=ave_weapon_injury,

    name='Injuries',

    marker=dict(color='#FFD7E9')

)

trace2 = go.Bar(

    x=weapon_categories,

    y=ave_weapon_fatality,

    name='Fatalities',

    marker=dict(color='#EB89B5')

)



data = [trace1, trace2]

layout = go.Layout(

    title='Average Casualties by Weapon Type in Global Terrorist Attacks (1970 - 2016)',

    xaxis = dict(title = 'Weapon Type'),

    yaxis = dict(title = 'Count'),

    barmode='group',

    bargap=0.2,

    bargroupgap=0.1

)



figure = dict(data = data, layout = layout);

iplot(figure)
## Casualties by Weapon Type in Global Terrorist Attacks per year (Animation)

# make figure

figure = {

    'data': [],

    'layout': {},

    'frames': []

}



# layout

figure['layout']['xaxis'] = {'range': [0, 5],'title': 'Injuries', 'type': 'log','nticks':6}

figure['layout']['yaxis'] = {'range': [0, 5.2],'title': 'Fatalities', 'type': 'log','nticks':6}

figure['layout']['showlegend'] =False

figure['layout']['title'] = 'Casualties by Weapon Type in Global Terrorist Attacks per year (1970-2016)'

figure['layout']['hovermode'] = 'closest'

figure['layout']['sliders'] = {

    'args': [

        'transition', {

            'duration': 400,

            'easing': 'cubic-in-out'

        }

    ],

    'initialValue': '1970',

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



# data

year = 1970

for weapon in weapon_categories:

    dataset_by_year = terror_data[terror_data['iyear'] == year]

    dataset_by_year_and_weap = dataset_by_year[dataset_by_year['weaptype1_txt'] == weapon]

    weap_count_year = dataset_by_year.weaptype1_txt.count()

    weap_count = dataset_by_year_and_weap.weaptype1_txt.count()

    weap_percent = np.round((weap_count / weap_count_year * 100), 2)

    weap_fatality = dataset_by_year_and_weap['nkill'].sum()

    weap_injury = dataset_by_year_and_weap['nwound'].sum()

    dataset=dataset_by_year_and_weap.drop_duplicates(['weaptype1_txt'])

    dataset.loc[:,'nkill']=weap_fatality 

    dataset.loc[:,'nwound']=weap_injury

    weapon_text = (weapon + ' (' + str(weap_percent) + '%)<br>' + str(weap_count) + ' events, ' + str(weap_injury) + ' injuries, ' + str(weap_fatality) + ' fatalities.')



    data_dict = {

        'x': (list(dataset['nwound'])),

        'y': (list(dataset['nkill'])),

        'mode': 'markers',

        'text': weapon_text,

        'marker': {

            'sizemode': 'area',

            'sizeref': 200000,

            'size': np.sqrt(float(weap_count))

        },

        'name': weapon

    }

    figure['data'].append(data_dict)



# frames

for year in years:

    frame = {'data': [], 'name': str(year)}

    for weapon in weapon_categories:

        dataset_by_year = terror_data[terror_data['iyear'] == year]

        dataset_by_year_and_weap = dataset_by_year[dataset_by_year['weaptype1_txt'] == weapon]

        weap_count_year = dataset_by_year.weaptype1_txt.count()

        weap_count = dataset_by_year_and_weap.weaptype1_txt.count()

        weap_percent = np.round((weap_count / weap_count_year * 100), 2)

        weap_fatality = dataset_by_year_and_weap['nkill'].sum()

        weap_injury = dataset_by_year_and_weap['nwound'].sum()

        dataset=dataset_by_year_and_weap.drop_duplicates(['weaptype1_txt'])

        dataset.loc[:,'nkill']=weap_fatality 

        dataset.loc[:,'nwound']=weap_injury

        weapon_text = (weapon + ' (' + str(weap_percent) + '%)<br>' + str(weap_count) + ' events, ' + str(weap_injury) + ' injuries, ' + str(weap_fatality) + ' fatalities.')

    

        data_dict = {

            'x': (list(dataset['nwound'])),

            'y': (list(dataset['nkill'])),

            'mode': 'markers',

            'text': weapon_text,

            'marker': {

                'sizemode': 'area',

                'sizeref': 200000,

                'size': np.sqrt(float(weap_count))

            },

            'name': weapon

        }

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



figure['layout']['sliders'] = [sliders_dict]



iplot(figure)