import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cufflinks as cf
import plotly
import plotly.offline as py
import plotly.graph_objs as go
cf.go_offline() # required to use plotly offline (no account required).
py.init_notebook_mode() # graphs charts inline (IPython).

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 60)

usa_states = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming'
}

# import data & create some new columns
df = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
df['year'] = pd.DatetimeIndex(df['date']).year
# Parse the date and set the index
df.date = pd.to_datetime(df.date)
df.set_index('date', inplace=True)
df['n_killed_or_injured'] = df['n_killed'] + df['n_injured']
df['state_code'] = df['state'].apply(lambda x: list(usa_states.keys())[list(usa_states.values()).index(x)] if x != 'District of Columbia' else 'MD')

# cleaning some data
df['participant_gender'] = df['participant_gender'].str.replace('\|\|', '|').str.replace('::', ':')
df['participant_type'] = df['participant_type'].str.replace('\|\|', '|').str.replace('::', ':')
# df.loc[49384, 'participant_gender'] = '0::Female'
df.head(2)
df.describe()
sum_result = df.isna().sum(axis=0).sort_values(ascending=False)
missing_values_columns = sum_result[sum_result > 0]
print('They are %s columns with missing values : \n%s' % (missing_values_columns.count(), [(index, value) for (index, value) in missing_values_columns.iteritems()]))
layout = go.Layout(
    title= 'Incidents count for each years of the dataset',
    height= 400,
    width= 850,
    xaxis= {
        'tickformat': 'd',
    }
)

df['year'].value_counts().iplot(kind='bar', layout=layout)
from_2014_to_2017 = df[(df.year >= 2014) & (df.year <= 2017)]
nb_injured_and_killed_by_state = from_2014_to_2017.groupby('state_code')[['n_injured', 'n_killed', 'n_killed_or_injured']].sum()

colorscale = [[0.0, 'rgb(242,240,247)'], [0.2, 'rgb(218,218,235)'], [0.4, 'rgb(188,189,220)'],
       [0.6, 'rgb(158,154,200)'], [0.8, 'rgb(117,107,177)'], [1.0, 'rgb(84,39,143)']]

nb_injured_and_killed_by_state['hover_text'] = nb_injured_and_killed_by_state.index.map(lambda x: usa_states.get(x)) + '<br>' + \
             'Number of people killed : ' + nb_injured_and_killed_by_state['n_killed'].astype(str) + '<br>' + \
             'Number of people injured : ' + nb_injured_and_killed_by_state['n_injured'].astype(str) + '<br>'

data = [dict(
    type='choropleth',
    colorscale=colorscale,
    autocolorscale=False,
    locations=nb_injured_and_killed_by_state.index,
    z=nb_injured_and_killed_by_state['n_killed_or_injured'].astype(int),
    locationmode='USA-states',
    text=nb_injured_and_killed_by_state['hover_text'],
    marker=dict(
        line=dict(
            color='rgb(255,255,255)',
            width=2
        )),
    colorbar=dict(
        title="Millions USD")
)]

layout = dict(
    title='Victims geographic repartition between 2014 and 2017',
    height=400,
    width=850,
    geo=dict(
        scope='usa',
        projection=dict(type='albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'),
)

figure = dict(data=data, layout=layout)
py.iplot(figure)
# Highest number of dead by gunshot by state
nb_killed_by_state = df.groupby("state")["n_killed"].sum().sort_values(ascending=False)

layout = go.Layout(
    title= 'States with the highest total of people killed by gunshot',
    height= 400,
    width= 850,
    xaxis= {
        'tickformat': 'd',
    },
    margin = dict(l = 100, r = 50, b = 50, t = 50, pad = 4)
)

figure = dict(data=data, layout=layout)

plot = nb_killed_by_state[0:15].sort_values(ascending=True).iplot(kind='barh', layout=layout)
# Number of kills per year:
from_2014_to_2017 = df[(df.year >= 2014) & (df.year <= 2017)]
layout = go.Layout(
    title= 'Death number evolution between 2014 and 2017 in the USA',
    height= 400,
    width= 750,
    xaxis= {
        'tickformat': 'd',
    }
)
nb_killed_by_year = from_2014_to_2017.groupby("year")["n_killed"].sum().iplot(kind='bar', layout=layout)
incidents_by_month = from_2014_to_2017.groupby(pd.Grouper(freq='w'))['n_killed_or_injured'].sum()
layout = go.Layout(
    title= 'Victims of incidents by week',
    height= 450,
    width= 800
)
incidents_by_month.iplot(layout=layout)
# Proportion of killed / injured per year:
layout = go.Layout(
    barmode= 'stack',
    title= 'Gun violence evolution between 2014 and 2017 in the USA',
    height= 400,
    width= 750,
    xaxis= {
        'tickformat': 'd',
    }
)
from_2014_to_2017.groupby("year")[["n_injured", "n_killed"]].sum().iplot(kind='bar', layout=layout)
# States with the biggest evolution of gun violence between 2014 and 2017
# Proportion of killed / injured per year:
layout = go.Layout(
    title= 'States with biggest evolution between 2014 and 2017',
    height= 400,
    width= 750,
    xaxis= {
        'tickformat': 'd',
    }
)
death_evolution_by_states = (df[df['year'] == 2017].groupby('state')['n_killed_or_injured'].sum() - df[df['year'] == 2014].groupby('state')['n_killed_or_injured'].sum()).sort_values(ascending=False)
death_evolution_by_states[0:15].iplot(kind='bar', title='15 first states with biggest increase of victims between 2014 and 2017', layout=layout)
# Mean of age shooters
import re

def extract_right_part(original_str, split_chars):
    splited = str(original_str).split(split_chars)
    if len(splited) > 1:
        return splited[1]
    
def count_male_suspect(row):
    print(row['participant_type'])

limit = 100   
df2 = df.head(limit)['participant_gender'].str.split('\|\|').apply(pd.Series)
df2 = df2.applymap(lambda x: extract_right_part(x, '::'))
# df2.index = df.head(limit).set_index(['participant_type', 'participant_age_group', 'participant_gender']).index
# df2.stack().reset_index(['participant_type', 'participant_age_group', 'participant_gender'])
#df2.apply(count_male_suspect, axis=1)
df2.head(limit)
data = {
    'group_id' : [1, 2], 
    'persons': [
        ['John', 'Anna'], 
        ['Virginia', 'Bob']
    ], 
    'ages': [
        ['18', '20'], 
        ['22', '45']
    ]
}

df1 = pd.DataFrame(data)
print('df1')
print(df1)

s = df1.apply(lambda x: pd.Series(x['persons']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'persons'
df2 = df1.drop('persons', axis=1).join(s)

print('\ndf2')
print (df2)

s2 = df1.apply(lambda x: pd.Series(x['ages']),axis=1).stack().reset_index(level=1, drop=True)
s2.name = 'ages'
df3 = df1.drop('ages', axis=1).join(s2)

print('\ndf3')
print(df3)

df3['ages'] = df2['ages']

print('\ndf3')
print(df3)

def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]
            
df4 = explode(df1, ['persons', 'ages'], None)

print('\ndf4')
print(df4)
# TODO : https://stackoverflow.com/questions/27263805/pandas-when-cell-contents-are-lists-create-a-row-for-each-element-in-the-list/27266225
# TODO : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.stack.html
# Transform "0::18||1::46||2::14||3::47" with [18, 46, 14, 47]
import re
import math

def map_multiple_value_cell_as_list(multi_value_cell):  
    if multi_value_cell and not isinstance(multi_value_cell, list):
        tmp = re.sub('^\d+::', '', str(multi_value_cell))
        coma_separated = re.sub('\|\|\d+::', ',', tmp)
        return coma_separated.split(',') 
    else:
        return multi_value_cell

def map_multiple_value_cell_as_list_and_complete_missing(multi_value_cell, participant_type):    
    if multi_value_cell and not isinstance(multi_value_cell, list) and isinstance(participant_type, str):
        nb_participant = int(participant_type.split('||')[len(participant_type.split('||')) - 1].split('::')[0])
        list_elems = map_multiple_value_cell_as_list(multi_value_cell)
        for x in range(len(list_elems), int(nb_participant) + 1):
            list_elems.append('NA')
        return list_elems
    else:
        return multi_value_cell

limit = 100000

df.head(limit).loc[:, 'participant_gender'] = df.head(limit).apply(lambda x: map_multiple_value_cell_as_list_and_complete_missing(x['participant_gender'], x['participant_type']), axis=1)
df.head(limit).loc[:, 'participant_type'] = df.head(limit).apply(lambda x: map_multiple_value_cell_as_list(x['participant_type']), axis=1)

df.head(limit)[['participant_gender', 'participant_type']]

df2 = explode(df.head(limit), ['participant_gender', 'participant_type'], None)

df2.head(limit)
# TODO
# - percentage of accendital / intentional / unknown
# - people ages responsible of kills
# - USA map with incidents number
# - Data enrichment with more variables (about states)
# participant status (injured, killed, arrested, unharmed)
# month / year with more incidents