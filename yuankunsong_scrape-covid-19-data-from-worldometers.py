import pandas as pd
import numpy as np
import requests

import urllib.request

from bs4 import BeautifulSoup

# plotly
import plotly as py
from plotly.offline import init_notebook_mode, iplot, plot

import plotly.express as px
import plotly.graph_objs as go
init_notebook_mode(connected=True)

# matplotlib
import matplotlib.pyplot as plt
# ulr of website
URL = 'https://www.worldometers.info/coronavirus/'

# get the page
page = requests.get(url=URL)

# soup
soup = BeautifulSoup(page.text)

# choose today or yesterday's table, 1 = today, 0 = yesterday
today = 0
if today == 1:
    # today's table
    full_table = soup.find('table', {'id':'main_table_countries_today'})
else:
    # sometimes today's table are not updated yet and contain many blanks, then use yesterday's table
    full_table = soup.find('table', {'id':'main_table_countries_yesterday'})

# generate the dataframe from HTML

def generate_dataframe_world(html_table):
    
    # generate columns for the df

    A = [] # country
    B = [] # confirmed
    C = [] # new case
    D = [] # death
    E = [] # recover

    for row in html_table.find_all('tr'):
        cell = row.find_all('td')

        if (len(cell) == 19):
            A.append(cell[1].find(text = True))
            B.append(cell[2].find(text = True))
            C.append(cell[3].find(text = True))
            D.append(cell[4].find(text = True))
            E.append(cell[6].find(text = True))

    # put togather in to df

    df = pd.DataFrame()
    df['Country'] = A
    df['Confirmed'] = B
    df['New_cases'] = C
    df['Deaths'] = D
    df['Recovered'] = E

    # remove some unnecessary rows
    df.set_index('Country', inplace=True)
    df.drop(['\n','World','Total:'], inplace=True)
    df.reset_index(inplace=True)
    df.fillna('0', inplace=True)

    # remove symbols
    df['Confirmed'] = df.Confirmed.str.replace(',','')
    df['New_cases'] = df.New_cases.str.replace(',','')
    df['New_cases'] = df.New_cases.str.replace('+','')
    df['Deaths'] = df.Deaths.str.replace(',','')
    df['Recovered'] = df.Recovered.str.replace(',','')
    df['Recovered'] = df.Recovered.str.replace('N/A','0')
    df.replace(r'^\s*$', 0, regex=True, inplace=True) # replace blank spaces in cells with 0

    # set data type
    df = df.astype({'Country':str, 'Confirmed':int, 'New_cases':int, 'Deaths':int, 'Recovered':int})
    
    return df
df = generate_dataframe_world(full_table)
df
# pie chart

fig = px.pie(df, values='Confirmed', names='Country', title='Percentage of Confirmed')
fig.update_traces(textposition='inside', textinfo='label+percent')
fig.update_layout(title_x=0.5)

fig.show()
# top 20 countries for 'Confirmed'
top20 = df.copy()
top20 = top20.sort_values(by='Confirmed',ascending=True).tail(20).reset_index()

# bar chart
color_map = {'Confirmed':'dodgerblue', 'Deaths':'red', 'Recovered':'limegreen'}
fig = px.bar(top20, x=['Confirmed','Recovered','Deaths'], y='Country', title='Confirmed, Recovered and Deaths',
             labels={'value': 'Number of confirmed cases'},
             barmode='overlay', 
             opacity=1,
            color_discrete_map=color_map)

fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.update_layout(template='ggplot2')
fig.show()
# top 20 countries for 'New cases'
top20 = df.copy()
top20 = top20.sort_values(by='New_cases',ascending=True).tail(20).reset_index()

# bar chart
fig = px.bar(top20, x=['New_cases'], y='Country', title='Today\'s New Cases',
             labels={'value': 'Number of new cases'},
             barmode='overlay', 
             opacity=1,
            color='Country')

fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.update_layout(template='ggplot2')
fig.show()
# confirmed cases right now

fig = px.choropleth(df, locations='Country',
                    color='New_cases',
                    locationmode='country names', 
                    hover_name='Country', 
                    color_continuous_scale=px.colors.sequential.YlOrRd )
fig.update_layout(
    title='New Cases In Each Country',title_x=0.5)

fig.show()
# ulr of website
URL = 'https://www.worldometers.info/coronavirus/country/us/'

# get the page
page = requests.get(url=URL)

# soup
soup = BeautifulSoup(page.text)

# find the table
if today == 1:
    # today's table
    us_table = soup.find('table', {'id':'usa_table_countries_today'})
else:
    # sometimes today's table are not updated yet and contain many blanks, then use yesterday's table
    us_table = soup.find('table', {'id':'usa_table_countries_yesterday'})
# generate the dataframe from HTML

def generate_dataframe_us(html_table):
    
    # generate columns for the df

    A = [] # state
    B = [] # confirmed
    C = [] # new case
    D = [] # death


    for row in html_table.find_all('tr'):
        cell = row.find_all('td')

        if (len(cell) == 11):
            
            # HTML format issue, source code contain '\n' and state name is below it
            if cell[0].find(text = True) == '\n':
                A.append(cell[0].find_next().find(text = True))
            else:
                A.append(cell[0].find(text = True))
                
            B.append(cell[1].find(text = True))
            C.append(cell[2].find(text = True))
            D.append(cell[3].find(text = True))
            

    # put togather in to df

    df = pd.DataFrame()
    df['State'] = A
    df['Confirmed'] = B
    df['New_cases'] = C
    df['Deaths'] = D


    # remove some unnecessary rows
    col_drop = ['USA Total','Total:']
    
    df.set_index('State', inplace=True)
    df.drop(col_drop, inplace=True)
    df.reset_index(inplace=True)
    df.fillna('0', inplace=True)

    # remove symbols
    df['Confirmed'] = df.Confirmed.str.replace(',','')
    df['New_cases'] = df.New_cases.str.replace(',','')
    df['New_cases'] = df.New_cases.str.replace('+','')
    df['Deaths'] = df.Deaths.str.replace(',','')

    df.replace(r'^\s*$', 0, regex=True, inplace=True) # replace blank spaces in cells with 0

    # set data type
    df = df.astype({'State':str, 'Confirmed':int, 'New_cases':int, 'Deaths':int})
    
    return df.iloc[0:51,:] 
df = generate_dataframe_us(us_table)
df
# add new column for state abbr, abbr are used for map plot
# add new column 'death rate'

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

df['State_abbr'] = df['State']
df['State_abbr'] = df['State_abbr'].replace(us_state_abbrev)
df['Death_rate'] = df['Deaths'] / df['Confirmed'] 


# Confirmed percentage in US

fig = px.pie(df, values='Confirmed', names='State', title='Confirmed cases in the US')
fig.update_traces(textposition='inside', textinfo='label+percent')
fig.update_layout(title_x=0.5)

fig.show()
# New cases in the U.S. during the past day

fig = px.choropleth(df, locations='State_abbr',
                    color='New_cases',
                    locationmode='USA-states',
                    scope='usa',
                    hover_name='State', 
                    color_continuous_scale=px.colors.sequential.YlOrRd )

fig.update_traces(text='New_cases')

fig.update_layout(
    title='New cases in the U.S. during the past day', title_x=0.5)

fig.show()
# top 20 state for 'New cases'
top20 = df.copy()
top20 = top20.sort_values(by='New_cases',ascending=True).tail(20).reset_index()

# bar chart
fig = px.bar(top20, x=['New_cases'], y='State', title='Today\'s New Cases',
             labels={'value': 'Number of new cases'},
             barmode='overlay', 
             opacity=1,
            color='State')

fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.update_layout(template='ggplot2')
fig.show()