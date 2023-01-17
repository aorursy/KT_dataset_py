# Import libraries and define useful functions



# import pandas as pd

from urllib.request import urlopen

import json

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as po

import datetime

import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression



# put plotly into notebook mode for graphics rendering

po.init_notebook_mode(connected=False)



# function to compute time to double for exponential growth

# This won't be valid when the growth curve starts to look logistic

def doubling_time(x,y):

    # pass in two numpy arrays 

    # The y values are assumed to be already in log

    reg = LinearRegression().fit(x,y)

    dbl_time = int(np.log(2)/reg.coef_.item())

    

    return dbl_time



# number of days to do moving average over

ma_days = 3



# read in the county wide data

df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv')

df.head()
# Figure out what the most current date is inside the dataset and get values for that date

latest_date = df.iloc[-1,0]

is_nj = df['state']=='New Jersey'

is_latest = df['date']==latest_date

cases_nj_now = df[is_nj & is_latest].copy()

# Get rid of rows with a county that is Unknown (I think these are cases that can't be tied to a county)

nan_rows = cases_nj_now[cases_nj_now['county']=='Unknown'].index

cases_nj_now.drop(nan_rows,inplace=True)



cases_nj_hist = df[is_nj].copy()

cases_nj_hist.drop(cases_nj_now[cases_nj_now['county']=='Unknown'].index, inplace=True)

earliest_nj_date = cases_nj_hist.iloc[0,0]
# Get fips county data to identify counties and use for map

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:

    counties = json.load(response)

# Use plotly's choropleth map

fig = px.choropleth_mapbox(cases_nj_now, geojson=counties, locations='fips', color='cases',

                           color_continuous_scale="Reds",

                           mapbox_style="carto-positron",

                           opacity=0.5,

                           labels={'cases':'cases'},

                           zoom=6, center = {"lat": 40.0583, "lon": -74.4057}

                          )



fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
# get population data for per capita numbers

pop_all = pd.read_csv('https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv',encoding='mac-roman')

nj = pop_all['STNAME']=='New Jersey'

counties = pop_all['COUNTY']!=0

nj_pops = pop_all[nj & counties].copy()

fips = nj_pops['STATE'].astype(str) + nj_pops['COUNTY'].apply(lambda x: str(x).zfill(3)).astype(str)

nj_pops['fips'] = fips.astype(float)

orig_cols = list(nj_pops.columns)

keep_cols = ['fips', 'POPESTIMATE2019']

res = [ele for ele in orig_cols]

for a in orig_cols:

  if a in keep_cols:

    res.remove(a)

drop_cols = res

nj_pops = nj_pops.drop(drop_cols,axis=1)

cases_nj_hist = cases_nj_hist.set_index('fips').join(nj_pops.set_index('fips'))

cases_nj_hist['cases_per_100k']=cases_nj_hist['cases']*1e5/cases_nj_hist['POPESTIMATE2019']

cases_nj_hist['deaths_per_100k']=cases_nj_hist['deaths']*1e5/cases_nj_hist['POPESTIMATE2019']

cases_nj_hist.head()



# Add a new trace to the plot for each county

container = []

for index, row in cases_nj_now.iterrows():

    county=row['county']

    county_data=cases_nj_hist[cases_nj_hist['county']==county]

    days = (pd.to_datetime(county_data['date'],format='%Y-%m-%d')

         -pd.to_datetime(earliest_nj_date,format='%Y-%m-%d')).dt.days.to_list()

    cases = county_data['cases_per_100k']

    legend_name = county

    trace=go.Scatter(

        x=days,

        y=cases,

        name=legend_name)

    container.append(trace)

    

fig = go.Figure(container)

ttext = "NJ Cases per 100k of population as of "+datetime.datetime.today().strftime('%m/%d/%Y')+' by county'

fig.update_layout(xaxis_type="linear", yaxis_type="log", title=ttext, xaxis_title='days')

fig.show()
# Now we plot multiple traces for deaths per county

container = []

for index, row in cases_nj_now.iterrows():

    county=row['county']

    county_data=cases_nj_hist[cases_nj_hist['county']==county]

    days = (pd.to_datetime(county_data['date'],format='%Y-%m-%d')

         -pd.to_datetime(earliest_nj_date,format='%Y-%m-%d')).dt.days.to_list()

    deaths = county_data['deaths_per_100k']

    legend_name = county

    trace=go.Scatter(

        x=days,

        y=deaths,

        name=legend_name)

    container.append(trace)

    

fig = go.Figure(container)

ttext = "NJ Deaths per 100k population as of "+datetime.datetime.today().strftime('%m/%d/%Y')+' by county'

fig.update_layout(xaxis_type="linear", yaxis_type="log", title=ttext, xaxis_title='days')

fig.show()
# Pull in the state by state data (no county data) versus time from the NYT

df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv')

df.head()
# Plot versus days with respect to the first reported case

nj_timeline = df[df['state']=='New Jersey'].copy()

days = (pd.to_datetime(nj_timeline['date'],format='%Y-%m-%d')

         -pd.to_datetime(earliest_nj_date,format='%Y-%m-%d')).dt.days.to_list()

cases = nj_timeline['cases']

deaths = nj_timeline['deaths']

container = []

trace=go.Scatter(

    x=nj_timeline['date'],

    y=cases,

    name='cases')

container.append(trace)

trace=go.Scatter(

    x=nj_timeline['date'],

    y=deaths,

    name='deaths')

container.append(trace)



dbl_time = doubling_time(np.asarray(days[-7:]).reshape(-1,1), np.log(cases.values[-7:].reshape(-1,1)))



tick_space = 86400000.0*2

fig = go.Figure(container)

ttext = "NJ Data as of "+datetime.datetime.today().strftime('%m/%d/%Y')+' starting on '+earliest_nj_date+'; double time = '+str(dbl_time)+' days'

fig.update_layout(xaxis_type="date", yaxis_type="log", title=ttext, xaxis_title='date')

fig.update_layout(xaxis_dtick=tick_space,xaxis_tickformat="%m/%d")

fig.show()
nj_timeline['d_cases']=nj_timeline['cases'].shift(-1)-nj_timeline['cases']

nj_timeline['d_deaths']=nj_timeline['deaths'].shift(-1)-nj_timeline['deaths']

nj_timeline['ma_d_cases']=nj_timeline.d_cases.rolling(ma_days).mean()

nj_timeline['ma_d_deaths']=nj_timeline.d_deaths.rolling(ma_days).mean()





container = []

trace=go.Bar(

    x=nj_timeline['date'],

    y=nj_timeline['d_cases'],

    name='new cases')

container.append(trace)



trace=go.Scatter(

    x=nj_timeline['date'],

    y=nj_timeline['ma_d_cases'],

    name='smoothed new cases')

container.append(trace)



trace=go.Bar(

    x=nj_timeline['date'],

    y=nj_timeline['d_deaths'],

    name='new deaths')

container.append(trace)



trace=go.Scatter(

    x=nj_timeline['date'],

    y=nj_timeline['ma_d_deaths'],

    name='smoothed new deaths')

container.append(trace)



tick_space = 86400000.0*2

fig = go.Figure(container)

ttext = "NJ Change Data as of "+datetime.datetime.today().strftime('%m/%d/%Y')+' starting on '+earliest_nj_date

fig.update_layout(xaxis_type="date", yaxis_type="linear", title=ttext, xaxis_title='date',yaxis_title='new cases/deaths')

fig.update_layout(xaxis_dtick=tick_space,xaxis_tickformat="%m/%d")

fig.show()

# Define a function that plots the new daily cases given county name as an input

def plot_county_change(county):

    #county='Monmouth'

    monmouth_county_data=cases_nj_hist[cases_nj_hist['county']==county].copy()

    monmouth_county_data['d_cases']=monmouth_county_data['cases'].shift(-1)-monmouth_county_data['cases']

    monmouth_county_data['ma_d_cases']=monmouth_county_data.d_cases.rolling(ma_days).mean()



    container = []

    trace=go.Bar(

        x=monmouth_county_data['date'],

        y=monmouth_county_data['d_cases'],

        name='new cases')

    container.append(trace)



    trace=go.Scatter(

        x=monmouth_county_data['date'],

        y=monmouth_county_data['ma_d_cases'],

        name='smoothed new cases')

    container.append(trace)

    

    dbl_time = doubling_time(np.asarray(days[-7:]).reshape(-1,1), np.log(monmouth_county_data['cases'].values[-7:].reshape(-1,1)))

    

    tick_space = 86400000.0*2

    fig = go.Figure(container)

    ttext = county+" County Change Data as of "+datetime.datetime.today().strftime('%m/%d/%Y')+' starting on '+earliest_nj_date+'; double time = '+str(dbl_time)+' days'

    fig.update_layout(xaxis_type="date", yaxis_type="linear", title=ttext, xaxis_title='date',yaxis_title='new cases')

    fig.update_layout(xaxis_dtick=tick_space,xaxis_tickformat="%m/%d")

    fig.show()

    

plot_county_change('Monmouth')
plot_county_change('Morris')
plot_county_change('Somerset')
plot_county_change('Ocean')
plot_county_change('Mercer')
df = pd.read_json('https://covidtracking.com/api/v1/states/daily.json')

df_nj = df[df['state']=='NJ'].copy()

df_nj['pct_pos']=100.*df_nj['positiveIncrease']/df_nj['totalTestResultsIncrease']

df_nj['date'] = pd.to_datetime(df_nj['date'].astype(str), format='%Y%m%d')

df_nj[df_nj['pct_pos'] > 60]=np.nan

trace=go.Scatter(

    x=df_nj['date'][:-20],

    y=df_nj['pct_pos'][:-20])



tick_space = 86400000.0*2

fig = go.Figure(trace)

ttext = 'Daily percent positive tests'

fig.update_layout(xaxis_type="date", yaxis_type="linear", title=ttext, xaxis_title='date',yaxis_title='percent positive')

fig.update_layout(xaxis_dtick=tick_space,xaxis_tickformat="%m/%d")

fig.show()