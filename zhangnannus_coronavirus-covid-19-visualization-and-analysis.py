from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 
plt.style.use('fivethirtyeight')
%matplotlib inline 
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
# Merge the three tables to one table so that the it is easy to control : new table columns = ['Province/State','Country/Region','Lat','Long', 'Date', 'No.Confirmed', 'No.death', 'No.Recoveries']
# Unpivot the tables 
basic_columns = ['Province/State','Country/Region', 'Lat','Long']
confirmed_df_pivote = confirmed_df.melt(id_vars=basic_columns, var_name='Date', value_name='No.Confirmed')
confirmed_df_pivote['Date'] = pd.to_datetime(confirmed_df_pivote['Date'])
deaths_df_pivote = deaths_df.melt(id_vars=basic_columns, var_name='Date', value_name='No.death')
deaths_df_pivote['Date'] = pd.to_datetime(deaths_df_pivote['Date'])
recoveries_df_pivote = recoveries_df.melt(id_vars=basic_columns, var_name='Date', value_name='No.Recoveries')
recoveries_df_pivote['Date'] = pd.to_datetime(recoveries_df_pivote['Date'])
# merge the sperate tables into one 
result = pd.merge(confirmed_df_pivote, deaths_df_pivote,  how='left', on = ['Province/State','Country/Region', 'Lat','Long', 'Date'])
result = pd.merge(result, recoveries_df_pivote,  how='left', on = ['Province/State','Country/Region', 'Lat','Long', 'Date'])
result.head()
# get overall trend data first

df_overall  = result.groupby('Date').agg({'No.Confirmed':'sum', 'No.death':'sum', 'No.Recoveries': 'sum'})\
                    .reset_index().sort_values(by='Date',ascending=True)
import plotly.graph_objects as go
# Create traces
fig = go.Figure()
fig.add_trace(go.Line(x=df_overall['Date'], y=df_overall['No.Confirmed'],
                    mode='lines',
                    name='No.Confirmed'))
fig.add_trace(go.Line(x=df_overall['Date'], y=df_overall['No.death'],
                    mode='lines',
                    name='No.death'))
fig.add_trace(go.Line(x=df_overall['Date'], y=df_overall['No.Recoveries'],
                    mode='lines', name='No.Recoveries'))

fig.update_layout(title_text="<b>Global trend</b>",)

fig.update_yaxes(title_text="<b>Numbers</b>")

fig.show()
# the top 10 countries data

top_countries  = list(result.groupby('Country/Region')\
                .agg({'No.Confirmed':'sum'})
                .sort_values(by='No.Confirmed',ascending=False)
                .reset_index()
                .head(10)['Country/Region'])

df_topCountries = result.loc[result['Country/Region'].isin(top_countries)]
# start plotting the top 10 countries  
fig = go.Figure()

for country in top_countries:
   
    df_inter =  df_topCountries.loc[df_topCountries['Country/Region'] == country]\
                .groupby(['Date','Country/Region'])\
                .agg({'No.Confirmed':'sum'})\
                .reset_index()\
                .sort_values(by='Date',ascending=True)
    
    fig.add_trace(go.Line(x=df_inter['Date'], y=df_inter['No.Confirmed'],
                    mode='lines',
                    name= country))

fig.update_layout(title_text="<b>Top 10 countries trend for confirmed cases</b>",)

fig.update_yaxes(title_text="<b>Numbers</b>")

fig.show()
# start plotting the top 10 countries  
fig = go.Figure()

for country in top_countries:
   
    df_inter =  df_topCountries.loc[df_topCountries['Country/Region'] == country]\
                .groupby(['Date','Country/Region'])\
                .agg({'No.death':'sum'})\
                .reset_index()\
                .sort_values(by='Date',ascending=True)
    
    fig.add_trace(go.Line(x=df_inter['Date'], y=df_inter['No.death'],
                    mode='lines',
                    name= country))

fig.update_layout(title_text="<b>Top 10 countries trend for Death cases</b>",)

fig.update_yaxes(title_text="<b>Numbers</b>")

fig.show()
# start plotting the top 10 countries  
fig = go.Figure()

for country in top_countries:
   
    df_inter =  df_topCountries.loc[df_topCountries['Country/Region'] == country]\
                .groupby(['Date','Country/Region'])\
                .agg({'No.Recoveries':'sum'})\
                .reset_index()\
                .sort_values(by='Date',ascending=True)
    
    fig.add_trace(go.Line(x=df_inter['Date'], y=df_inter['No.Recoveries'],
                    mode='lines',
                    name= country))

fig.update_layout(title_text="<b>Top 10 countries trend for Recoveries cases</b>",)

fig.update_yaxes(title_text="<b>Numbers</b>")

fig.show()
result['Date'] = result['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
result_map = result.groupby(['Date','Country/Region'])\
                .agg({'No.Confirmed':'sum', 'No.death':'sum', 'No.Recoveries': 'sum'})\
                .reset_index()\
                .sort_values(by='Date',ascending=True)
import plotly.express as px
fig = px.choropleth(result_map, 
                    locations="Country/Region", 
                    locationmode = "country names",
                    color="No.Confirmed", 
                    hover_name="Country/Region", 
                    animation_frame="Date"
                   )

fig.update_layout(
    title_text = 'Spread of Coronavirus',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()
set(result['Country/Region'])
def plot_country(result,country):

    df = result.loc[result['Country/Region'] == country]
    
    fig = go.Figure()
    fig.add_trace(go.Line(x=df['Date'], y=df['No.Confirmed'],
                        mode='lines',
                        name='No.Confirmed'))
    fig.add_trace(go.Line(x=df['Date'], y=df['No.death'],
                        mode='lines',
                        name='No.death'))
    fig.add_trace(go.Line(x=df['Date'], y=df['No.Recoveries'],
                        mode='lines', name='No.Recoveries'))

    fig.update_layout(title_text=f"<b>{country} trend</b>",)

    fig.update_yaxes(title_text="<b>Numbers</b>")

    fig.show()
    
plot_country(result,'Singapore')
plot_country(result,'US')
plot_country(result,'Japan')