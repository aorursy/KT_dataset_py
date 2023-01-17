import pandas as pd

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import json

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import warnings

from geojson import dump

warnings.filterwarnings('ignore')
temperatureByCity = pd.read_csv("/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv", encoding="utf-8")

temperatureByCountry = pd.read_csv("/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv", encoding="utf-8")

temperatureEarth = pd.read_csv("/kaggle/input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv", encoding="utf-8")
temperatureEarth.head()
def remove_outliers(df,col):

    q_low = df[col].quantile(0.01)

    q_hi  = df[col].quantile(0.99)

    df_filtered = df[(df[col] < q_hi) & (df[col] > q_low)]

    return df_filtered
temperatureEarth['year'] = [val[0:4] for val in temperatureEarth['dt']]

temperatureEarth = temperatureEarth.groupby('year').mean().reset_index()

temperatureEarth = remove_outliers(temperatureEarth,"LandAverageTemperature").astype({'year':'int32'})
fig = px.scatter(temperatureEarth,x='year', y='LandAverageTemperature')

fig.update_layout(title='<b>Evolution of the temperature per year in the world</b>',xaxis_title='', yaxis_title='')

fig.show()
temperatureFrance = temperatureByCountry.query("Country=='France'")

temperatureFrance['year'] = [val[0:4] for val in temperatureFrance['dt']]

temperatureFranceYear = temperatureFrance.groupby('year').mean().reset_index()
temperatureFranceYear = remove_outliers(temperatureFranceYear,"AverageTemperature")

fig = px.scatter(temperatureFranceYear,x='year', y='AverageTemperature')

fig.update_layout(title='<b>Evolution of the temperature per year in France</b>',xaxis_title='', yaxis_title='')

fig.show()
temperatureFrance['month'] = [val[5:7] for val in temperatureFrance['dt']]

month_name = {'01':'January','02':'February','03':"March",'04':"April",'05':"May",'06':"Jun",'07':"July",'08':"August",'09':"September",'10':"October",'11':"November",'12':"December"}

temperatureFranceMonth = temperatureFrance.groupby(['year','month']).mean().reset_index()

temperatureFranceMonth = remove_outliers(temperatureFranceMonth,"AverageTemperature")

temperatureFranceMonth['month'] = [month_name[row] for row in temperatureFranceMonth['month']]





months = temperatureFranceMonth.month.unique()

df_changement_temp_month = pd.DataFrame(columns=['month', 'diff'])

i=0



for month in months: 

    dfByMonth = temperatureFranceMonth.query("month==\""+month+"\"").astype({'year':'int32'})

    try:

        mean_val_1850_1900 = dfByMonth.query("year >= 1850 and year <= 1900")["AverageTemperature"].mean()

        mean_val_1990_2013 = dfByMonth.query("year >= 1990 and year <= 2013")["AverageTemperature"].mean()



        diff = mean_val_1990_2013-mean_val_1850_1900

    except:

        diff=0

        

    df_changement_temp_month.loc[i] = [month,diff]

    

    i+=1

df_changement_temp_month = df_changement_temp_month.set_index('month').reindex(["January","February","March","April","May","Jun","July","August","September","October","November","December"]).reset_index()



fig = px.bar(

        df_changement_temp_month,x='month', y='diff'

        )

    

fig.update_layout(title='<b></b>',xaxis_title='Month', yaxis_title='Degree increase')

fig.show() 
with open('/kaggle/input/geocountries/countries.geojson', 'r') as outfile:

    boundaries_courties = json.load(outfile)



avgTemperatureCountry = temperatureByCountry.groupby('Country').mean().reset_index()



size_json = len(boundaries_courties["features"])

for i in range(0,size_json):

    boundaries_courties["features"][i]['id']=boundaries_courties["features"][i]['properties']['ADMIN']
fig = px.choropleth_mapbox(avgTemperatureCountry, geojson=boundaries_courties, 

                           locations='Country',

                           color='AverageTemperature',

                           color_continuous_scale="balance",

                           range_color=(avgTemperatureCountry['AverageTemperature'].min(), avgTemperatureCountry['AverageTemperature'].max()),

                           mapbox_style="carto-positron",

                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},

                           opacity=0.5,

                           labels={'AverageTemperature':''}

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
temperatureByCountry['year'] = [val[0:4] for val in temperatureByCountry['dt']]

temperatureByCountry = temperatureByCountry.astype({'year': 'int32'})

avgYearCountry = temperatureByCountry.groupby(['Country','year']).mean().reset_index()

avgYearCountry = avgYearCountry.query("year >= 1850")

avgYearCountry.head()
countries = avgYearCountry.Country.unique()

df_changement_temp = pd.DataFrame(columns=['Country', 'diff'])

i=0



for country in countries: 

    dfByCountry = avgYearCountry.query("Country==\""+country+"\"")

    try:

        mean_val_1850_1900 = dfByCountry.query("year >= 1850 and year <= 1900")["AverageTemperature"].mean()

        mean_val_1990_2013 = dfByCountry.query("year >= 1990 and year <= 2013")["AverageTemperature"].mean()

        diff = mean_val_1990_2013-mean_val_1850_1900



    except :

        diff = 0

        

    df_changement_temp.loc[i] = [country,diff]

    

    i+=1

fig = px.choropleth_mapbox(df_changement_temp, geojson=boundaries_courties, 

                           locations='Country',

                           color='diff',

                           color_continuous_scale="balance",

                           range_color=(df_changement_temp['diff'].min(), df_changement_temp['diff'].max()),

                           mapbox_style="carto-positron",

                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},

                           opacity=0.5,

                           labels={'diff':''}

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()