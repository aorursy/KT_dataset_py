import numpy as np

import pandas as pd



# EDA packages

import pandas_profiling as pp



# visualization packages

import seaborn as sns

from matplotlib import pyplot as plt

import matplotlib.dates as mdates

import plotly.graph_objects as go

import pycountry

import plotly.express as px





# forecast packages

from fbprophet import Prophet

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
recov_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

death_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

conf_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

open_line_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

line_list_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')

covid_19_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
countries = pd.read_csv('/kaggle/input/countries-of-the-world-iso-codes-and-population/countries_by_population_2019.csv')

countries_iso = pd.read_csv('/kaggle/input/countries-of-the-world-iso-codes-and-population/country_codes_2020.csv')
covid_19_df.head()
covid_19_df['ObservationDate']=pd.to_datetime(covid_19_df['ObservationDate'])

covid_19_df['Last Update']=pd.to_datetime(covid_19_df['Last Update'])
covid_19_pp_report = pp.ProfileReport(covid_19_df,html={'style':{'full_width':True}},progress_bar=False)
covid_19_pp_report
# Rows where update datetime is more than 1 day later than observation date

covid_19_df[covid_19_df['Last Update']-covid_19_df['ObservationDate']>'1 day']
# Rows where update datetime is more than 1 day before the observation date

covid_19_df[covid_19_df['Last Update']-covid_19_df['ObservationDate']<'-1 day']
gl_cumm = covid_19_df.groupby('ObservationDate').sum()[['Confirmed','Deaths','Recovered']]
plt.figure(figsize=(16,8))

sns.set_style("darkgrid", {'axes.facecolor': ".9",'grid.linestyle': '--'})

sns.lineplot(data=gl_cumm, x=gl_cumm.index, y='Confirmed', label='Confirmed', lw=3).xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

sns.lineplot(data=gl_cumm, x=gl_cumm.index, y='Deaths',label='Deaths',lw=3, color='darkred')

sns.lineplot(data=gl_cumm, x=gl_cumm.index, y='Recovered',label='Recovered',lw=3, color='darkgreen')

plt.show()
cntry_cumm = covid_19_df.groupby(['ObservationDate','Country/Region']).sum()[['Confirmed','Deaths','Recovered']].reset_index()

top_20  =cntry_cumm[cntry_cumm['ObservationDate']==cntry_cumm['ObservationDate'].max()].sort_values('Confirmed',ascending=False)['Country/Region'][:20].tolist()
plt.figure(figsize=(16,8))

sns.set_style("darkgrid", {'axes.facecolor': ".9",'grid.linestyle': '--'})

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region'].isin(top_20)], 

             x='ObservationDate', y='Confirmed',hue='Country/Region'

            ).xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

plt.show()
plt.figure(figsize=(16,8))

sns.set_style("darkgrid", {'axes.facecolor': ".9",'grid.linestyle': '--'})

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region'].isin(top_20[1:])], 

             x='ObservationDate', y='Confirmed',hue='Country/Region'

            ).xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

plt.show()
# adding view into 'remaining cases'

cntry_cumm['Remaining'] = cntry_cumm['Confirmed'] -  cntry_cumm['Recovered'] - cntry_cumm['Deaths']
cntry_cumm.head()
plt.figure(figsize=(16,8))

sns.set_style("darkgrid", {'axes.facecolor': ".9",'grid.linestyle': '--'})

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region']=='Mainland China'], 

             x='ObservationDate', y='Confirmed', label= 'Confirmed', lw=5).xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region']=='Mainland China'], 

             x='ObservationDate', y='Recovered', label= 'Recovered', color='darkgreen', lw=3)

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region']=='Mainland China'], 

             x='ObservationDate', y='Deaths', label= 'Deaths', color='darkred', lw=3)

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region']=='Mainland China'], 

             x='ObservationDate', y='Remaining', label= 'Remaining', color='black', lw=5)

plt.show()
plt.figure(figsize=(16,8))

sns.set_style("darkgrid", {'axes.facecolor': ".9",'grid.linestyle': '--'})

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region']=='Canada'], 

             x='ObservationDate', y='Confirmed', label= 'Confirmed', lw=5).xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region']=='Canada'], 

             x='ObservationDate', y='Recovered', label= 'Recovered', color='darkgreen', lw=3)

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region']=='Canada'], 

             x='ObservationDate', y='Deaths', label= 'Deaths', color='darkred', lw=3)

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region']=='Canada'], 

             x='ObservationDate', y='Remaining', label= 'Remaining', color='black', lw=5)

plt.show()
plt.figure(figsize=(16,8))

sns.set_style("darkgrid", {'axes.facecolor': ".9",'grid.linestyle': '--'})

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region']=='Italy'], 

             x='ObservationDate', y='Confirmed', label= 'Confirmed', lw=5).xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region']=='Italy'], 

             x='ObservationDate', y='Recovered', label= 'Recovered', color='darkgreen', lw=3)

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region']=='Italy'], 

             x='ObservationDate', y='Deaths', label= 'Deaths', color='darkred', lw=3)

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region']=='Italy'], 

             x='ObservationDate', y='Remaining', label= 'Remaining', color='black', lw=5)

plt.show()
plt.figure(figsize=(16,8))

sns.set_style("darkgrid", {'axes.facecolor': ".9",'grid.linestyle': '--'})

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region']=='South Korea'], 

             x='ObservationDate', y='Confirmed', label= 'Confirmed', lw=5).xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region']=='South Korea'], 

             x='ObservationDate', y='Recovered', label= 'Recovered', color='darkgreen', lw=3)

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region']=='South Korea'], 

             x='ObservationDate', y='Deaths', label= 'Deaths', color='darkred', lw=3)

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region']=='South Korea'], 

             x='ObservationDate', y='Remaining', label= 'Remaining', color='black', lw=5)

plt.show()
countries.head()
countries = countries.drop('Rank',axis=1)

countries['name'].replace({'China': 'Mainland China'}, inplace=True)

countries['name'].replace({'United States':'US'}, inplace=True)

countries['name'].replace({'United Kingdom': 'UK'}, inplace=True)
cntry_cumm = pd.merge(cntry_cumm, countries, left_on='Country/Region', right_on='name', how='left')

cntry_cumm['confirmed_per_cap'] = cntry_cumm['Confirmed']/cntry_cumm['pop2019']
top_20_pc  = cntry_cumm[cntry_cumm['ObservationDate']==cntry_cumm['ObservationDate'].max()].sort_values('confirmed_per_cap',ascending=False)['Country/Region'][:20].tolist()
plt.figure(figsize=(24,8))

sns.set_style("darkgrid", {'axes.facecolor': ".9",'grid.linestyle': '--'})



plt.subplot(1, 2, 1)

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region'].isin(top_20_pc[:10])], 

             x='ObservationDate', y='confirmed_per_cap',hue='Country/Region'

            ).xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

plt.title('Top 1-10 Countries')



plt.subplot(1, 2, 2)

sns.lineplot(data=cntry_cumm[cntry_cumm['Country/Region'].isin(top_20_pc[10:])], 

             x='ObservationDate', y='confirmed_per_cap',hue='Country/Region'

            ).xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

plt.title('Top 11-20 Countries')

plt.show()
corr = cntry_cumm[cntry_cumm['ObservationDate']==cntry_cumm['ObservationDate'].max()][['Confirmed','confirmed_per_cap','Density']].corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



f, ax = plt.subplots(figsize=(6, 6))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap="YlGnBu", vmax=.3, center=0,annot=True,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
conf_df.head()
death_df.head()
recov_df.head()
# check if all 3 datasets have the same dimensions

print(conf_df.shape)

print(recov_df.shape)

print(death_df.shape)
cntry_geo_df = conf_df[['Province/State','Country/Region','Lat','Long']].drop_duplicates()

cntry_geo_df['Country/Region'].replace({'China': 'Mainland China'}, inplace=True)
# take lat/long info to covid-19 dataset

covid_19_df2 = pd.merge(covid_19_df, cntry_geo_df, on=["Country/Region", "Province/State"], how='left')

covid_19_df2['Date']=covid_19_df2['ObservationDate'].astype(str)

covid_19_df2['Remaining'] = covid_19_df2['Confirmed'] - covid_19_df2['Recovered'] - covid_19_df2['Deaths']
covid_19_df2.shape
covid_19_df2.head()
fig = px.density_mapbox(covid_19_df2, 

                        lat="Lat", 

                        lon="Long", 

                        hover_name='Province/State', 

                        hover_data=['Confirmed','Deaths','Recovered','Remaining'], 

                        animation_frame='Date',

                        #color_continuous_scale="Portland",

                        radius=10, 

                        zoom=0,

                        height=800)

fig.update_layout(title='Worldwide Corona Virus Cases Time Lapse - Confirmed, Deaths, Recovered & Remaining',

                  font=dict(family="Courier New, monospace",

                            size=18,

                            color="#7f7f7f")

                 )

fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})





fig.show()
# drop null columns

line_list_df = line_list_df.dropna(axis=0,how='all')

line_list_df = line_list_df.dropna(axis=1,how='all')

open_line_df = open_line_df.dropna(axis=0,how='all')

open_line_df = open_line_df.dropna(axis=1,how='all')
line_list_report = pp.ProfileReport(line_list_df,html={'style':{'full_width':True}},progress_bar=False)

open_line_report = pp.ProfileReport(open_line_df,html={'style':{'full_width':True}},progress_bar=False)
line_list_report
open_line_report
# few intersting variables

open_line_df[['ID','age','sex','city','province','country','date_confirmation','chronic_disease','chronic_disease_binary','symptoms', 'travel_history_dates', 'travel_history_location']]
model_df = pd.concat([covid_19_df2.groupby('ObservationDate').sum()[['Confirmed','Deaths','Recovered','Remaining']],

                      covid_19_df2.groupby('ObservationDate').mean()[['Confirmed','Deaths','Recovered','Remaining']],

                      covid_19_df2.groupby('ObservationDate').var()[['Confirmed','Deaths','Recovered','Remaining']],

                      covid_19_df2.groupby('ObservationDate').skew()[['Confirmed','Deaths','Recovered','Remaining']],

                      covid_19_df2.groupby('ObservationDate').count()[['Country/Region','Province/State']]

                     ],axis=1).reset_index()

model_df.columns = ['ds','y','CummDeaths','CummRecovered','CummRemaining',

                    'AvgConfirmed','AvgDeaths','AvgRecovered','AvgRemaining',

                    'VarConfirmed','VarDeaths','VarRecovered','VarRemaining',

                    'SkewConfirmed','SkewDeaths','SkewRecovered','SkewRemaining',

                    'NoCountries','NoProvinces']
model_df.head(5)
model_df['IncrConfirmed'] = model_df['y'] - model_df['y'].shift(1)

model_df['IncrDeaths'] = model_df['CummDeaths'] - model_df['CummDeaths'].shift(1)

model_df['IncrRecovered'] = model_df['CummRecovered'] - model_df['CummRecovered'].shift(1)

model_df['IncrRemaining'] = model_df['CummRemaining'] - model_df['CummRemaining'].shift(1)
train_df = model_df[(model_df.ds < '2020-03-18') & (model_df.ds > '2020-01-22')]

test_df = model_df[model_df.ds >= '2020-03-18']
reg_var = model_df.columns[2:].tolist()
from fbprophet import Prophet
model = Prophet(interval_width=0.95, 

                #weekly_seasonality=True, 

                #daily_seasonality=True,

                #holidays=holidays,

                #changepoint_prior_scale=10,

                seasonality_mode='multiplicative')



model.add_seasonality(name='biweekly', period=14, fourier_order=100, mode = 'multiplicative')



for var in reg_var:

    model.add_regressor(var)



model.fit(train_df)
future = model.make_future_dataframe(freq='D',periods = 7)

future[reg_var] = test_df[reg_var]

forecast = model.predict(test_df.drop('y',axis=1))
fig1 = model.plot(forecast, xlabel=u'Date', ylabel=u'Expected contract inception').set_size_inches(10,5)

plt.title('GAM prediction interval', fontsize=20)

plt.show()
model.plot_components(forecast).set_size_inches(10,10)