import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', None)



import matplotlib.style as style

style.use('fivethirtyeight')

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.dpi'] = 150 #set figure size



from plotly.offline import iplot, init_notebook_mode

import plotly.express as px

#import plotly.plotly as py

import plotly.graph_objs as go

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)



import folium



#df = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

#COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", parse_dates=['Last Update'])

df.rename(columns={'Country/Region':'Country'}, inplace=True)

df = df.drop(columns = ['SNo', "Last Update"]) #only confuses



df_conf = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df_conf.rename(columns={'Country/Region':'Country'}, inplace=True)



df_death = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

df_death.rename(columns={'Country/Region':'Country'}, inplace=True)

# time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

countries = pd.read_csv("../input/countries-of-the-world-iso-codes-and-population/countries_by_population_2019.csv")

countries_iso = pd.read_csv("../input/countries-of-the-world-iso-codes-and-population/country_codes_2020.csv")



us_covid = pd.read_csv('../input/covid19-us-county-jhu-data-demographics/covid_us_county.csv')

us_county = pd.read_csv('../input/covid19-us-county-jhu-data-demographics/us_county.csv')
df.sample(5)
#strip white spaces are there is one country (Azerbaijan) with a whitespace observation

df['Country'] = df['Country'].str.strip()



#fill missing Province/State with Country

df.loc[df['Province/State'].isnull(), 'Province/State'] = df.loc[df['Province/State'].isnull(), 'Country']



#keep most recent line per Province/State and Country

df.sort_values(['Country', 'Province/State', 'ObservationDate'], ascending = [True,True,False], inplace = True)

df = df.drop_duplicates(['Country', 'Province/State'], keep = "first")



#keep a copy for later on

df_state = df.copy()



df = df.drop(columns = "ObservationDate")



#groupby Country

df_country = df.groupby(['Country'], as_index=False)['Confirmed', 'Deaths'].sum()



#drop some columns

cols_to_drop = ['Rank', 'pop2018','GrowthRate', 'area', 'Density']

countries = countries.drop(columns = cols_to_drop)



#add ISO Alpha 3 code that I uploaded in another CSV

countries = countries.merge(countries_iso[['name', 'cca3']], on = ['name'], how = "left")



cols_to_rename = {'name': 'Country', 'pop2019': 'Population', 'cca3': 'ISO'}

countries = countries.rename(columns = cols_to_rename)



#just fixing the most important mismatches

countries_to_rename = {'US': 'United States',\

                       'Mainland China': 'China',\

                       'UK': 'United Kingdom',\

                       'Congo (Kinshasa)': 'DR Congo',\

                       'North Macedonia': 'Macedonia',\

                       'Republic of Ireland': 'Ireland',\

                       'Congo (Brazzaville)': 'Republic of the Congo'}



df_country['Country'] = df_country['Country'].replace(countries_to_rename)



df_country = df_country.merge(countries, on = "Country", how = "left")



#check mismatches

#df_country[df_country.ISO.isnull()].sort_values(['Confirmed'], ascending = False)



#dropping not matching countries, only small islands left

df_country = df_country.dropna()



#rounding population to millions with 2 digits, and creating two new columns

df_country['Population'] = round((df_country['Population']/1000),2)

df_country = df_country.rename(columns = {'Population': 'Population (million)'})

df_country['Cases per Million'] = round((df_country['Confirmed']/df_country['Population (million)']),2)

df_country['Deaths per Million'] = round((df_country['Deaths']/df_country['Population (million)']),2)



#filter out countries with less than a million population as for instance San Marino has extremely high figures on a very small population

df_country = df_country[(df_country['Population (million)'] > 1)]



df_country.sample(5)
df_country = df_country.sort_values(['Cases per Million'], ascending = False).reset_index(drop=True)

df_country.drop(columns = ['ISO', 'Deaths', 'Deaths per Million']).head(10).style.background_gradient(cmap='Reds', subset = ['Cases per Million'])
fig = px.choropleth(df_country, locations="ISO",

                    color="Cases per Million",

                    hover_name="Country",

                    color_continuous_scale=px.colors.sequential.YlOrRd)



layout = go.Layout(

    title=go.layout.Title(

        text="Corona confirmed cases per million inhabitants",

        x=0.5

    ),

    font=dict(size=14),

    width = 750,

    height = 350,

    margin=dict(l=0,r=0,b=0,t=30)

)



fig.update_layout(layout)



fig.show()
df_country = df_country.sort_values(['Deaths per Million'], ascending = False).reset_index(drop=True)

df_country.drop(columns = ['ISO', 'Confirmed', 'Cases per Million']).head(10).style.background_gradient(cmap='Reds', subset = ['Deaths per Million'])
fig = px.choropleth(df_country, locations="ISO",

                    color="Deaths per Million",

                    hover_name="Country",

                    color_continuous_scale=px.colors.sequential.YlOrRd)



layout = go.Layout(

    title=go.layout.Title(

        text="Corona deaths per million inhabitants",

        x=0.5

    ),

    font=dict(size=14),

    width = 750,

    height = 350,

    margin=dict(l=0,r=0,b=0,t=30)

)



fig.update_layout(layout)



fig.show()
#get names of first 4 and last 5 columns

cols_to_select = list(df_conf.columns[0:4]) + list(df_conf.columns[-6:])

df_conf.loc[(df_conf['Country'] == "Netherlands"), cols_to_select]
#only keep last date available

cols_to_keep = list(df_conf.columns[0:4]) + list(df_conf.columns[-1:])

df_conf_last = df_conf[cols_to_keep]

df_conf_last.columns.values[-1] = "Confirmed"



df_conf_last.head()
#float required

df_conf_last['Confirmed'] = df_conf_last['Confirmed'].astype(float)



map1 = folium.Map(location=[30.6, 114], zoom_start=3) #US=[39,-98] Europe =[45, 5]



for i in range(0,len(df_conf_last)):

   folium.Circle(

      location=[df_conf_last.iloc[i]['Lat'], df_conf_last.iloc[i]['Long']],

      tooltip = "Country: "+df_conf_last.iloc[i]['Country']+"<br>Province/State: "+str(df_conf_last.iloc[i]['Province/State'])+"<br>Confirmed cases: "+str(df_conf_last.iloc[i]['Confirmed'].astype(int)),

      radius=df_conf_last.iloc[i]['Confirmed']*5,

      color='crimson',

      fill=True,

      fill_color='crimson'

   ).add_to(map1)



map1
#only keep last date available

cols_to_keep = list(df_death.columns[0:4]) + list(df_death.columns[-1:])

df_death_last = df_death[cols_to_keep]

df_death_last.columns.values[-1] = "Death"



#float required

df_death_last['Death'] = df_death_last['Death'].astype(float)



map2 = folium.Map(location=[30.6, 114], zoom_start=3)



for i in range(0,len(df_death_last)):

   folium.Circle(

      location=[df_death_last.iloc[i]['Lat'], df_death_last.iloc[i]['Long']],

      tooltip = "Country: "+df_death_last.iloc[i]['Country']+"<br>Province/State: "+str(df_death_last.iloc[i]['Province/State'])+"<br>Deaths: "+str(df_death_last.iloc[i]['Death'].astype(int)),

      radius=df_death_last.iloc[i]['Death']*100,

      color='crimson',

      fill=True,

      fill_color='crimson'

   ).add_to(map2)



map2
ts_country = df_conf.drop(columns = ['Lat', 'Long', 'Province/State'])

ts_country = ts_country.groupby(['Country']).sum()



#get countries with most cases on last date in dataframe

ts_country = ts_country.sort_values(by = ts_country.columns[-1], ascending = False).head(7)

#drop last date as not always updated

#ts_country.drop(ts_country.columns[len(ts_country.columns)-1], axis=1, inplace=True)



ts_country.transpose().iplot(title = 'Time series of confirmed cases of countries with most confirmed cases')
ts_country = df_death.drop(columns = ['Lat', 'Long', 'Province/State'])

ts_country = ts_country.groupby(['Country']).sum()



#get countries with most cases on last date in dataframe

ts_country = ts_country.sort_values(by = ts_country.columns[-1], ascending = False).head(7)

#drop last date as not always updated

ts_country.drop(ts_country.columns[len(ts_country.columns)-1], axis=1, inplace=True)



ts_country.transpose().iplot(title = 'Time series of deaths of countries with most victims')
ts_country = ts_country.transpose()



df1 = ts_country.iloc[:,0].to_frame()

df1 = df1[df1.iloc[:,0] !=0].reset_index(drop=True)



for i in range(1,ts_country.shape[1]):

    df = ts_country.iloc[:,i].to_frame()

    df = df[df.iloc[:,0] !=0].reset_index(drop=True)

    df1 = pd.concat([df1, df], join='outer', axis=1)



    

df1.iplot(title = 'Time series of deaths since first victim', xTitle = 'Days since first reported Death', yTitle = 'Number of Deaths')
df_country = df_country.drop(columns = ['Population (million)', 'ISO', 'Cases per Million', 'Deaths per Million'])

df_country['Percent Death'] = round(((df_country.Deaths / df_country.Confirmed)*100),2)

#filter countries with at least 100 deaths

df_country = df_country[(df_country.Deaths >= 100)]



#set font size for plotting

#plt.rcParams.update({'font.size': 12})



#create barplot

se = df_country[['Country', 'Percent Death']].sort_values(by = "Percent Death", ascending = False).set_index("Country")

se = se[0:10].sort_values(by = "Percent Death", ascending = True)

se.plot.barh()

plt.title("Countries with worst ratio confirmed cases vs. Deaths")

plt.xticks(rotation=0);
#create barplot

se = df_country[['Country', 'Percent Death']].sort_values(by = "Percent Death", ascending = False).set_index("Country")

se = se[-10:]

se.plot.barh()

plt.title("Countries with doing best regarding confirmed cases vs. Deaths")

plt.xticks(rotation=0);
#fips of 2 counties are missing (Dukes and Nantucket, Kansas City)

#quick fix for now

us_covid = us_covid[us_covid.fips.notnull()]

us_covid['fips'] = us_covid['fips'].astype(object)

us_county['fips'] = us_county['fips'].astype(object)



#add popultation from second csv

us_covid = us_covid.merge(us_county[['fips', 'population']], on = ['fips'], how = "left")



#keep latest date only

us_cum = us_covid.sort_values(by = ['county', 'state', 'date'], ascending = [True, True, False])

us_cum = us_cum.drop_duplicates(subset = ['county', 'state'], keep = "first")



#save a copy

counties_us = us_cum.copy()



#groupby State

us_cum = us_cum.groupby(['state', 'date'], as_index=False)['cases', 'deaths', 'population'].sum()



us_cum['population'] = us_cum['population'].astype(int)



#rounding population to millions with 2 digits, and creating two new columns

us_cum['population'] = round((us_cum['population']/1000000),2)

us_cum = us_cum.rename(columns = {'population': 'Population (million)'})

us_cum['Cases per Million'] = round((us_cum['cases']/us_cum['Population (million)']),2)

us_cum['Deaths per Million'] = round((us_cum['deaths']/us_cum['Population (million)']),2)



#remove states with missing population

us_cum = us_cum[(us_cum['Population (million)'] != 0)]
us_cum = us_cum.sort_values(by = "Deaths per Million", ascending = False).reset_index(drop=True)

us_cum.head(10).style.background_gradient(cmap='Reds', subset = ['Deaths per Million'])
nyc_counties = ['New York', 'Kings', 'Queens', 'Bronx', 'Richmond']

new_york = counties_us[((counties_us.state == "New York") & (counties_us.county.isin(nyc_counties)))].sort_values(by="fips")

new_york
nyc = new_york.groupby(['state', 'date'])['cases', 'deaths', 'population'].sum()

nyc.index.names = ['city', 'date']



nyc['population'] = nyc['population'].astype(int)



#rounding population to millions with 2 digits, and creating two new columns

nyc['population'] = round((nyc['population']/1000000),2)

nyc = nyc.rename(columns = {'population': 'Population (million)'})

nyc['Cases per Million'] = round((nyc['cases']/nyc['Population (million)']),2).astype(int)

nyc['Deaths per Million'] = round((nyc['deaths']/nyc['Population (million)']),2).astype(int)



nyc