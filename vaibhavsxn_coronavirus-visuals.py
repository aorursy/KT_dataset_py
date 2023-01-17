# import the necessary libraries

import numpy as np 

import pandas as pd 



# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

import pycountry

py.init_notebook_mode(connected=True)

import folium 

from folium import plugins



# Graphics in retina format 

%config InlineBackend.figure_format = 'retina' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

#plt.rcParams['image.cmap'] = 'viridis'





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Disable warnings 

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200201.csv', parse_dates = ['Last Update'])
world_data = pd.read_csv('/kaggle/input/world-cod/world_coordinates.csv')
data.info()
data.shape
data.head()
# Countries affected



countries = data['Country/Region'].unique().tolist()

print(countries)



print("\nTotal countries affected by virus: ",len(countries))
#Combining China and Mainland China cases



data['Country/Region'].replace({'Mainland China':'China'},inplace=True)

countries = data['Country/Region'].unique().tolist()

print(countries)

print("\nTotal countries affected by virus: ",len(countries))
confimed_mc = data.groupby('Country/Region')['Confirmed'].sum().reset_index(drop = False)
ccc = confimed_mc['Confirmed'][confimed_mc['Country/Region'] == 'China'].values.astype(int)[0]
print('The total number of confimed cases in China are', ccc)
world_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')
# Creating a dataframe with total no of confirmed cases for every country

Number_of_countries = len(data['Country/Region'].value_counts())





cases = pd.DataFrame(data.groupby('Country/Region')['Confirmed'].sum())

cases['Country'] = cases.index

cases.index=np.arange(1,Number_of_countries+1)



global_cases = cases[['Country','Confirmed']]

#global_cases.sort_values(by=['Confirmed'],ascending=False)

global_cases
# Merging the coordinates dataframe with original dataframe

world_data = pd.merge(world_data,global_cases,on='Country')

world_data.head()
for lat, lon, value, name in zip(world_data['latitude'], world_data['longitude'], world_data['Confirmed'], world_data['Country']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(world_map)

world_map
d = data['Last Update'][-1:].astype('str')

year = int(d.values[0].split('-')[0])

month = int(d.values[0].split('-')[1])

day = int(d.values[0].split('-')[2].split()[0])



from datetime import date

data_latest = data[data['Last Update'] > pd.Timestamp(date(year,month,day))]

data_latest.head()
# A look at the different cases - confirmed, death and recovered

print('Globally Confirmed Cases: ',data['Confirmed'].sum())

print('Global Deaths: ',data['Death'].sum())

print('Globally Recovered Cases: ',data['Recovered'].sum())
# Let's look the various Provinces/States affected



data.groupby(['Country/Region','Province/State']).sum()
# Provinces where deaths have taken place

data.groupby('Country/Region')['Death'].sum().sort_values(ascending=False)[:5]
# Lets also look at the Recovered stats

data.groupby('Country/Region')['Recovered'].sum().sort_values(ascending=False)[:5]
#Mainland China

China = data[data['Country/Region']=='China']

China
f, ax = plt.subplots(figsize=(12, 8))



sns.set_color_codes("pastel")

sns.barplot(x="Confirmed", y="Province/State", data=China[1:],

            label="Confirmed", color="r")



sns.set_color_codes("muted")

sns.barplot(x="Recovered", y="Province/State", data=China[1:],

            label="Recovered", color="g")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 400), ylabel="",

       xlabel="Stats")

sns.despine(left=True, bottom=True)
latitude = 39.91666667

longitude = 116.383333

 

# create map and display it

china_map = folium.Map(location=[latitude, longitude], zoom_start=12)



china_coordinates= pd.read_csv("/kaggle/input/china-coordinates/cn.csv")

china_coordinates.rename(columns={'admin':'Province/State'},inplace=True)

df_china_virus = China.merge(china_coordinates)





# Make a data frame with dots to show on the map

data1 = pd.DataFrame({

   'name':list(df_china_virus['Province/State']),

   'lat':list(df_china_virus['lat']),

   'lon':list(df_china_virus['lng']),

   'Confirmed':list(df_china_virus['Confirmed']),

   'Recovered':list(df_china_virus['Recovered']),

   'Deaths':list(df_china_virus['Death'])

})



data1.head()
# create map for total confirmed cases in china till date

china_map1 = folium.Map(location=[latitude, longitude], zoom_start=4,tiles='Stamen Toner')



for lat, lon, value, name in zip(data1['lat'], data1['lon'], data1['Confirmed'], data1['name']):

    folium.CircleMarker([lat, lon],

                        radius=5,

                        popup = ('Province: ' + str(name).capitalize() + '<br>'

                        'Confirmed: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(china_map1)

    folium.Map(titles='jj', attr="attribution")    

china_map1
china_map = folium.Map(location=[latitude, longitude], zoom_start=4,tiles='Stamen Toner')



for lat, lon, value, name in zip(data1['lat'], data1['lon'], data1['Deaths'], data1['name']):

    folium.CircleMarker([lat, lon],

                        radius=value*0.2,

                        popup = ('Province: ' + str(name).capitalize() + '<br>'

                        'Deaths: ' + str(value) + '<br>'),

                        color='black',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(china_map)

    folium.Map(titles='jj', attr="attribution")    

china_map
china_map = folium.Map(location=[latitude, longitude], zoom_start=4,tiles='Stamen Toner')



for lat, lon, value, name in zip(data1['lat'], data1['lon'], data1['Deaths'], data1['name']):

    folium.CircleMarker([lat, lon],

                        radius=value*0.2,

                        popup = ('Province: ' + str(name).capitalize() + '<br>'

                        'Deaths: ' + str(value) + '<br>'),

                        color='black',

                        

                        fill_color='green',

                        fill_opacity=0.7 ).add_to(china_map)

    folium.Map(titles='jj', attr="attribution")    

china_map