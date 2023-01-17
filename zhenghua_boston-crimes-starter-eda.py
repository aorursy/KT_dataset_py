# Load libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

import folium

import os

from folium.plugins import HeatMap

from haversine import haversine, Unit

# Import data

data = pd.read_csv('/kaggle/input/crimes-in-boston/crime.csv', encoding='latin-1')

raw_data=data

# Peek

data.head()

data['DISTRICT'].unique()
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data.describe()
# Keep only data from complete years (2016, 2017)

data = data.loc[data['YEAR'].isin([2016,2017])]



# Keep only data on UCR Part One offenses

data = data.loc[data['UCR_PART'] == 'Part One']



# Remove unused columns

data = data.drop(['INCIDENT_NUMBER','OFFENSE_CODE','UCR_PART','Location'], axis=1)



# Convert OCCURED_ON_DATE to datetime

data['OCCURRED_ON_DATE'] = pd.to_datetime(data['OCCURRED_ON_DATE'])



# Fill in nans in SHOOTING column

data.SHOOTING.fillna('N', inplace=True)



# Convert DAY_OF_WEEK to an ordered category

data.DAY_OF_WEEK = pd.Categorical(data.DAY_OF_WEEK, 

              categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],

              ordered=True)



# Replace -1 values in Lat/Long with Nan

data.Lat.replace(-1, None, inplace=True)

data.Long.replace(-1, None, inplace=True)



# Rename columns to something easier to type (the all-caps are annoying!)

rename = {'OFFENSE_CODE_GROUP':'Group',

         'OFFENSE_DESCRIPTION':'Description',

         'DISTRICT':'District',

         'REPORTING_AREA':'Area',

         'SHOOTING':'Shooting',

         'OCCURRED_ON_DATE':'Date',

         'YEAR':'Year',

         'MONTH':'Month',

         'DAY_OF_WEEK':'Day',

         'HOUR':'Hour',

         'STREET':'Street'}

data.rename(index=str, columns=rename, inplace=True)



# Check

data.head()
# A few more data checks

data.dtypes

data.isnull().sum()

data.shape

# Countplot for crime types

sns.catplot(y='Group',

           kind='count',

            height=8, 

            aspect=1.5,

            order=data.Group.value_counts().index,

           data=data)
# Crimes by hour of the day

sns.catplot(x='Hour',

           kind='count',

            height=8.27, 

            aspect=3,

            #color='blue',

           data=data)

plt.xticks(size=30)

plt.yticks(size=30)

plt.xlabel('Hour', fontsize=40)

plt.ylabel('Count', fontsize=40)
# Crimes by day of the week

sns.catplot(x='Day',

           kind='count',

            height=8, 

            aspect=3,

           data=data)

plt.xticks(size=30)

plt.yticks(size=30)

plt.xlabel('')

plt.ylabel('Count', fontsize=40)
# # Crimes by month of year

# months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# sns.catplot(x='Month',

#            kind='count',

#             height=8, 

#             aspect=3,

#             #color='gray',

#            data=data)

# plt.xticks(np.arange(12), months, size=30)

# plt.yticks(size=30)

# plt.xlabel('')

# plt.ylabel('Count', fontsize=40)
# # Create data for plotting

# data['Day_of_year'] = data.Date.dt.dayofyear

# data_holidays = data[data.Year == 2017].groupby(['Day_of_year']).size().reset_index(name='counts')



# # Dates of major U.S. holidays in 2017

# holidays = pd.Series(['2017-01-01', # New Years Day

#                      '2017-01-16', # MLK Day

#                      '2017-03-17', # St. Patrick's Day

#                      '2017-04-17', # Boston marathon

#                      '2017-05-29', # Memorial Day

#                      '2017-07-04', # Independence Day

#                      '2017-09-04', # Labor Day

#                      '2017-10-10', # Veterans Day

#                      '2017-11-23', # Thanksgiving

#                      '2017-12-25']) # Christmas

# holidays = pd.to_datetime(holidays).dt.dayofyear

# holidays_names = ['NY',

#                  'MLK',

#                  'St Pats',

#                  'Marathon',

#                  'Mem',

#                  'July 4',

#                  'Labor',

#                  'Vets',

#                  'Thnx',

#                  'Xmas']



# import datetime as dt

# # Plot crimes and holidays

# fig, ax = plt.subplots(figsize=(11,6))

# sns.lineplot(x='Day_of_year',

#             y='counts',

#             ax=ax,

#             data=data_holidays)

# plt.xlabel('Day of the year')

# plt.vlines(holidays, 20, 80, alpha=0.5, color ='r')

# for i in range(len(holidays)):

#     plt.text(x=holidays[i], y=82, s=holidays_names[i])
# fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 7))

# # Simple scatterplot

# sns.scatterplot(x='Lat',

#                y='Long',

#                 alpha=0.01,

#                data=data,

#                ax = axes[0])

# # Plot districts

# sns.scatterplot(x='Lat',

#                y='Long',

#                 hue='District',

#                 alpha=0.1,

#                data=data,

#                ax = axes[1])
# Plot districts

sns.scatterplot(x='Lat',

               y='Long',

                hue='District',

                alpha=0.1,

               data=data)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
# Create basic Folium crime map

crime_map = folium.Map(location=[42.3125,-71.0875], 

                       tiles = "Stamen Toner",

                      zoom_start = 11)



# Add data for heatmp 

data_heatmap = data[data.Year == 2017]

data_heatmap = data[['Lat','Long']]

data_heatmap = data.dropna(axis=0, subset=['Lat','Long'])

data_heatmap = [[row['Lat'],row['Long']] for index, row in data_heatmap.iterrows()]

HeatMap(data_heatmap, radius=10).add_to(crime_map)



# Plot!

crime_map
import os

import pandas as pd

import json

import geopandas as gpd





gdf = gpd.read_file('/kaggle/input/geo-json-boston/Police_Districts.geojson')

gdf.head()

state_data = raw_data[['DISTRICT','INCIDENT_NUMBER']].groupby('DISTRICT').agg('count') 

state_data = state_data / state_data.sum()

merged = gdf.merge(state_data, left_on='DISTRICT', right_on='DISTRICT')

result_data = pd.read_csv('/kaggle/input/distributiond/sensitivity_specialist.csv', encoding='latin-1')

result_data['DISTRICT'] = result_data['Unnamed: 0']

result_data.drop('Unnamed: 0', inplace=True, axis= 1)

result_data



merged = merged.merge(result_data, left_on = 'DISTRICT', right_on = 'DISTRICT')
merged
merged.iloc[:, [1, 9]]
spatial_gdf = gpd.GeoDataFrame(merged.iloc[:, [1, 8]])

crime_distribution = merged.iloc[:, [1, 9]]

specialist_dist = merged.iloc[:, [1, 10]]

officer_dist = merged.iloc[:, [1, 11]]

geo_str = spatial_gdf.to_json()
import os

import pandas as pd

import json

import geopandas as gpd





# Create basic Folium crime map

crime_map = folium.Map(location=[42.3125,-71.0875], 

                       tiles="Cartodb Positron",

                      zoom_start = 12)





m1 = folium.Choropleth(

    geo_data=geo_str,

    name='crime',

    data=crime_distribution,

    columns=['DISTRICT', 'INCIDENT_NUMBER'],

    key_on='properties.DISTRICT',

    fill_color='OrRd',

    fill_opacity=0.6,

    line_opacity=0.6,

    legend_name='Crime Rate (%)'

).add_to(crime_map)





m2 = folium.Choropleth(

    geo_data=geo_str,

    name='officer',

    data=specialist_dist,

    columns=['DISTRICT', 'specialist'],

    key_on='properties.DISTRICT',

    fill_color='OrRd',

    fill_opacity=0.6,

    line_opacity=0.6,

    legend_name='officer allocation rate'

).add_to(crime_map)



m3 = folium.Choropleth(

    geo_data=geo_str,

    name='specialist',

    data=officer_dist,

    columns=['DISTRICT', 'office'],

    key_on='properties.DISTRICT',

    fill_color='OrRd',

    fill_opacity=0.6,

    line_opacity=0.6,

    legend_name='specialist allocation rate'

).add_to(crime_map)



crime_map.add_child(m1).add_child(m2).add_child(m3)

folium.LayerControl().add_to(crime_map)



crime_map.save('folium_chloropleth_country.html')



# crime_map = folium.Map(location=[42.3125,-71.0875], 

#                       tiles = "Stamen Toner",

#                       zoom_start = 12)





# folium.Choropleth(

#     geo_data=geo_str,

#     name='choropleth',

#     data=crime_distribution,

#     columns=['DISTRICT', 'INCIDENT_NUMBER'],

#     key_on='properties.DISTRICT',

#     fill_color='OrRd',

#     fill_opacity=0.6,

#     line_opacity=0.6,

#     legend_name='Crime Rate (%)'

# ).add_to(crime_map)



# folium.LayerControl().add_to(crime_map)



# crime_map.save('folium_chloropleth_country2.html')

# state_data = pd.read_csv('/kaggle/input/distributiond/sensitivity_specialist.csv', encoding='latin-1')

# state_data['DISTRICT'] = state_data['Unnamed: 0']

# state_data.drop('Unnamed: 0', inplace=True, axis= 1)
# import os

# import pandas as pd

# import json

# import geopandas as gpd



# gdf = gpd.read_file('/kaggle/input/geo-json-boston/Police_Districts.geojson')

# merged = gdf.merge(state_data, left_on='DISTRICT', right_on='DISTRICT')

# spatial_gdf = gpd.GeoDataFrame(merged.iloc[:, [1, 8]])

# merged
# styledata = {}

# #merged['opacity'] = 0.5

# for i in merged.index:

#     df = pd.DataFrame(

#      {'color': merged.iloc[i,9:18].transpose().values,

#       'opacity': [0.6]*9}, index=[p * 1000 for p in range(1,10)])

#     #df.index.name = 'speciality_resource'

#     styledata[i] = df.rename({str(i):'color'}, axis=1)

    

# # for i in range(1,10):

# #     df = merged[[str(i),'opacity']]

# #     df.index.name = "speciality_resource"

# #     styledata[i] = df.rename({str(i):'color'}, axis=1)
# styledata.get(0)
# from branca.colormap import linear



# max_color, min_color = 0, 0



# for time, stydat in styledata.items():

#     max_color = max(max_color, stydat['color'].max())

#     min_color = min(min_color, stydat['color'].min())

        

# cmap = linear.BuPu_06.scale(min_color, max_color)



# for country, stydat in styledata.items():

#     stydat['color'] = stydat['color'].apply(cmap)

#     stydat['opacity'] = (stydat['opacity'])
# stydat
# styledict = {

#     str(record): stydat.to_dict(orient='index') for

#     record, stydat in styledata.items()

# }
# from folium.plugins import TimeSliderChoropleth

# crime_map = folium.Map(location=[42.3125,-71.0875], 

#                       tiles = "Stamen Toner",

#                       zoom_start = 12)



# g = TimeSliderChoropleth(

#     spatial_gdf.to_json(),

#     styledict=styledict, overlay = True

# ).add_to(crime_map)

# folium.LayerControl().add_to(crime_map)



# # crime_map.choropleth(

# #     geo_data=geo_str,

# #     name='choropleth',

# #     data=crime_distribution,

# #     columns=['DISTRICT', 'INCIDENT_NUMBER'],

# #     key_on='properties.DISTRICT',

# #     fill_color='BuPu',

# #     fill_opacity=0.0,

# #     line_opacity=0.0,

# #     legend_name='Crime Rate (%)'

# # )



# # folium.LayerControl().add_to(crime_map)



# crime_map

# crime_map.save('folium_dynamic_chloropleth.html')
# import geopandas as gpd





# assert 'naturalearth_lowres' in gpd.datasets.available

# datapath = gpd.datasets.get_path('naturalearth_lowres')

# gdf = gpd.read_file(datapath)

# %matplotlib inline



# ax = gdf.plot(figsize=(10, 10))



# import pandas as pd





# n_periods, n_sample = 48, 40



# assert n_sample < n_periods



# datetime_index = pd.date_range('2016-1-1', periods=n_periods, freq='M')

# dt_index_epochs = datetime_index.astype(int) // 10**9

# dt_index = dt_index_epochs.astype('U10')



# dt_index

# styledata = {}



# for country in gdf.index:

#     df = pd.DataFrame(

#         {'color': np.random.normal(size=n_periods),

#          'opacity': np.random.normal(size=n_periods)},

#         index=dt_index

#     )

#     df = df.cumsum()

#     df.sample(n_sample, replace=False).sort_index()

#     styledata[country] = df

    

# max_color, min_color, max_opacity, min_opacity = 0, 0, 0, 0



# for country, data in styledata.items():

#     max_color = max(max_color, data['color'].max())

#     min_color = min(max_color, data['color'].min())

#     max_opacity = max(max_color, data['opacity'].max())

#     max_opacity = min(max_color, data['opacity'].max())

    

# from branca.colormap import linear





# cmap = linear.PuRd_09.scale(min_color, max_color)





# def norm(x):

#     return (x - x.min()) / (x.max() - x.min())





# for country, data in styledata.items():

#     data['color'] = data['color'].apply(cmap)

#     data['opacity'] = norm(data['opacity'])

# styledict = {

#     str(country): data.to_dict(orient='index') for

#     country, data in styledata.items()

# }



# m = folium.Map([0, 0], tiles='Stamen Toner', zoom_start=2)



# g = TimeSliderChoropleth(

#     gdf.to_json(),

#     styledict=styledict,



# ).add_to(m)



# #m.save(os.path.join('results', 'TimeSliderChoropleth.html'))



# m
import pandas as pd

sensitivity_specialist = pd.read_csv("../input/sensitivity_specialist.csv")
import pandas as pd

sensitivity_specialist = pd.read_csv("../input/sensitivity_specialist.csv")