# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

import folium

import difflib 

import geopandas as gpd

import math 
# Reads in the Covid-19 data

df_covid19 = pd.read_csv("/kaggle/input/bangladesh-district-level-covid19-dataset/BD_COVID_19_Time_Series_Data.csv")

lowercase = df_covid19["DISTRICT"].map(lambda x: x.title())

df_covid19 = df_covid19.set_index(lowercase)

df_covid19.head(3)
total_confirmedcases = df_covid19['TOTAL'].sum()

last_update = df_covid19.columns[-1]

print(f'Total number of confirmed cases (as of {last_update} AM):', total_confirmedcases)
# Reads in the district coords data

district_coords = pd.read_csv('/kaggle/input/bangladesh-administrative-maps-geocoordinates/districts.csv')

lowercase = district_coords["name"].map(lambda x: x.title())

district_coords = district_coords.set_index(lowercase)

district_coords.head(3)
# Corrects for small variations in English spelling of district names between data frames (e.g Comilla and Cumilla) in preparation for joining

district_coords.index = district_coords.index.map(lambda x: difflib.get_close_matches(x, df_covid19.index)[0])

district_coords.head(3)
# Combines the data frames of Covid dataset and district coord dataset

df_districts_cv19 = df_covid19.join(district_coords, how = 'outer') 

df_districts_cv19.head(3)
#Accounts for the difference in name between Chittagong and Chattogram in different data sets and checks output

#(due to official English renaming of the city from Chittagong to Chattogram in 2018)

df_districts_cv19['geoname'] = df_districts_cv19['name']

df_districts_cv19.loc[df_districts_cv19['DIST_CODE']==15, 'geoname'] = 'Chittagong'

df_districts_cv19[df_districts_cv19['DIST_CODE']==15]
#Take log of number of cases 

df_districts_cv19['logtotal'] = df_districts_cv19['TOTAL'].apply(lambda x: math.log10(x) if x > 0 else 1E-2)  
#Shapefiles for the divisional and district boundaries for mapping 

district_geodata = gpd.read_file("/kaggle/input/bangladesh-administrative-boundaries-shapefiles/bgd_admbnda_adm2_bbs_20180410.shp")

division_geodata = gpd.read_file("/kaggle/input/bangladesh-administrative-boundaries-shapefiles/bgd_admbnda_adm1_bbs_20180410.shp")
#Matches the name of districts in the shapefile with the Covid-19 dataframe for combining datasets later

df_districts_cv19['geoname'] = df_districts_cv19['geoname'].apply(lambda x: difflib.get_close_matches(x, district_geodata['ADM2_EN'])[0])

df_districts_cv19['logtotal'] = df_districts_cv19['TOTAL'].apply(lambda x: math.log10(x) if x > 0 else 1E-2)
#Create a datframe of divisional Covid-19 data and coords

df_divisions = df_districts_cv19.groupby(by = 'DIVISION').sum()

df_divisions['lat'] = df_districts_cv19.groupby(by = 'DIVISION')['lat'].mean()

df_divisions['lon'] = df_districts_cv19.groupby(by = 'DIVISION')['lon'].mean()

df_divisions['division_id'] = df_districts_cv19.groupby(by = 'DIVISION')['division_id'].mean()

df_divisions['logtotal'] = df_divisions['TOTAL'].apply(lambda x: math.log10(x) if x > 0 else 1E-2)  

df_divisions['geoname'] = df_divisions.index

df_divisions['geoname'] = df_divisions['geoname'].apply(lambda x: x.title())

df_divisions.loc[df_divisions['division_id']==1, 'geoname'] = 'Chittagong'

df_divisions
#Create map of district and divisional case numbers which will include clorepleth and markers layered for each 



m = folium.Map(

    location=[np.median(df_districts_cv19['lat']),np.median(df_districts_cv19['lon'])],

    zoom_start=7, 

)



layer1=folium.FeatureGroup(name='District Markers', show=False)

layer2=folium.FeatureGroup(name='Division Markers', show=False)



m.add_child(layer1)

m.add_child(layer2)



for index,rows in df_districts_cv19.iterrows():

    if rows['lat'] and rows['lon']:

        folium.Marker([rows['lat'], rows['lon']], popup='District: '+(str(rows['name']))+'\n\nConfirmed cases: '+str(rows['TOTAL']), 

                      icon=folium.Icon(color='orange', icon='info-sign')                     

                     ).add_to(layer1)



for index,rows in df_divisions.iterrows():

    if rows['lat'] and rows['lon']:

        folium.Marker([rows['lat'], rows['lon']], popup='Division: '+(str(rows['geoname']))+'\n\nConfirmed cases: '+str(rows['TOTAL']), 

                      icon=folium.Icon(color='darkpurple', icon='info-sign')                     

                     ).add_to(layer2)

        

folium.Choropleth(

    geo_data = district_geodata,

    name='District Chloropleth',

    data=df_districts_cv19,

    columns=['geoname', 'logtotal'],

    key_on='feature.properties.ADM2_EN',

    legend_name='Confirmed Cases of Covid-19 [log]',

    fill_color='Oranges',

    highlight = True, 

    #bins = [0.0001, math.log10(10),math.log10(25),math.log10(50),math.log10(100),math.log10(200),math.log10(400),math.log10(1000),math.log10(10000),math.log10(50000)]    

).add_to(m)



folium.Choropleth(

    geo_data = division_geodata,

    name='Division Chloropleth',

    data=df_divisions,

    columns=['geoname', 'logtotal'],

    key_on='feature.properties.ADM1_EN',

    legend_name='Confirmed Cases of Covid-19 [log]',

    fill_color='Purples',

    highlight = True, 

    #bins = [0.0001, math.log10(10),math.log10(25),math.log10(50),math.log10(100),math.log10(200),math.log10(500),math.log10(1000),math.log10(5000),math.log10(10000),math.log10(15000)]    

).add_to(m)



folium.LayerControl(collapsed = False).add_to(m)



m.save('Covid19_GeographicDistribution_BD.html')

m
plt.figure(figsize = (12,6))

df_districts_cv19['logtotal'].plot.hist(bins = 20)

plt.xlabel('District wise mumber of cases [$log_{10}(N)$]', fontsize = 20)

plt.xlim(0,4)
plt.figure(figsize = (12,6))

df_divisions['logtotal'].plot.hist(bins = 20)

plt.xlabel('Division wise mumber of cases [$log_{10}(N)$]', fontsize = 20)

plt.xlim(0,4)
#plt.figure(figsize = (6,6))

g = sns.FacetGrid(df_districts_cv19,col='DIVISION')

g = g.map(plt.hist,'logtotal', bins = 10)

for i in range(0,8):

    g.axes[0,i].set_xlabel('District wise mumber of cases [$log_{10}(N)$]', fontsize = 10)