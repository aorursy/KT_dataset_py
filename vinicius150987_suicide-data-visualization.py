import numpy as np

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='white')

import warnings

warnings.filterwarnings('ignore')

bins = range(0,100,10)

%matplotlib inline

import os

print(os.listdir("../input"))
df_raw = pd.read_csv('../input/master.csv')
df_raw.head()
df_feature = df_raw.copy()
df_feature.head()
#Installing country_converter package

!pip install country_converter --upgrade
#Creating a Continent Column

import country_converter as coco

cc = coco.CountryConverter()

continent = np.array([])

for i in range(0, len(df_feature)):

    continent= np.append(continent, cc.convert(names=df_feature['country'][i], to='Continent' ))

df_feature['continent'] = pd.DataFrame(continent) 

df_feature.columns

df_feature = df_feature[['country', 'continent','year', 'sex', 'age', 'suicides_no', 'population',

       'suicides/100k pop', 'country-year', 'HDI for year',

       ' gdp_for_year ($) ', 'gdp_per_capita ($)', 'generation', ]]
#Deleting unnecessary columns

sns.heatmap(df_feature.isnull(), yticklabels=False, cbar=False, cmap='viridis')

df_feature = df_feature.drop(['country-year', 'HDI for year'], axis=1)
df = df_feature.copy() 
#Taking latitude and longitude

from geopy.geocoders import Nominatim

lat = np.array([])

lon = np.array([])

country = np.array([])

countries = df_feature.groupby('country')['country'].unique().sort_values()



for i in range(0, len(countries)):

    geolocator = Nominatim(user_agent='tito', timeout=100)

    location = geolocator.geocode(countries.index[i], timeout=100)

    lat = np.append(lat, location.latitude)

    lon = np.append(lon, location.longitude)

    country = np.append(country, countries.index[i])
#Importing Map

import folium

data = pd.DataFrame({

'lat':lat,

'lon':lon,

'name':country})

data.head()    



m = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2 , )

country_map = list(zip(data['name'].values, data['lat'].values, data['lon'].values))

# add features

for country_map in country_map:

    folium.Marker(

        location=[float(country_map[1]), float(country_map[2])],

        popup=folium.Popup(country_map[0], parse_html=True),

        icon=folium.Icon(icon='home')

    ).add_to(m)  
m
#Functions

def bar_chart(feature1, feature2):

    from matplotlib import cm

    total = df[feature2].sum()

    color = cm.inferno_r(np.linspace(.4,.8, 30))

    g = df.groupby(feature1)[feature2].sum().plot(kind='bar', figsize=(15,10), rot = 45, color= color)

    ax = g.axes

    for p in ax.patches:

     ax.annotate(f"{p.get_height() * 100 / total:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),

         ha='center', va='center', fontsize=20, color='gray', rotation=0, xytext=(0, 10),

         textcoords='offset points') 

    plt.grid(b=True, which='major', linestyle='--')

    plt.title('Suicides per {}'.format(feature1), fontsize=20)

    plt.xlabel('{}'.format(feature1), fontsize=20)

    plt.xticks(fontsize=20)

    plt.yticks(fontsize=20)

    plt.tight_layout()

    plt.ylabel('Quantity', fontsize=20)

        

def bar_chart_group(feature1, feature2, feature3):

    suic_sum_yr = pd.DataFrame(df[feature1].groupby([df[feature2],df[feature3]]).sum())

    suic_sum_yr = suic_sum_yr.reset_index().sort_index(by=feature1,ascending=False)

    most_cont_yr = suic_sum_yr

    plt.figure(figsize=(20,5))

    plt.title('Suicides per {} / {}'.format(feature2, feature3), fontsize=20)

    plt.xticks(rotation=45, fontsize=20)

    plt.yticks(fontsize=20)

    sns.set(font_scale=1)

    sns.barplot(y=feature1,x=feature3,hue=feature2,data=most_cont_yr,palette='viridis');

    plt.ylabel('Quantity', fontsize=20)

    plt.xlabel('{}'.format(feature3),fontsize=20)

    plt.tight_layout()

   

def bar_chart_continent(continent, continent_name, suicides, country):

    from matplotlib import cm

    df_new = df[df[continent] == continent_name]  

    total = df_new[suicides].sum()

    color = cm.inferno_r(np.linspace(.4,.8, 30))

    df_new.groupby(country)[suicides].sum().sort_values(ascending=False).plot(kind='bar', figsize=(20,10), rot = 90, color= color)

    plt.grid(b=True, which='major', linestyle='--')

    plt.title('Suicides per {}'.format(continent), fontsize=20)

    plt.xlabel('{}'.format(continent_name), fontsize=20)

    plt.tight_layout()

    plt.xticks(fontsize=20)

    plt.yticks(fontsize=20)

    plt.ylabel('Quantity', fontsize=20)    



def bar_chart_continent_mean(continent, continent_name, mean, country):

    from matplotlib import cm

    df_new = df[df[continent] == continent_name]  

    color = cm.inferno_r(np.linspace(.4,.8, 30))

    g = df_new.groupby(country)[mean].mean().sort_values(ascending=True).plot(kind='barh', figsize=(20,10), rot = 0, color= color)

    plt.grid(b=True, which='major', linestyle='--')

    plt.title('{} per {}'.format(mean,continent), fontsize=20)

    plt.xlabel('{}'.format(mean), fontsize=20)

    plt.tight_layout()

    plt.xticks(fontsize=20)

    plt.yticks(fontsize=20)

    plt.ylabel('{}'.format(continent_name), fontsize=20)

    



def bar_chart_continent_double(continent, continent_name, suicides, country, population):

    from matplotlib import cm

    df_new = df[df[continent] == continent_name]  

    color = cm.inferno_r(np.linspace(.4,.8, 30))

    fig = plt.figure()

    ax1 = fig.add_subplot(211)

    ax2 = fig.add_subplot(212)

    df_new.groupby(country)[suicides].sum().sort_values(ascending=False).plot(kind='bar', figsize=(80,60), rot = 90, color= color, ax=ax1)

    df_new.groupby(country)[population].sum().sort_values(ascending=False).plot(kind='bar', figsize=(80,60), rot = 90, color= color, ax= ax2)

    ax1.grid(b=True, which='major', linestyle='--')

    ax1.set_title('{} on {}'.format(suicides, continent_name), fontsize=60)

    ax1.set_ylabel('Quantity', fontsize=60)

    ax1.set_xlabel('{}'.format(country),fontsize=60)

    ax1.xaxis.set_tick_params(labelsize=60)

    ax1.yaxis.set_tick_params(labelsize=60)

    ax2.grid(b=True, which='major', linestyle='--')

    ax2.set_title('{} on {}'.format(population, continent_name), fontsize=60)

    ax2.set_ylabel('Quantity', fontsize=60)

    ax2.set_xlabel('{}'.format(country),fontsize=60)

    ax2.xaxis.set_tick_params(labelsize=60)

    ax2.yaxis.set_tick_params(labelsize=60)

    plt.tight_layout()

    
bar_chart('continent','suicides_no')
bar_chart('age','suicides_no')
bar_chart('sex','suicides_no')
bar_chart('generation','suicides_no')
bar_chart('year','suicides_no')
bar_chart_group('suicides_no', 'generation', 'year')
bar_chart_group('suicides_no', 'continent', 'year')
bar_chart_group('suicides_no', 'continent', 'sex')
bar_chart_group('suicides_no', 'generation', 'sex')
bar_chart_continent('continent', 'Europe', 'suicides_no', 'country')
bar_chart_continent('continent', 'America', 'suicides_no', 'country')
bar_chart_continent('continent', 'Africa', 'suicides_no', 'country')
bar_chart_continent('continent', 'Asia', 'suicides_no', 'country')
bar_chart_continent('continent', 'Oceania', 'suicides_no', 'country')
bar_chart_continent_mean('continent', 'Europe', 'suicides/100k pop', 'country')
bar_chart_continent_mean('continent', 'America', 'suicides/100k pop', 'country')
bar_chart_continent_mean('continent', 'Africa', 'suicides/100k pop', 'country')
bar_chart_continent_mean('continent', 'Asia', 'suicides/100k pop', 'country')
bar_chart_continent_mean('continent', 'Oceania', 'suicides/100k pop', 'country')
## Correlation with suicides_no 

df2 = df_feature.drop('suicides_no', axis=1)

df2.corrwith(df_feature['suicides_no']).plot.bar(

        figsize = (10, 10), title = "Correlation with suicides_no", fontsize = 15,

        rot = 45, grid = True)
bar_chart_continent_double('continent', 'Europe', 'suicides_no', 'country', 'population')
bar_chart_continent_double('continent', 'America', 'suicides_no', 'country', 'population')
bar_chart_continent_double('continent', 'Africa', 'suicides_no', 'country', 'population')
bar_chart_continent_double('continent', 'Asia', 'suicides_no', 'country', 'population')
bar_chart_continent_double('continent', 'Oceania', 'suicides_no', 'country', 'population')
bar_chart_continent_double('continent', 'Europe', 'suicides_no', 'country', 'suicides/100k pop')
bar_chart_continent_double('continent', 'America', 'suicides_no', 'country', 'suicides/100k pop')
bar_chart_continent_double('continent', 'Africa', 'suicides_no', 'country', 'suicides/100k pop')
bar_chart_continent_double('continent', 'Asia', 'suicides_no', 'country', 'suicides/100k pop')
bar_chart_continent_double('continent', 'Oceania', 'suicides_no', 'country', 'suicides/100k pop')