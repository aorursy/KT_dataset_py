# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import folium

import json

import os



pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)

%config InlineBackend.figure_format = 'svg'
# Loading data



data1 = pd.read_csv('../input/conventional-power-plants-in-europe/conventional_power_plants_EU.csv')

data2 = pd.read_csv('../input/conventional-power-plants-in-europe/conventional_power_plants_DE.csv')

data2.rename(columns = {'name_bnetza': 'name', 'country_code': 'country', 

                        'capacity_net_bnetza': 'capacity', 'fuel': 'energy_source'}, inplace = True)



# I want to use just these columns for analysis

cols = ['name', 'company', 'city', 'country', 'capacity', 'energy_source', 'technology', 'chp', 

             'commissioned', 'type', 'lat', 'lon', 'energy_source_level_1', 'energy_source_level_2',

             'energy_source_level_3']



data1 = data1[cols]

data2 = data2[cols]



data = pd.concat([data1, data2])



data['country'].replace({'AT': 'Austria', 'CH': 'Switzerland', 'CZ': 'Czech',

                        'DE': 'Germany', 'DK ': 'Denmark', 'ES': 'Spain', 'FI': 'Finland',

                        'FR': 'France', 'IT': 'Italy', 'LU': 'Luxembourg', 'NL': 'Netherlands',

                        'NO': 'Norway', 'PL': 'Poland', 'SE': 'Sweden', 'SI': 'Slovenia',

                        'SK': 'Slovakia', 'UK': 'United Kingdom'}, inplace = True)
data.head()
data.shape
# Countries in dataset

countries = data['country'].unique()

print(countries)

print(f'Total number of countries: {len(countries)}')
# Number of power plants in ech country

other = ['Luxembourg', 'Czech', 'Netherlands', 'Denmark', 'Norway', 'Sweden', 'Slovakia']

countries = data['country'].replace(other, ['Other']*len(other)).value_counts()

percent = np.round((countries.values / data.shape[0]) * 100, 2)

explode = [0.1]*countries.shape[0]



fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"), figsize = (7, 7))

wedges, text = plt.pie(countries.values, startangle = 50, wedgeprops=dict(width=0.5), explode = explode)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")



for i, p in enumerate(wedges):

    ang = (p.theta2 - p.theta1)/2. + p.theta1

    y = np.sin(np.deg2rad(ang))

    x = np.cos(np.deg2rad(ang))

    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

    connectionstyle = "angle,angleA=0,angleB={}".format(ang)

    kw["arrowprops"].update({"connectionstyle": connectionstyle})



    ax.annotate(f'{countries.index[i]}: {countries.values[i]} ({percent[i]}%)', xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),

                horizontalalignment=horizontalalignment, **kw)



plt.title('Number of power plants in EU countries')

plt.show()



o = data['country'].value_counts().loc[other]

print(f'Note: "Other" category includes next countries and number of power plants:')

for i in range(o.shape[0]):

    print(f'- {o.index[i]}: {o.values[i]} units')

# Capacity pie chart

other = ['Luxembourg', 'Slovenia', 'Denmark', 'Slovakia', 'Czech']

cap_data = data[['country', 'capacity']].replace(other, ['Other']*len(other)).groupby('country').sum().sort_values(by = 'capacity', ascending = False)

percent = np.round((cap_data['capacity'] / data['capacity'].sum()) * 100, 2)

explode = [0.1]*cap_data.shape[0]



fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"), figsize = (6, 6))

wedges, text = plt.pie(cap_data['capacity'], startangle = 60, wedgeprops=dict(width=0.5), explode = explode)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")



for i, p in enumerate(wedges):

    ang = (p.theta2 - p.theta1)/2. + p.theta1

    y = np.sin(np.deg2rad(ang))

    x = np.cos(np.deg2rad(ang))

    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

    connectionstyle = "angle,angleA=0,angleB={}".format(ang)

    kw["arrowprops"].update({"connectionstyle": connectionstyle})    

    

    ax.annotate(f'{cap_data.index[i]}: {int(cap_data["capacity"][i])} ({percent[i]}%)', xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),

                horizontalalignment=horizontalalignment, **kw)



plt.title('Power plants capacity by country')

plt.show()



o = data[['country', 'capacity']].groupby('country').sum().sort_values(by = 'capacity').loc[other]

print(f'Note: "Other" category includes next countries and total capacity of power plants:')

for i in range(o.shape[0]):

    print(f'- {o.index[i]}: {int(o.capacity[i])} MW')

# Data for choropleth map

cap = data[['country', 'capacity']].groupby('country', as_index = False).sum().copy()



# Choropleth with folium

f = folium.Figure(width=550, height=800)

m = folium.Map(location=[58.38811, 10.21540], zoom_start = 4,).add_to(f)



geo = '../input/conventional-power-plants-in-europe/geo_med_eu.json'



folium.Choropleth(

    geo_data=geo, # json file with geodata  

    data = cap, # Pandas dataframe or Series

    columns  = ['country', 'capacity'], # Columns from Dataframe

    key_on = 'feature.properties.name', # Where in json file search countries

    fill_opacity=0.8,

    nan_fill_opacity = 0.1,

    line_opacity=0.4,

    fill_color='OrRd',    

    bins = [0, 20000, 40000, 60000, 80000, 100000, 120000],

    legend_name = 'Capacities of conventional power plants in EU countries',

    name = 'capacities',

    highlight = True

).add_to(m)



m
print(f'Total number of companies on market: {data["company"].unique().shape[0]}')
# Company analysis

top_15 = data['company'].value_counts()[1:16]

top_15_cap = data[['company', 'capacity']].groupby('company', as_index = False).sum().sort_values(by = 'capacity', ascending = False)[1:16]



fig = plt.figure(figsize = (8, 8))

fig.add_subplot('211')

sns.barplot(x = top_15.values, y = top_15.index).set_title('Companies with the most power plants')



fig.add_subplot('212')

sns.barplot(x = 'capacity', y = 'company', data = top_15_cap).set_title('Power plants capacities by company')

plt.show()
other = ['Other fossil fuels', 'Other bioenergy and renewable waste', 'Mixed fossil fuels',

         'Non-renewable waste', 'Waste', 'Other or unspecified energy sources']

en_source = data['energy_source'].replace(other, ['Other fuels']*len(other)).value_counts()



percent = np.round((en_source.values / data.shape[0]) * 100, 2)

explode = [0.1]*en_source.shape[0]





fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"), figsize = (6, 6))

wedges, text = plt.pie(en_source.values, startangle = 30, wedgeprops=dict(width=0.5), explode = explode)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")



for i, p in enumerate(wedges):

    ang = (p.theta2 - p.theta1)/2. + p.theta1

    y = np.sin(np.deg2rad(ang))

    x = np.cos(np.deg2rad(ang))

    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

    connectionstyle = "angle,angleA=0,angleB={}".format(ang)

    kw["arrowprops"].update({"connectionstyle": connectionstyle})



    ax.annotate(f'{en_source.index[i]}: {en_source.values[i]} ({percent[i]}%)', xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),

                horizontalalignment=horizontalalignment, **kw)



plt.title('Power plants by energy source')

plt.show()



o = data['energy_source'].value_counts().loc[other]

print(f'Note: "Other" category includes next types of power plants:')

for i in range(o.shape[0]):

    print(f'- {o.index[i]}: {o.values[i]} units')
other = ['Other fossil fuels', 'Other bioenergy and renewable waste', 'Mixed fossil fuels', 'Non-renewable waste', 'Waste', 'Other fuels', 'Other or unspecified energy sources']



type_cap = data[['energy_source', 'capacity']].replace(other, ['Other']*len(other)).groupby('energy_source').sum().sort_values(by = 'capacity', ascending = False)



percent = np.round((type_cap['capacity'] / data['capacity'].sum()) * 100, 2)

explode = [0.1]*type_cap.shape[0]



fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"), figsize = (6, 6))

wedges, text = plt.pie(type_cap['capacity'], startangle = 20, wedgeprops=dict(width=0.5), explode = explode)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")



for i, p in enumerate(wedges):

    ang = (p.theta2 - p.theta1)/2. + p.theta1

    y = np.sin(np.deg2rad(ang))

    x = np.cos(np.deg2rad(ang))

    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

    connectionstyle = "angle,angleA=0,angleB={}".format(ang)

    kw["arrowprops"].update({"connectionstyle": connectionstyle})



    ax.annotate(f'{type_cap.index[i]}: {int(type_cap["capacity"][i])} ({percent[i]}%)', xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),

                horizontalalignment=horizontalalignment, **kw)



plt.title('Capacities of power plants by energy source')

plt.show()



o = data['energy_source'].value_counts().loc[other]

print(f'Note: "Other" category includes next types of power plants ant their capacities:')

for i in range(o.shape[0]):

    print(f'- {o.index[i]}: {o.values[i]} MW')
other = ['Other bioenergy and renewable waste', 'Other bioenergy and renewable waste', 'Other fuels', 

         'Other fossil fuels', 'Mixed fossil fuels', 'Other or unspecified energy sources']

# colors = ['peru', 'gray', 'black', 'cornflowerblue', 'purple', 'blue', 'pink', 'yellow', 'olive', 'cyan', 'teal']

cross_country = pd.crosstab(data['energy_source'].replace(other, ['Other']*len(other)), data['country'])

cross_country = cross_country.apply(lambda x: x / x.sum())



fig = plt.figure(figsize = (9, 5))

ax = fig.add_subplot('111')

cross_country.transpose().plot(kind = 'bar', stacked = True, ax = ax, width = 0.8, colormap = 'tab20b')

vals = ax.get_yticks()

ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

ax.yaxis.grid(True)

ax.set_axisbelow(True)

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

ax.set_title('Proportions of different power plants types in EU countries')

plt.show()
other = ['Other bioenergy and renewable waste', 'Other bioenergy and renewable waste', 'Other fuels', 

         'Other fossil fuels', 'Mixed fossil fuels', 'Other or unspecified energy sources']

# colors = ['peru', 'gray', 'black', 'cornflowerblue', 'purple', 'blue', 'pink', 'yellow', 'olive', 'cyan', 'teal']

cross_country = pd.crosstab(data['energy_source'].replace(other, ['Other']*len(other)), data['country'], values = data['capacity'], aggfunc = 'sum')

cross_country = cross_country.apply(lambda x: x / x.sum())



fig = plt.figure(figsize = (9, 5))

ax = fig.add_subplot('111')

cross_country.transpose().plot(kind = 'bar', stacked = True, ax = ax, width = 0.8, colormap = 'tab20b')

vals = ax.get_yticks()

ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

ax.yaxis.grid(True)

ax.set_axisbelow(True)

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

ax.set_title('Capacity propotions by power plant type')

plt.show()