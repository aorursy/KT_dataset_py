import pandas as pd
import numpy as np
import random
import sqlite3
import matplotlib.pyplot as plt
import folium
%matplotlib inline
import os
print(os.environ['PATH'])
# Connecting to sqlite database
cnx = sqlite3.connect('../input/world-development-indicators/database.sqlite')
cursor = cnx.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())
data = pd.read_sql_query("select * from Indicators;", cnx)
data.head()
print('Number of observation is ', len(data))
print('Number of distinct Indicator is', len(data['IndicatorCode'].unique()))
print('Range of indicators year is [', data['Year'].min(),',',data['Year'].max(),']')
print('Number of concerned country is ', len(data["CountryCode"].unique()))
series = pd.read_sql_query("select * from Series;", cnx)
series.head()
for tpc in series['Topic'].unique():
    print(tpc) 
series[series['Topic']=='Poverty: Income distribution']
# GINI index (World Bank estimate)
# SI.POV.GINI
print(series[series['Topic']=='Poverty: Income distribution'].loc[940, 'IndicatorName'])
print(series[series['Topic']=='Poverty: Income distribution'].loc[940, 'LongDefinition'])
series[series['Topic']=='Poverty: Poverty rates']
# Poverty gap at $1.90 a day (2011 PPP) (%)
# SI.POV.GAPS
print(series[series['Topic']=='Poverty: Poverty rates'].loc[948,'IndicatorName'])
print(series[series['Topic']=='Poverty: Poverty rates'].loc[948,'LongDefinition'])
# Poverty gap at national poverty lines (%)
# SI.POV.NAGP
print(series[series['Topic']=='Poverty: Poverty rates'].loc[950,'IndicatorName'])
print(series[series['Topic']=='Poverty: Poverty rates'].loc[950,'LongDefinition'])
series[series['Topic']=='Poverty: Shared prosperity']
# Annualized average growth rate in per capita real survey mean consumption or income, bottom 40% of population (%)
# SI.SPR.PC40.ZG
print(series[series['Topic']=='Poverty: Shared prosperity'].loc[958,'IndicatorName'])
print(series[series['Topic']=='Poverty: Shared prosperity'].loc[958,'LongDefinition'])
# Survey mean consumption or income per capita, bottom 40% of population (2011 PPP $ per day)
# SI.SPR.PC40
series[series['Topic']=='Poverty: Shared prosperity'].loc[961,'LongDefinition']
# # Create list of only countries
# df_countries = pd.DataFrame(columns=['CountryCode', 'CountryName'])
# with open('../input/country-alpha-codes/Country_Code_alpha3.txt') as f:
#     for line in f:
# #         print({'CountryCode' : line[:3], 'CountryName' : line[4:-2]})
#         df_countries = df_countries.append({'CountryCode' : line[:3], 'CountryName' : line[4:-2]}, ignore_index=True)
# df_countries.to_csv('../input/country-alpha-codes/Country_Code_alpha3.csv', index=False)  
# # df_countries.head(100)
# Get Countries code without groups of countries
onlycontries = pd.read_csv('../input/country-codes/Country_Code_alpha3.csv')
    
onlycontries.drop(['CountryName'], axis=1, inplace=True)
onlycontries.head()
print(len(data))
data = pd.merge(data, onlycontries, how='inner')
print(len(data))
print('Number of concerned country is ', len(data["CountryCode"].unique()))
gini = data[data['IndicatorCode']=='SI.POV.GINI']
print('Number of concerned country is ', len(gini["CountryCode"].unique()))
gini["CountryCode"].value_counts()
cntry = []
for y in range(1960,2015):
    cntry.append(gini[gini['Year']==y].shape[0])
print([(i+1960,j) for i, j in enumerate(cntry) ])
country_geo = '../input/python-folio-country-boundaries/world-countries.json'
# Setup a folium map at a high-level zoom @Alok - what is the 100,0, doesn't seem like lat long
map = folium.Map(location=[40, 20], zoom_start=1.5)
# choropleth maps bind Pandas Data Frames and json geometries.  This allows us to quickly visualize data combinations
map.choropleth(geo_data=country_geo, data=gini,
             columns=['CountryCode', 'Value'],
             key_on='feature.id',
             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,
             legend_name=gini.iloc[0]['IndicatorName'])
# Create Folium plot
map.save('plot_gini.html')

# Import the Folium interactive html file
from IPython.display import HTML
HTML('<iframe src=plot_gini.html width=900 height=550></iframe>')
Poverty_gap=data[data['IndicatorCode']=='SI.POV.GAPS']
print('Number of concerned country is ', len(Poverty_gap["CountryCode"].unique()))
Poverty_gap["CountryCode"].value_counts()
cntry = []
for y in range(1960,2015):
    cntry.append(Poverty_gap[Poverty_gap['Year']==y].shape[0])
print([(i+1960,j) for i, j in enumerate(cntry) ])
country_geo = '../input/python-folio-country-boundaries/world-countries.json'
# Setup a folium map at a high-level zoom @Alok - what is the 100,0, doesn't seem like lat long
map = folium.Map(location=[40, 20], zoom_start=1.5)
# choropleth maps bind Pandas Data Frames and json geometries.  This allows us to quickly visualize data combinations
map.choropleth(geo_data=country_geo, data=Poverty_gap,
             columns=['CountryCode', 'Value'],
             key_on='feature.id',
             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,
             legend_name=Poverty_gap.iloc[0]['IndicatorName'])
# Create Folium plot
map.save('plot_Poverty_gap.html')

# Import the Folium interactive html file
from IPython.display import HTML
HTML('<iframe src=plot_Poverty_gap.html width=900 height=550></iframe>')
#### National Poverty Gap
National_Poverty_Gap=data[data['IndicatorCode']=='SI.POV.NAGP']
print('Number of concerned country is ', len(National_Poverty_Gap["CountryCode"].unique()))
National_Poverty_Gap["CountryCode"].value_counts()
series[series['Topic']=='Economic Policy & Debt: National accounts: US$ at current prices: Aggregate indicators']
GDP_capita = data[(data['IndicatorCode'] == 'NY.GDP.PCAP.CD')]
GDP_capita.head()
print('year Range of GDP per capita is [', GDP_capita['Year'].min(),',',GDP_capita['Year'].max(),']')
top10_gdp_capita = GDP_capita[GDP_capita['Year']==2014].nlargest(10, 'Value')
ax = top10_gdp_capita.plot.bar(x='CountryName', y='Value', figsize=(10,5), legend=False)
ax.set_ylabel('GDP per capita (US$)')
ax.set_xlabel('Country Name')
top10_gdp_capita = GDP_capita[GDP_capita['Year']==2010].nlargest(10, 'Value')
ax = top10_gdp_capita.plot.bar(x='CountryName', y='Value', figsize=(8,5), legend=False)
ax.set_ylabel('GDP per capita (US$) 2010')
ax.set_xlabel('Country Name')
# plt.savefig('figures/gdp_capita_2010.png')
# Visualize GDP per capita using Folium
country_geo = '../input/python-folio-country-boundaries/world-countries.json'
# Setup a folium map at a high-level zoom @Alok - what is the 100,0, doesn't seem like lat long
map = folium.Map(location=[40, 20], zoom_start=1.5)
# choropleth maps bind Pandas Data Frames and json geometries.  This allows us to quickly visualize data combinations
map.choropleth(geo_data=country_geo, data=GDP_capita[GDP_capita['Year']==2014],
             columns=['CountryCode', 'Value'],
             key_on='feature.id',
             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,
             legend_name='GDP per capita, 2014')
# Create Folium plot
map.save('plot_gdp_capita.html')

# Import the Folium interactive html file
from IPython.display import HTML
HTML('<iframe src=plot_gdp_capita.html width=900 height=550></iframe>')
GDP = data[(data['IndicatorCode'] == 'NY.GDP.MKTP.CD')]
GDP.head()
top10_gdp = GDP[GDP['Year']==2014].nlargest(10, 'Value')
top10_gdp.head(10)
ax = top10_gdp.plot.bar(x='CountryName', y='Value', figsize=(10,5), legend=False)
ax.set_ylabel('GDP (US$)')
# Find Population indicator
series[series['IndicatorName'].str.contains('population')]
# Get the rural and urban population
Rural_pop = data[(data['IndicatorCode'] == 'SP.RUR.TOTL')]
Urban_pop = data[(data['IndicatorCode'] == 'SP.URB.TOTL')]
population = pd.merge(Rural_pop, Urban_pop, how='inner', left_on=['CountryCode', 'Year'], right_on=['CountryCode', 'Year'])
population['population'] = population['Value_x'] + population['Value_y']
population = population[['CountryName_x','CountryCode','Year','population']]
population.columns = ['CountryName','CountryCode','Year','population']
top10_population = population[population['Year']==2014].nlargest(10, 'population')
top10_population.head(10)
top10_population.plot.bar(x='CountryName', y='population', figsize=(10,5), legend=False)
series[series['Topic']=='Environment: Natural resources contribution to GDP']
natural_ress = data[data["IndicatorCode"]=='NY.GDP.TOTL.RT.ZS']
natural_ress.head()
top10_natural_ress = natural_ress[natural_ress['Year']==2010].nlargest(10, 'Value')
ax = top10_natural_ress.plot.bar(x='CountryName', y='Value', figsize=(8,6), legend=False)
ax.set_ylabel('Total natural resources rents (% of GDP)')
ax.set_xlabel('County Name')
# First, let's calculate the real value of the natural ressources in US$: GDP*Natural resources contribution to GDP/100
ress_value = pd.merge(GDP, natural_ress, how='inner', left_on=['CountryCode', 'Year'], right_on=['CountryCode', 'Year'])
ress_value['ress_value'] = ress_value['Value_x']*ress_value['Value_y']/100
ress_value = ress_value[['CountryName_x','CountryCode','Year','ress_value']]
ress_value.columns = ['CountryName','CountryCode','Year','ress_value']
ress_value.head()
# Visualize GDP per capita using Folium
country_geo = '../input/python-folio-country-boundaries/world-countries.json'
# Setup a folium map at a high-level zoom @Alok - what is the 100,0, doesn't seem like lat long
map = folium.Map(location=[40, 20], zoom_start=1.5)
# choropleth maps bind Pandas Data Frames and json geometries.  This allows us to quickly visualize data combinations
map.choropleth(geo_data=country_geo, data=ress_value[ress_value['Year']==2010],
             columns=['CountryCode', 'ress_value'],
             key_on='feature.id',
             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,
             legend_name='Natural ressource value (US$), 2010')
# Create Folium plot
map.save('Natural_ressource_value.html')

# Import the Folium interactive html file
from IPython.display import HTML
HTML('<iframe src=Natural_ressource_value.html width=900 height=550></iframe>')
# let's calculate now the natural ressource value per capita in US$
ress_capita = pd.merge(ress_value, population, how='inner', left_on=['CountryCode', 'Year'], right_on=['CountryCode', 'Year'])
ress_capita['ress_capita'] = ress_capita['ress_value']/ress_capita['population']
ress_capita = ress_capita[['CountryName_x','CountryCode','Year','ress_capita']]
ress_capita.columns = ['CountryName','CountryCode','Year','ress_capita']
ress_capita.head()
# Visualize GDP per capita using Folium
country_geo = '../input/python-folio-country-boundaries/world-countries.json'
# Setup a folium map at a high-level zoom @Alok - what is the 100,0, doesn't seem like lat long
map = folium.Map(location=[40, 20], zoom_start=1.5)
# choropleth maps bind Pandas Data Frames and json geometries.  This allows us to quickly visualize data combinations
map.choropleth(geo_data=country_geo, data=ress_capita[ress_capita['Year']==2010],
             columns=['CountryCode', 'ress_capita'],
             key_on='feature.id',
             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,
             legend_name='Natural ressource value per capita(US$), 2010')
# Create Folium plot
map.save('Natural_ressource_value_per_capita.html')

# Import the Folium interactive html file
from IPython.display import HTML
HTML('<iframe src=Natural_ressource_value_per_capita.html width=900 height=550></iframe>')
top10_ress_capita = ress_capita[ress_capita['Year']==2010].nlargest(10, 'ress_capita')
ax = top10_ress_capita.plot.bar(x='CountryName', y='ress_capita', figsize=(8,5), legend=False)
ax.set_xlabel('Country Name')
ax.set_ylabel('Natural Ressource Value per capita (US$)')
# plt.savefig('figures/nat_ress_capita_2010.png')
gini_gdp = pd.merge(gini, GDP_capita, how='inner', left_on=['CountryCode', 'Year'], right_on=['CountryCode', 'Year'])
gini_gdp.head()
gini_gdp[gini_gdp['Year'].between(2004, 2012, inclusive=True)][['Value_x', 'Value_y']].corr()
gdp_ress = pd.merge(GDP_capita, ress_capita, how='inner', left_on=['CountryCode', 'Year'], right_on=['CountryCode', 'Year'])
gdp_ress = gdp_ress[['CountryName_x', 'CountryCode', 'Year', 'Value', 'ress_capita']]
gdp_ress.columns = ['CountryName', 'CountryCode', 'Year', 'gdp_capita', 'ress_capita']
gdp_ress.head()
ax = gdp_ress[gdp_ress['Year']==2010].plot.scatter('gdp_capita', 'ress_capita', figsize=(15,15))
labels = gdp_ress[(gdp_ress['Year']==2010)&((gdp_ress['gdp_capita']>50000)|(gdp_ress['ress_capita']>1000))]
for index, row in labels.iterrows():
    ax.text(row['gdp_capita'], row['ress_capita'], row['CountryName'], fontsize='x-large', color='red')
ax.set_xlabel('GDP per capita (US$)', fontsize='x-large')
ax.set_ylabel('Natural ressource value per capita (US$)', fontsize='x-large')
gdp_ress[['gdp_capita','ress_capita']].corr()
print(gdp_ress['Year'].min(),gdp_ress['Year'].max())
gdp_ress.head()