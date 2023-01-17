# This Python 3 environment comes with many helpful analytics libraries installed



import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



#The packages below are used for making choropleth maps.

import geopandas as gpd

import json

from bokeh.io import output_notebook, show, output_file

from bokeh.plotting import figure

from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar

from bokeh.palettes import brewer



# Input data files are available in the read-only "../input/" directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/cee-498-project10/aquastat.csv', header=0)

df.head()
df = df.rename(columns={"Unnamed: 0": "country", "Unnamed: 1": "variable"})

colnames = ['country','variable','1978-1982', '1983-1987', '1988-1992', '1993-1997', '1998-2002', '2003-2007', '2008-2012', '2013-2017']

df = df[colnames]

df = df.dropna(subset=['variable'])

df[['1978-1982', '1983-1987', '1988-1992', '1993-1997', '1998-2002', '2003-2007', '2008-2012', '2013-2017']] = df[['1978-1982', '1983-1987', '1988-1992', '1993-1997', '1998-2002', '2003-2007', '2008-2012', '2013-2017']].astype('float64') 

df.reset_index(drop=True, inplace=True)

df.head()
print(df.shape)

print('-----------')

print(df.country.value_counts())

print('-----------')

print(df.variable.value_counts())
df_2017 = pd.pivot(df.copy(), values='2013-2017', index=['country'],

                    columns=['variable'])

df_2017 = df_2017.dropna(subset=['Total water withdrawal per capita (m3/inhab/year)'])

df_2017.reset_index(inplace = True)

df_2017
df_code = pd.read_csv('/kaggle/input/country-code/country_code.csv')

df_2017 = df_2017.merge(df_code, left_on = 'country', right_on = 'Short name', how = 'left')
shapefile = '/kaggle/input/natural-earth-1110m-countries/ne_110m_admin_0_countries.shp'

#Read shapefile using Geopandas

gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]

gdf.columns = ['country', 'country_code', 'geometry']

#Drop row corresponding to 'Antarctica'

gdf = gdf.drop(gdf.index[159])

gdf
#Merge dataframes gdf and df_2017

merged = gdf.merge(df_2017, left_on = 'country_code', right_on = 'ISO3', how = 'left')

merged.fillna('No data', inplace = True)

merged.head()
#Read data to json.

merged_json = json.loads(merged.to_json())

#Convert to String like object.

json_data = json.dumps(merged_json)



#Input GeoJSON source that contains features for plotting.

geosource = GeoJSONDataSource(geojson = json_data)



#Define a sequential multi-hue color palette.

palette = brewer['YlGnBu'][8]



#Reverse color order so that dark blue is highest obesity.

palette = palette[::-1]



#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.

color_mapper = LinearColorMapper(palette = palette, low = 0, high = 1600, nan_color = '#d9d9d9')



#Define custom tick labels for color bar.

tick_labels = {'0': '0', '200': '200', '400':'400', '600':'600', '800':'800', '1000':'1000','1200':'1200','1400':'1400', '1600': '>1600'}



#Create color bar. 

color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,

border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)



#Create figure object.

p = figure(title = 'Total water withdrawal per capita during 2013-2017 (m3/inhab/year)', plot_height = 600 , plot_width = 950, toolbar_location = None)

p.xgrid.grid_line_color = None

p.ygrid.grid_line_color = None



#Add patch renderer to figure. 

p.patches('xs','ys', source = geosource,fill_color = {'field' :'Total water withdrawal per capita (m3/inhab/year)', 'transform' : color_mapper},

          line_color = 'black', line_width = 0.25, fill_alpha = 1)

#Specify figure layout.

p.add_layout(color_bar, 'below')

#Display figure inline in Jupyter Notebook.

output_notebook()

#Display figure.

show(p)
df_2017['Total water withdrawal per capita (m3/inhab/year)'].hist() ;
#Read data to json

merged_json = json.loads(merged.to_json())

#Convert to String like object.

json_data = json.dumps(merged_json)



#Input GeoJSON source that contains features for plotting.

geosource = GeoJSONDataSource(geojson = json_data)



#Define a sequential multi-hue color palette.

palette = brewer['YlGnBu'][8]



#Reverse color order so that dark blue is highest obesity.

palette = palette[::-1]



#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.

color_mapper = LinearColorMapper(palette = palette, low = 0, high = 2000, nan_color = '#d9d9d9')



#Define custom tick labels for color bar.

tick_labels = {'0': '0', '250': '250', '500':'500', '750':'750', '1000':'1000', '1250':'1250','1500':'1500','1750':'1750', '2000': '>2000'}



#Create color bar. 

color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,

border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)



#Create figure object.

p = figure(title = 'Total renewable water resources (10^9 m3/year)', plot_height = 600 , plot_width = 950, toolbar_location = None)

p.xgrid.grid_line_color = None

p.ygrid.grid_line_color = None



#Add patch renderer to figure. 

p.patches('xs','ys', source = geosource,fill_color = {'field' :'Total renewable water resources (10^9 m3/year)', 'transform' : color_mapper},

          line_color = 'black', line_width = 0.25, fill_alpha = 1)

#Specify figure layout.

p.add_layout(color_bar, 'below')



#Display figure inline in Jupyter Notebook.

output_notebook()

#Display figure.

show(p)
df_2017.hist(figsize=(20,20));
corr = df_2017.corr()

corr = corr.apply(lambda x: round(x, 2))

fig, ax = plt.subplots(figsize = (8,8))



sns.heatmap(corr, vmin=0, vmax=1, cmap="Blues", linewidths=0.75, annot=True, ax = ax);
df_2017_a = df_2017.copy()[df_2017['Total water withdrawal per capita (m3/inhab/year)'] > 500]

df_2017_a.loc[:,"Total water withdrawal per capita"] = '>500 m3/inhab/year'

df_2017_b = df_2017.copy()[df_2017['Total water withdrawal per capita (m3/inhab/year)'] < 500]

df_2017_b.loc[:,"Total water withdrawal per capita"]  = '<500 m3/inhab/year'

df_2017_c = pd.concat([df_2017_a, df_2017_b])
fig, axes = plt.subplots(2,2,figsize = (14,14))

sns.boxplot(x = 'Total water withdrawal per capita', y = 'Dam capacity per capita (m3/inhab)', data = df_2017_c, ax= axes[1][1]);

sns.boxplot(x = 'Total water withdrawal per capita', y = 'Agricultural water withdrawal as % of total water withdrawal (%)', data = df_2017_c, ax= axes[0][0]); 

sns.boxplot(x = 'Total water withdrawal per capita', y = 'GDP per capita (current US$/inhab)', data = df_2017_c, ax= axes[1][0]); 

sns.boxplot(x = 'Total water withdrawal per capita', y = 'Cultivated area (arable land + permanent crops) (1000 ha)', data = df_2017_c, ax= axes[0][1]); 
df_m = df.copy()

df_m = df_m.fillna(float(-9999))

df_m = pd.melt(df_m, id_vars=['country','variable'], var_name='year', value_name='value')

df_m = pd.pivot_table(df_m, index = ['country', 'year'], columns = 'variable', values = 'value', aggfunc= np.sum)

df_m = df_m.reset_index()

df_m = df_m[df_m['Total water withdrawal per capita (m3/inhab/year)'] != -9999]

df_m = df_m.reset_index(drop = True)

df_m
df_m[df_m['Human Development Index (HDI) [highest = 1] (-)'] == -9999]['country'].count()
df_m[df_m['Total population with access to safe drinking-water (JMP) (%)'] == -9999]['country'].count()
df_m = df_m.replace(-9999, np.nan)

df_m = df_m.drop(columns = ['Human Development Index (HDI) [highest = 1] (-)','Total population with access to safe drinking-water (JMP) (%)']).dropna()

df_m