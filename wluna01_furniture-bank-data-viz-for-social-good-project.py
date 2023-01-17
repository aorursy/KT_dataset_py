# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd

import json



from bokeh.io import output_notebook, show

from bokeh.plotting import figure, ColumnDataSource, show

from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, HoverTool, BoxZoomTool, ZoomOutTool, ResetTool, Legend, LegendItem

from bokeh.palettes import brewer, Spectral4





import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Load mapping data

mapping_filepath = "../input/furniture-bank/MAPPING DATA.csv"

mapping = pd.read_csv(mapping_filepath, index_col=0, encoding="latin-1")

mapping.columns = mapping.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

mapping = mapping[mapping['mailing_city'] == 'Toronto']
#Load data

coordinate_filepath = "../input/furniture-bank/ONTARIO COORDINATE FILE.csv"

coordinate = pd.read_csv(coordinate_filepath, index_col=0, encoding="latin-1")



#clean up column names

coordinate.columns = coordinate.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

toronto_coordinates = coordinate[coordinate.place_name == 'Toronto']



#standardize formatting in preparation for a join

mapping['postalcode'] = mapping['postalcode'].str.strip().str.upper().str.replace(' ', '')

coordinate.index = coordinate.index.str.strip().str.upper()
mapping.head()
mapping['in_kind_value_tax_receipt_value'] = mapping['in_kind_value_tax_receipt_value'].replace('[\$,]', '', regex=True).astype(float)
mapping = mapping.filter(['account_name', 'type', 'in_kind_value_tax_receipt_value','postalcode'])

mapping.sample()
mapping['in_kind_value_tax_receipt_value'] = mapping['in_kind_value_tax_receipt_value']

tax_mappings = pd.DataFrame(mapping.groupby(['postalcode','type'])['in_kind_value_tax_receipt_value'].sum())
#merge donor/client tax data on location, then filter

tax_coordinates = coordinate.reset_index()

tax_mappings = tax_mappings.reset_index()



donation_mappings = pd.merge(tax_mappings, 

                  coordinate,

                  left_on='postalcode',

                 right_on='PostalCode', 

                   how='inner')

donation_mappings = donation_mappings.filter(['postalcode','type','in_kind_value_tax_receipt_value', 'latitude','longitude'])

donation_mappings = donation_mappings.rename(columns = {'in_kind_value_tax_receipt_value':'sumtotal'})
donation_mappings = donation_mappings.sort_values(by=['postalcode','type'])

donation_mappings.sample(5)
#merge

merged_mapping = pd.merge(mapping, 

                  coordinate,

                  left_on='postalcode',

                 right_on='PostalCode')
#create pivot table of number of locations within a zipcode

fsa_counts = pd.DataFrame(merged_mapping['fsa'].value_counts().reset_index())

fsa_counts.columns = ['zip', 'count']



#remove all zip codes that have under 10 donations

removed_nums = list(range(100))

fsa_counts = fsa_counts[~fsa_counts['count'].isin(removed_nums)]

merged_mapping = merged_mapping.filter(['fsa', 'latitude', 'longitude'])

select_toronto_zipcodes = fsa_counts.zip
donation_locations = merged_mapping.filter(['fsa', 'latitude', 'longitude'])

donation_locations = donation_locations[donation_locations['fsa'].isin(select_toronto_zipcodes)]
donation_locations.count()
latitude = donation_locations.latitude

longitude = donation_locations.longitude
#import geojson map of Ontario

ontario_map = gpd.read_file('../input/ontariofsa/Ontario_FSAs.geojson')

ontario_geometry = ontario_map.filter(['CFSAUID','geometry'])



#run a merge so that each row will also have the total number of donations in each zip code

ontario_geometry = pd.merge(ontario_geometry, 

                  fsa_counts,

                  left_on='CFSAUID',

                 right_on='zip')



#remove unecessary columns

ontario_geometry = ontario_geometry.filter(['geometry','zip','count'])
ontario_geometry.head()
ontario_json = json.loads(ontario_geometry.to_json())

json_data = json.dumps(ontario_json)

geosource = GeoJSONDataSource(geojson = json_data)
#Define a sequential multi-hue color palette.

palette = brewer['Blues'][8]

#Reverse color order so that dark blue is highest obesity.

palette = palette[::-1]

#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.

color_mapper = LinearColorMapper(palette = palette, low = (fsa_counts['count'].min()), high = (fsa_counts['count'].max()))



#choropleth map attempt

hover = HoverTool(tooltips = [ ('number of donations', '@count'), ('zipcode','@zip')])

p = figure(title="Number of items donated in Toronto by zipcode",tools=[hover])

p.patches('xs','ys', source = geosource,fill_color = {'field' :'count', 'transform' : color_mapper},

          line_color = 'black', line_width = 0.25, fill_alpha = 1)

p.add_tools(BoxZoomTool(), ZoomOutTool(), ResetTool())

#p.circle(x=longitude,y=latitude,size=1, color="#410000", alpha=0.5)

output_notebook()

show(p)
cutoff = donation_mappings["sumtotal"].quantile(0.99)

donation_mappings = donation_mappings[donation_mappings["sumtotal"] < cutoff]

sumtotal = donation_mappings['sumtotal']
new_donation_mappings = donation_mappings.replace('In Kind Donation', '#315902')

new_donation_mappings = new_donation_mappings.replace('Delivery', '#410000')

new_sumtotal = sumtotal.replace()



longitude = donation_mappings['longitude']

latitude = donation_mappings['latitude']

donation_type = donation_mappings['type']

donation_color = new_donation_mappings['type']
longitude = list(longitude)

latitude = list(latitude)

sumtotal = list(sumtotal)

donation_color = list(donation_color)

donation_type = list(donation_type)



longitude = list(map(float, longitude))

latitude = list(map(float, latitude))

sumtotal = list(map(float, sumtotal))

divisor = 450

new_sumtotal = [x / divisor for x in sumtotal]
#Define a sequential multi-hue color palette.

#palette = 'FFFCFC'

#Reverse color order so that dark blue is highest obesity.

#palette = palette[::-1]

#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.

#color_mapper = LinearColorMapper(palette = palette, low = (fsa_counts['count'].min()), high = (fsa_counts['count'].max()))



#choropleth map attempt

#hover = HoverTool(tooltips = [ ('donation type', donation_color), ('something here')])

location_source = ColumnDataSource(data=dict(

    x = longitude,

    y = latitude,

    total = sumtotal,

    size = new_sumtotal,

    color = donation_color,

    dtype = donation_type

))

hover = HoverTool(tooltips = [("donation type", "@dtype"),

                          ("total amount donated", "@size")])



#TOOLTIPS = [

 #   ("dtype", "$dtype"),

#]



p = figure(title="Furniture Donations in Toronto (drag box to zoom)", 

           x_range=[-79.7,-79.1], y_range=[43.55,43.875])

p.add_tools(hover)

p.patches('xs','ys', color='white', source = geosource,

          line_color = 'black', line_width = 0.25, fill_alpha = 1)

#p.add_tools(BoxZoomTool(), ZoomOutTool(), ResetTool())

p.circle(x='x',y='y',size='size', color='color', alpha=0.5, source=location_source, hover_fill_color = 'black')

output_notebook()

show(p)