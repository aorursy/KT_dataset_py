# I'll ude bokeh for data visualisation

import pandas as pd

import pyproj

from bokeh.plotting import figure, show

from bokeh.tile_providers import get_provider, Vendors

from bokeh.io import output_notebook

from bokeh.models import ColumnDataSource



# Make bokeh show plot in jupyter notebook

output_notebook() 
# Reading our data

data = pd.read_csv('../input/Rus_prisons_coords.csv', encoding = 'windows-1251')

data.drop('Unnamed: 0', inplace = True, axis = 1)
# Preparing coordinates

lat = data['lat'].values

lon = data['lon'].values
# Transforming coordinates

project_projection = pyproj.Proj("+init=EPSG:4326")  # wgs84

google_projection = pyproj.Proj("+init=EPSG:3857")  # default google projection

x, y = pyproj.transform(project_projection, google_projection, lon, lat)



# Plotting data

p = figure(x_range=(2000000, 18000000), y_range=(6000000, 11000000),

            plot_width=800, plot_height=500, x_axis_type="mercator", y_axis_type="mercator")

p.add_tile(get_provider(Vendors.CARTODBPOSITRON_RETINA))

p.circle(x=x, y=y, size=3, fill_color="blue")

show(p)