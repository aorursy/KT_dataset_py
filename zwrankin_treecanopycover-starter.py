import numpy as np

import seaborn as sns

import folium

import geopandas as gpd

import rasterio

import affine

from shapely.geometry import Polygon

import plotly.graph_objects as go
src = rasterio.open('/kaggle/input/usfs-tree-canopy-cover/tcc_10km_resolution.tif')

# src = rasterio.open('/kaggle/input/usfs-tree-canopy-cover/tcc_1km_resolution.tif')  # If you want full resolution raster - MUCH slower

array = src.read(1)
def map_tcc(array: np.array, transform: affine.Affine):

    CENTER = (45, -100)

    m = folium.Map(location=CENTER,

                tiles='cartodbpositron',  # a "cleaner" base map

                    zoom_start=5)



    # Use the affine transformation to define the raster bounds in lat/long coords

    bounds_trans = (transform.c,

                transform.f + transform.e*array.shape[0],

                transform.c + transform.a*array.shape[1],

                transform.f)



    pal = sns.color_palette('Greens', n_colors=11)

    pal_rbg = []

    for t in pal:

        pal_rbg += [(t[0]*255, t[1]*255, t[2]*255)]

        

    def colorfun(x):

        if x > 100: 

            x = 0 # topcode since missingness is 255

        return pal_rbg[int(x/10)]



    image_overlay = folium.raster_layers.ImageOverlay(array,

                                                  [[bounds_trans[1],

                                                    bounds_trans[0]],

                                                   [bounds_trans[3],

                                                    bounds_trans[2]]], 

                                                  colormap=colorfun,

                                                  opacity=0.6, 

                                                  name='Tree Cover Canopy',

                                                  mercator_project=True)



    m.add_child(image_overlay)

    return m
m = map_tcc(array, src.transform)

m
def to_gpdf(array: np.array, transform: affine.Affine):

    """There's gotta be a better way of doing this"""

    assert len(array.shape) == 2, 'only 2D arrays supported'

    resolution = transform.a

    rows, cols = np.indices(array.shape)  

    geoms = []

    vals = []

    for col,row in zip(np.ravel(cols), np.ravel(rows)):

        # NOTE - top_left is x,y

        top_left = transform * (col,row)

        latitude = top_left[1]

        longitude = top_left[0]



        geoms += [Polygon([top_left, 

                           (top_left[0]+resolution, top_left[1]),

                           (top_left[0]+resolution, top_left[1]-resolution),

                           (top_left[0], top_left[1]-resolution)])]

        # NOTE - out_array is index by (row,col) wherease it is transformed by (col,row)

        vals += [array[(row, col)]]



    gpdf = gpd.GeoDataFrame(vals, geoms, columns=['tcc'])

    gpdf.index.name = 'geometry'

    gpdf = gpdf.reset_index()

    return gpdf
gpdf = to_gpdf(array, src.transform)

gpdf['longitude'] = gpdf.geometry.centroid.x

gpdf['latitude'] = gpdf.geometry.centroid.y

print(gpdf.shape)

gpdf.head()
# Load city shapefiles and calculate population density

# Seems to be city proper, rather than the urban areas available from the census website

df_city = gpd.read_file("/kaggle/input/usfs-tree-canopy-cover/CityBoundaries.shp")

df_city['area_square_meters'] = df_city.geometry.area

df_city['area_square_km'] = df_city['area_square_meters'] / 1000**2

df_city['area_square_miles'] = df_city['area_square_km'] * 0.621371**2

df_city['population_density'] = df_city['POP2010'] / df_city['area_square_miles']



df_city = df_city.to_crs({'init': 'epsg:4326'})  # Note: newer pyproj versions prefer 'EPSG:4326' rather than {'init': 'epsg:4326'}

df_city = df_city.sort_values('POP2010', ascending=False).iloc[:50]  # start with 50 most populous

# WARNING - PLACEFIPS is not unique...

# df_city['id'] = df_city['PLACEFIPS'].astype('str')

df_city = df_city.reset_index()

df_city['id'] = df_city['index'].astype('str')

df_city.head()
def aggregate_tcc(polygon, tolerance=0.1):

    # 10X speedup for tolerance 0.1

    polygon = polygon.simplify(tolerance)

    

    # crude selection of tcc

    x_min, y_min, x_max, y_max = polygon.bounds

    df_tcc = gpdf.loc[gpdf.latitude.between(y_min, y_max) & 

                      gpdf.longitude.between(x_min, x_max)]

    

    # Aggregate TCC within polygon

    try: 

        tcc_polygon = df_tcc.loc[df_tcc.centroid.within(polygon)]

        return tcc_polygon.tcc.mean()

    # for the 1km resolution (but NOT the 10km resolution), getting TopologyException

    # internet suggests self-intersecting polygons? 

    except: 

        return np.NaN
df_city['tcc'] = df_city.geometry.apply(aggregate_tcc)

df_city = df_city.loc[~df_city.tcc.isna()]  # Exclude Hawaii & Alaska (TCC data is continental US), and cities without TCC tiles at given resolution 
data_json = df_city.to_json()



folium.GeoJson(

    data_json,

    style_function=lambda x: {'fillOpacity': 0},    

    name='geojson'

).add_to(m)



m
def scatter_tcc_by_city(df_city: gpd.GeoDataFrame):

    data = df_city



    data['size'] = data['POP2010'] / data['POP2010'].max() * 100



    cities_to_label=['New York', 'Miami', 'Boston', 'Raleigh', 'Nashville', 'San Francisco', 'Atlanta', 

                    'Phoenix', 'Los Angeles', 'Chicago', 'Washington', 'Seattle']

    d = data.loc[~data.NAME.isin(cities_to_label)]

    fig = go.Figure(data=go.Scatter(

        x=d['population_density'],

        y=d['tcc'],

        mode='markers',

        marker=dict(size=d['size'],

                    color=d['tcc'],

                   colorscale='YlGn'),

        text=d['NAME'],

    ))



    d = data.loc[data.NAME.isin(cities_to_label)]

    fig.add_trace(go.Scatter(

        x=d['population_density'],

        y=d['tcc'],

        mode='markers+text',

        marker=dict(size=d['size'],

                    color=d['tcc'],

                   colorscale='YlGn'),

        text=d['NAME'],

        textposition="top center"

    ))





    fig.update_layout(title='Tree Cover Canopy for Major US Cities',

                      xaxis_title='Population Density (people per square mile)  |  Bubble size is population', yaxis_title='Tree Cover Canopy (%)',

                     showlegend=False)

    return fig

fig = scatter_tcc_by_city(df_city)

# fig.write_html("tcc_by_city_scatter.html")

fig.show()