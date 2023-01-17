import matplotlib.pyplot as plt

import geopandas

from cartopy import crs as ccrs



path = geopandas.datasets.get_path('naturalearth_lowres')

df = geopandas.read_file(path)

# Add a column we'll use later

df['gdp_pp'] = df['gdp_md_est'] / df['pop_est']
df.plot()
# Define the CartoPy CRS object.

crs = ccrs.AzimuthalEquidistant()



# This can be converted into a `proj4` string/dict compatible with GeoPandas

crs_proj4 = crs.proj4_init

df_ae = df.to_crs(crs_proj4)



# Here's what the plot looks like in GeoPandas

df_ae.plot()
fig, ax = plt.subplots(subplot_kw={'projection': crs})

ax.add_geometries(df_ae['geometry'], crs=crs)
crs_new = ccrs.AlbersEqualArea()

new_geometries = [crs_new.project_geometry(ii, src_crs=crs)

                  for ii in df_ae['geometry'].values]



fig, ax = plt.subplots(subplot_kw={'projection': crs_new})

ax.add_geometries(new_geometries, crs=crs_new)
df_aea = geopandas.GeoDataFrame(df['gdp_pp'], geometry=new_geometries,

                                crs=crs_new.proj4_init)

df_aea.plot()
# Generate a CartoPy figure and add the countries to it

fig, ax = plt.subplots(subplot_kw={'projection': crs_new})

ax.add_geometries(new_geometries, crs=crs_new)



# Calculate centroids and plot

df_aea_centroids = df_aea.geometry.centroid

# Need to provide "zorder" to ensure the points are plotted above the polygons

df_aea_centroids.plot(ax=ax, markersize=5, color='r', zorder=10)



plt.show()