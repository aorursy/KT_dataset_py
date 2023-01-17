import matplotlib.pyplot as plt # plotting

import xarray as xr

import cartopy.crs as ccrs 



import os 

import cartopy.feature as cfeature 

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Opening a given file 



ds  = xr.open_dataset("/kaggle/input/msl_ttc_1991.nc")
# Check what is inside the dataset. 

# Two variables are present : 

# - msl 

# - tcc



ds
# Selecting one time step and plotting for msl 

import matplotlib.pyplot as plt



fig = plt.figure(figsize=(20,7))

ax = plt.axes(projection=ccrs.PlateCarree())

ds["msl"].isel(time=0).plot.contourf(ax=ax,transform=ccrs.PlateCarree(),levels=20)

ax.coastlines(resolution='50m', color='white', linewidth=2)

ax.add_feature(cfeature.BORDERS.with_scale('50m'),edgecolor='white')

ax.gridlines(draw_labels=True)
import matplotlib.pyplot as plt





ds["msl"].isel(time=[0,1,2,3,4,5,6,7]).plot.contourf(levels=20,col='time', col_wrap=4)

ds["tcc"].isel(time=[0,1,2,3,4,5,6,7]).plot.contourf(levels=20,col='time', col_wrap=4)
# Seleting one time step and plotting for total cloud cover 

fig = plt.figure(figsize=(20,7))

ax = plt.axes(projection=ccrs.PlateCarree())

ds["tcc"].isel(time=0).plot.contourf(ax=ax,transform=ccrs.PlateCarree(),levels=20)

ax.coastlines(resolution='50m', color='white', linewidth=2)

ax.add_feature(cfeature.BORDERS.with_scale('50m'),edgecolor='white')

ax.gridlines(draw_labels=True)
# A bit of exploration using geoview 
import geoviews as gv 

import geoviews.feature as gf

from geoviews import opts

import geoviews.tile_sources as gts

gv.extension("bokeh")
coastline = gf.coastline(line_width=3,line_color='white').opts(projection=ccrs.GOOGLE_MERCATOR,scale='50m')

borders = gf.borders(line_width=3,line_color="black").options(scale='50m')

tile = gts.EsriImagery().opts(width=600, height=700)
ds.msl.min()
gv_msl = gv.Dataset(ds,vdims=["msl"],crs=ccrs.PlateCarree())

msl_image= gv_msl.to(gv.Image,['longitude', 'latitude'],"msl",dynamic=True).opts(opts.Image(colorbar=True,

                                                                                            clim=(95000,105000),cmap="jet",tools=["hover"],alpha=0.8))
msl_image * tile * borders * coastline
gv_tcc = gv.Dataset(ds,vdims=["tcc"],crs=ccrs.PlateCarree())

tcc_image= gv_tcc.to(gv.Image,['longitude', 'latitude'],"tcc",dynamic=True).opts(opts.Image(colorbar=True,

                                                                                            clim=(0,1.1),cmap="jet",

                                                                                            tools=["hover"],alpha=0.8))
tcc_image * tile * borders * coastline
# Opening a given file 



ds_o  = xr.open_dataset("/kaggle/input/wind_t2m_tp_1991.nc")
gv_msl = gv.Dataset(ds_o,vdims=["t2m"],crs=ccrs.PlateCarree())

msl_image= gv_msl.to(gv.Image,['longitude', 'latitude'],"t2m",dynamic=True).opts(opts.Image(colorbar=True,cmap="jet",tools=["hover"],alpha=0.8))

msl_image * tile * borders * coastline
gv_tp = gv.Dataset(ds_o,vdims=["tp"],crs=ccrs.PlateCarree())

tp_image= gv_tp.to(gv.Image,['longitude', 'latitude'],"tp",dynamic=True).opts(opts.Image(colorbar=True,cmap="jet",tools=["hover"],alpha=0.8))

tp_image * tile * borders * coastline
ds_o.tp