



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os 

import xarray as xr 

import pandas as pd 
ds = xr.open_dataset("/kaggle/input/era5-pressure-and-cloud-cover/wind_t2m_tp_1979.nc")
# variance expliquée par choix aléatoire

l = [i for i in range(730)]

np.random.shuffle(l)



ds1 = ds.isel(time=l[:220]) 

ds2 = ds.isel(time=l[220:440])

ds3 = ds.isel(time=l[440:])

TSS = ((ds["t2m"]-ds["t2m"].mean("time"))**2).sum("time")

WSS = ((ds1["t2m"]-ds1["t2m"].mean("time"))**2).sum("time") + ((ds2["t2m"]-ds2["t2m"].mean("time"))**2).sum("time") +((ds3["t2m"]-ds3["t2m"].mean("time"))**2).sum("time")
(1 - WSS/TSS).plot(vmin=0,vmax=1)
ds1 = ds.sel(time=ds.time[0::2])

ds2 = ds.sel(time=ds.time[1::2])



TSS = ((ds["t2m"]-ds["t2m"].mean("time"))**2).sum("time")

WSS = ((ds1["t2m"]-ds1["t2m"].mean("time"))**2).sum("time") + ((ds2["t2m"]-ds2["t2m"].mean("time"))**2).sum("time") 
(1 - WSS/TSS).plot(vmin=0,vmax=1)
dayofyear = pd.to_datetime(ds.time.values).dayofyear 

winter = (dayofyear < 75) + (dayofyear>365-75)

summer = (dayofyear >= 75) * (dayofyear<=365-75)
ds_winter = ds.isel(time=winter)#.sel(time=ds.time[::2])

ds_summer = ds.isel(time=summer)#.sel(time=ds.time[::2])

d1 = ds_winter.sel(time=ds_winter.time[::2])

d2 = ds_summer.sel(time=ds_summer.time[::2])

d3 = ds_winter.sel(time=ds_winter.time[1::2])

d4 = ds_summer.sel(time=ds_summer.time[1::2])


TSS = ((ds["t2m"]-ds["t2m"].mean("time"))**2).sum("time")

WSS = ((d1["t2m"]-d1["t2m"].mean("time"))**2).sum("time") + ((d2["t2m"]-d2["t2m"].mean("time"))**2).sum("time") + ((d3["t2m"]-d3["t2m"].mean("time"))**2).sum("time") + ((d4["t2m"]-d4["t2m"].mean("time"))**2).sum("time") 
(1 - WSS/TSS).plot(vmin=0,vmax=1)
import datetime as dt 



d12h = ds.sel(time=dt.time(12))

d00h = ds.sel(time=dt.time(0))

# On calcul les anomalies (par rapport à la moyenne du mois) pour chacun des deux horaires

mensual_anomalie00H = d00h.groupby("time.month") - d00h.groupby("time.month").mean()

mensual_anomalie12H = d12h.groupby("time.month") - d12h.groupby("time.month").mean()

# On reregroupe tout 

ds_ano = xr.merge([mensual_anomalie00H,mensual_anomalie12H])

ds_winter = ds_ano.isel(time=winter)

ds_summer = ds_ano.isel(time=summer)

d1 = ds_winter.sel(time=dt.time(0))

d2 = ds_summer.sel(time=dt.time(0))

d3 = ds_winter.sel(time=dt.time(12))

d4 = ds_summer.sel(time=dt.time(12))


TSS = ((ds_ano["t2m"]-ds_ano["t2m"].mean("time"))**2).sum("time")

WSS = ((d1["t2m"]-d1["t2m"].mean("time"))**2).sum("time") + ((d2["t2m"]-d2["t2m"].mean("time"))**2).sum("time") + ((d3["t2m"]-d3["t2m"].mean("time"))**2).sum("time") + ((d4["t2m"]-d4["t2m"].mean("time"))**2).sum("time") 
(1 - WSS/TSS).plot(vmin=0,vmax=1)
def compute_anomalie(ds):

    """

    Retourne l'anomalie mensuelle (pour chaque heure) du dataset 

    """

    ano_list = []

    for hour in ds.time.groupby("time.hour").groups.keys(): 

        dtemp = ds.sel(time=dt.time(hour))

        ano_list.append(dtemp.groupby("time.month") - dtemp.groupby("time.month").mean())

    return xr.merge(ano_list)

    



def explained_variance(cluster_list):

    """

    Calcul a partir de la liste des cluster fournies (chaque cluster est un dataArray) la variance expliquée (pour chacune des variables)

    """

    VIntra = xr.Dataset()

    first = True

    mean_cluster = []

    nb_cluster = []

    for cluster in cluster_list:

        cluster_mean = cluster.mean("time") 

        nb_cluster.append(cluster.time.size)

        mean_cluster.append(cluster_mean)

        if first:

            VIntra = ((cluster - cluster_mean)**2).sum("time")

            first = False

        else: 

            VIntra = VIntra + ((cluster - cluster_mean)**2).sum("time")

    Tmean = VIntra *0 

    VInter = VIntra *0 

    Total_elt = np.asarray(nb_cluster).sum()

    for i, c_mean in enumerate(mean_cluster):

        Tmean = Tmean + c_mean * nb_cluster[i] / Total_elt

    print(Tmean)

    for i,c_mean in enumerate(mean_cluster):

        VInter = VInter + nb_cluster[i]*(c_mean - Tmean)**2

    VTT = VInter + VIntra

    print(VTT)

    return 1 -  VIntra/VTT

    
# Calcul d'anomalie 

ds_ano_b =  compute_anomalie(ds)

# Select by cluster date 

ds_winter = ds_ano_b.isel(time=winter)

ds_summer = ds_ano_b.isel(time=summer)

d1 = ds_winter.sel(time=dt.time(0))

d2 = ds_summer.sel(time=dt.time(0))

d3 = ds_winter.sel(time=dt.time(12))

d4 = ds_summer.sel(time=dt.time(12))

# Compute explained variance by variable 

dout = explained_variance([d1,d2,d3,d4])
import cartopy.crs as ccrs

import matplotlib.pyplot as plt 

vmax = 1

plt.figure(figsize=(20,20))

ax1=plt.subplot(2, 2, 1,projection=ccrs.Orthographic(0, 35))

ax1.coastlines()

ax2=plt.subplot(2, 2, 2,projection=ccrs.Orthographic(0, 35))

ax2.coastlines()

ax3=plt.subplot(2, 2, 3,projection=ccrs.Orthographic(0, 35))

ax3.coastlines()

ax4=plt.subplot(2, 2, 4,projection=ccrs.Orthographic(0, 35))

ax4.coastlines()

ax1.gridlines()

dout["t2m"].plot(ax=ax1,transform=ccrs.PlateCarree(),robust=True,vmin=0,vmax=vmax)

dout["tp"].plot(ax=ax2,transform=ccrs.PlateCarree(),robust=True,vmin=0,vmax=vmax)

dout["u10"].plot(ax=ax3,transform=ccrs.PlateCarree(),robust=True,vmin=0,vmax=vmax)

dout["v10"].plot(ax=ax4,transform=ccrs.PlateCarree(),robust=True,vmin=0,vmax=vmax)