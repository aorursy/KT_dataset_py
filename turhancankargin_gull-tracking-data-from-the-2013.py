import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Introduction to GPS Tracking of Birds

birddata = pd.read_csv('/kaggle/input/bird-migration/bird_migration.csv')

birddata.info()

birddata.head()
# Simple Data Visualizations

import matplotlib.pyplot as plt

import numpy as np

ix = birddata.bird_name == "Eric"

x,y = birddata.longitude[ix],birddata.latitude[ix]

plt.figure(figsize=(7,7))

plt.plot(x,y,".")

bird_names = pd.unique(birddata.bird_name)

plt.figure(figsize=(7,7))

for bird_name in bird_names:

    ix = birddata.bird_name == bird_name

    x,y = birddata.longitude[ix],birddata.latitude[ix]

    plt.plot(x,y,".")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.legend(loc="lower right")
# Examining Flight Speed

ix = birddata.bird_name == "Eric"

speed = birddata.speed_2d[ix]

plt.hist(speed[0:10])

np.isnan(speed).any()

np.sum(np.isnan(speed))

ind = np.isnan(speed)

plt.hist(speed[~ind])
# Using Datetime

date_str = birddata.date_time[0]

import datetime

datetime.datetime.today()

date_str[:-3]

datetime.datetime.strptime(date_str[:-3],"%Y-%m-%d %H:%M:%S")

timestamps = []

for k in range(len(birddata)):

    timestamps.append(datetime.datetime.strptime\

    (birddata.date_time.iloc[k][:-3],"%Y-%m-%d %H:%M:%S"))

birddata["timestamp"] = pd.Series(timestamps,index = birddata.index)

birddata.head()

times = birddata.timestamp[birddata.bird_name == "Eric"]

elapsed_time = [time-times[0] for time in times]

elapsed_time[1000]

plt.plot(np.array(elapsed_time)/datetime.timedelta(days=1))

elapsed_days = np.array(elapsed_time)/datetime.timedelta(days=1)

data = birddata[birddata.bird_name == "Eric"]
#Calculating Daily Mean Speed

next_day =1

inds = []

daily_mean_speed = []

for (i,t) in enumerate(elapsed_days):

    if t < next_day:

        inds.append(i)

    else:

        daily_mean_speed.append(np.mean(data.speed_2d[inds]))

        next_day += 1

        inds = []

plt.figure(figsize=(6,6))

plt.plot(daily_mean_speed)
# Using the Cartopy Library

import cartopy.crs as ccrs

import cartopy.feature as cfeature

proj = ccrs.Mercator()

plt.figure(figsize=(10,10))

ax = plt.axes(projection=proj)

ax.set_extent((-25.0,20.0,52.0,10.0))

ax.add_feature(cfeature.LAND)

ax.add_feature(cfeature.OCEAN)

ax.add_feature(cfeature.COASTLINE)

ax.add_feature(cfeature.BORDERS,linestyle=":")



for name in bird_names:

    ix = birddata["bird_name"] == name

    x,y = birddata.longitude[ix],birddata.latitude[ix]

    ax.plot(x,y,".",transform=ccrs.Geodetic(),label=name)

plt.legend(loc="upper left")