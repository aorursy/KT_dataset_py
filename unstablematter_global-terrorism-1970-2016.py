# Contains graphs:

# Plots per region based on long and lattitude the amount of terrorist attacks

# Plots for each month the amount of terrorist attacks

# Plots for the day of the month the amount of terrorist attacks

# Plots for the long and latitude the amount of terrorist attacks with a hue on month

# Plots per region the percentage of chosen targets

# Plots the cohesion between cells.
# data analysis

import pandas as pd

import numpy as np



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from mpl_toolkits.basemap import Basemap

from IPython.display import HTML

import plotly.plotly as py

from plotly.graph_objs import *



#4 dubbelop

#5-6 onrelevant

#15-24 onrelevant, niet van toepassing op het onderzoek

    #19 heeft veel oninterresante tekst

#30-33 onrelevant voor onze vraag en vaak leeg.

#36-57 idem

#59-80 idem

#99 united states en scope is globaal

#102 united states en scope is globaal

#104-109 onrelevant voor onze vraag en vaak leeg.

#111-121 onrelevant voor onze vraag en vaak leeg.

#125-129 onrelevant voor onze vraag en vaak leeg.



t_file = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1',

                    usecols=[0, 1, 2, 3, 7, 9, 11, 12, 13, 14, 15, 19, 20, 21, 26, 28, 29, 34, 35, 36, 58, 68, 69, 81, 98, 101, 104, 134]) 

t_file.columns
df_day_coords = t_file[['imonth', 'iday', 'longitude', 'latitude', 'success']].copy()

df_day_coords = df_day_coords[df_day_coords['iday'] != 0]

#df_day_coords['date_time'] = pd.to_datetime(df_day_coords['iyear'] + df_day_coords['imonth'] + df_day_coords['iday'].astype(str), format = "%Y%m%d")



df_day_coords.head()

#df_day_coords['day-month'] = t_file.iday.astype(str).str.cat(t_file.imonth.astype(str), sep='-')

#df_day_coords.groupby(['imonth', 'iday']).size().reset_index(name='counts')


#Rounds the long- and latitude to a number withouth decimals, groups them on long- and latitude and counts the amount of attacks.

df_coords = t_file.round({'longitude':0, 'latitude':0}).groupby(["longitude", "latitude"]).size().to_frame(name = 'count').reset_index()

sns.jointplot(x='longitude', y='latitude', data=df_coords, kind="hex", color="#4CB391", size=15, stat_func=None, edgecolor="#EAEAF2", linewidth=.2)

plt.title('Amount of terrorist attacks per rounded coordinates')
fig, axs = plt.subplots(nrows=12)

fig.set_size_inches(15, 100, forward=True)



for i in range(1,13):

    monthly_data = df_day_coords[df_day_coords['imonth'] == i]

    sns.countplot(x="iday", data=monthly_data, hue="success", ax=axs[i-1])

    axs[i-1].set_xlabel('Day of the month')

    axs[i-1].set_ylabel('Amount of terrorist attacks')
fig, ax = plt.subplots(figsize=(15,15))

sns.countplot(x="iday", data=df_day_coords, ax=ax, palette=sns.cubehelix_palette(15, start=.3, rot=.3))

ax.set_xlabel('Day of the month')

ax.set_ylabel('Amount of terrorist attacks')
#long and latitude is max 180 and min 180

sns.lmplot(x='longitude', y='latitude', data=df_day_coords, fit_reg=False, hue='imonth', size=15, palette=sns.cubehelix_palette(15, start=.3, rot=.3))
df_target_per_country = t_file.groupby(["country", "targtype1"]).size().to_frame(name = 'count').reset_index()

sns.lmplot(x='country', y='count', data=df_target_per_country,

           fit_reg=False, hue='targtype1', size=15, palette=sns.color_palette("BrBG", 7))

dictionary_target = t_file.set_index("targtype1")["targtype1_txt"].to_dict()

#plt.legend(bbox_to_anchor=(1, 1), loc=2)
dc_rt = t_file.set_index("targtype1")["region"].to_dict()



# Targets per country

df_rt = t_file.groupby(["region", "targtype1"]).size().to_frame(name = 'counted_target_types').reset_index()

df_rt_size = int(df_rt.size/3)



fig, axes = plt.subplots(nrows=df_rt_size, ncols=3)

fig.set_size_inches(15, 910)

for i in range(0, df_rt_size):

    for j in range(0, 3):

        axes[i,j].pie(df_rt[(df_rt.region == 1)]["counted_target_types"], autopct='%1.1f%%', startangle=90)

corr = t_file.corr()

f, ax = plt.subplots(figsize=(20, 15))

sns.heatmap(corr, ax=ax)
df = t_file[((t_file.country == 499) | (t_file.country == 362))]

df.head()