# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns
conf_df = pd.read_csv('../input/corona-virus-report/time_series_2019-ncov-Confirmed.csv')

deaths_df = pd.read_csv('../input/corona-virus-report/time_series_2019-ncov-Deaths.csv')

recv_df = pd.read_csv('../input/corona-virus-report/time_series_2019-ncov-Recovered.csv')
conf_df['Type'] = 'Confirmed'

deaths_df['Type'] = 'Deaths'

recv_df['Type'] = 'Recoverd'

df = pd.concat([conf_df, deaths_df, recv_df])

df = pd.melt(df, id_vars = ['Type', 'Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Date')

df.value = pd.to_numeric(df.value, errors = 'coerce')

df.Date = pd.to_datetime(df.Date)

df = df.rename(columns = { 'Province/State' : 'State', 'Country/Region' : 'Country' })

df
#  Convert Mainland China to just China

df.loc[df.Country == 'Mainland China', 'Country'] = 'China'

df[df.Country == 'China']
df[(df.Country == 'China') & (df.State == 'Hubei') & (df.Type == 'Confirmed')].value
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



D = df[(df.Country == 'China') & (df.State == 'Hubei')].sort_values(by = ['Date'])



fig, ax = plt.subplots(figsize = (20, 10))

sns.set_style("ticks")

sns.lineplot(x = 'Date', y = 'value', hue = 'Type', data = D, ax = ax)



plt.axvline(x = '2020-02-12', color = 'gray')

plt.axvline(x = '2020-02-13', color = 'gray')

plt.annotate('Jump in cases due to changes in accounting method', ('2020-02-11', 50000),

             ha = 'right', size = 'x-large', color = 'red')

plt.title('Corona Virus Situation in Hubei, China', size = 'xx-large')
# Geo plot utility



from mpl_toolkits.basemap import Basemap

from itertools import chain



def draw_map(m, scale=0.2):

    # draw a shaded-relief image

    m.shadedrelief(scale=scale)

    

    # lats and longs are returned as a dictionary

    lats = m.drawparallels(np.linspace(-90, 90, 13))

    lons = m.drawmeridians(np.linspace(-180, 180, 13))



    # keys contain the plt.Line2D instances

    lat_lines = chain(*(tup[1][0] for tup in lats.items()))

    lon_lines = chain(*(tup[1][0] for tup in lons.items()))

    all_lines = chain(lat_lines, lon_lines)

    

    # cycle through these lines and set the desired style

    for line in all_lines:

        line.set(linestyle='-', alpha=0.3, color='w')
today = df[(df.Date == '2020-02-18') & (df.Type == 'Confirmed')]



fig = plt.figure(figsize=(20, 10), edgecolor='w')

m = Basemap(projection='cyl', resolution=None,

            llcrnrlat=-90, urcrnrlat=90,

            llcrnrlon=-180, urcrnrlon=180, )



m.scatter(today.Long, today.Lat, latlon=True,

          c=np.log10(today.value), #s=area,

          cmap='Reds', #alpha=0.8

         )



plt.colorbar(label=r'$\log_{10}({\rm Confirmed})$')

plt.clim(1, 6)

plt.title('Corona Virus Confirmed Cases as of Feb 18, 2020')



draw_map(m)