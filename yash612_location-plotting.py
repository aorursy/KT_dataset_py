%matplotlib inline

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebraimport math

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Distribution graphs (histogram/bar graph) of column data

city_data=pd.read_csv('/kaggle/input/world-cities-database/worldcitiespop.csv')

city_data=city_data.drop_duplicates(subset=['City','AccentCity'], keep=False)

df=pd.read_csv('../input/black-lives-matter-twitter-dataset/data.csv')

df['City']=df['Location']

df.drop(['Location','Sentiment'],axis=1,inplace=True)

cleaned=df.loc[df['City']!='not given']
def city_naming(s):

    splitter=s.split(',')

    return splitter[0]



def lowing(str):

    return str.lower()

cleaned=cleaned[cleaned['City'].notnull()]

cleaned=cleaned[cleaned.City.apply(lambda x: str(x).isalpha())]

cleaned['City']=cleaned['City'].apply(city_naming)

cleaned['City']=cleaned['City'].apply(lowing)

city_data['City']=city_data['City'].apply(lowing)

merged=pd.merge(cleaned,city_data,on='City')
merged['Population'].fillna((merged['Population'].mean()),inplace=True)
merged


import matplotlib.pyplot as plt

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
lat = merged['Latitude'].values

lon = merged['Longitude'].values

population =  merged['Population'].values

area = merged['Population'].values/merged['Population'].values.min()

cit=merged['City']
fig = plt.figure(figsize=(15, 15))

m = Basemap(projection='cyl', resolution=None,

            llcrnrlat=-90, urcrnrlat=90,

            llcrnrlon=-180, urcrnrlon=180, )

m.shadedrelief()

# 2. scatter city data, with color reflecting population

# and size reflecting area

m.scatter(lon, lat, latlon=True,

          c=np.log10(population), s=area,

          cmap='Reds', alpha=0.5)



# 3. create colorbar and legend

plt.colorbar(label=r'$\log_{10}({\rm population})$')

plt.clim(3, 7)



# make legend with dummy points

for a in [100, 300, 500]:

    plt.scatter([], [], c='k', alpha=0.5, s=a,

                label=str(a) + ' km$^2$')

plt.legend(scatterpoints=1, frameon=False,

           labelspacing=1, loc='lower left');
