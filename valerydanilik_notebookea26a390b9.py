# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from mpl_toolkits.basemap import Basemap #plotting

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



import warnings

warnings.simplefilter("ignore") # drop warnings

pd.options.display.max_columns = 200 # for showing all columns of tables





data = pd.read_csv("../input/globalterrorismdb_0616dist.csv", encoding='ISO-8859-1').set_index("eventid")
data['date'] = data.index.map(lambda index: '.'.join([str(index)[6:8],str(index)[4:6],str(index)[0:4]]))

data.tail()
russia_data = data.loc[data.country_txt == 'Russia',:]           
russia_data.latitude[russia_data.longitude.isnull()]
russia_data = russia_data.loc[:,('city','latitude','longitude', 'suicide', 'attacktype1_txt', 'success', 'nkill','date')]
russia_data = russia_data[russia_data.latitude.notnull()]
def draw_map(df, z=1):

    zoom = (10/3) + (1/3) * z

    m =  Basemap(projection='merc', llcrnrlon=df.longitude.min()-z,llcrnrlat=df.latitude.min()-z,urcrnrlon=df.longitude.max()+z,urcrnrlat=df.latitude.max()+z)

    m.bluemarble()

    m.scatter()

    plt.show()
draw_map(russia_data, 10)