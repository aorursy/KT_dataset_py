# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# importing library 
import plotly.express as px

# loading data 



df = pd.read_csv("../input/tr-data/TRNufus.csv")
df.rename(columns={"Id":"id"},inplace = True)
df.head(10)



# loading Turkey geojson
from urllib.request import urlopen
import json
with urlopen("https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json") as response:
    counties = json.load(response)
   
    
maxi = df.Pop.max()
mini = df.Pop.min()
# lets plot map

fig = px.choropleth_mapbox(df, geojson=counties, locations="id", color='Pop',
                           color_continuous_scale='rainbow',                      
                           hover_name ="City",
                           animation_frame="Year",
                           range_color=(mini,maxi),
                           mapbox_style="carto-positron",
                           zoom=5, center={"lat": 38.9597594, "lon": 34.9249653},
                           opacity=0.5,
                           labels={'Pop':'population'},
                           title="Population change in Turkey over the years"
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
