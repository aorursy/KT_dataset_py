# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import requests

import folium
# 확대 지정 (zoom_start)

map_osm = folium.Map(location=(37.56629, 126.979808), zoom_start=17)

# map_osm

시청_좌표=(37.56629, 126.979808)
df = pd.read_csv('../input/sanbul2.csv',encoding='cp949')

df.columns = df.columns.str.replace(' ','')

df.head()
df.columns
df = df[['발생일시_년', '발생일시_월','발생장소_시도','발생장소_시군구','X-좌표','Y-좌표','발생원인_세부원인','피해면적_합계']]

df.head(10)
df['X-좌표'] = df['X-좌표'].astype(float)

df['Y-좌표'] = df['Y-좌표'].astype(float)
map_osm = folium.Map(location=(37.56629, 126.979808), zoom_start=17)

# map_osm

시청_좌표=(37.56629, 126.979808)
map_osm = folium.Map(location=시청_좌표, zoom_start=11)



for ix, row in df.iterrows():

    location = (row['Y-좌표'], row['X-좌표'])

    folium.Marker(location, popup=row['피해면적_합계']).add_to(map_osm)



map_osm
df3 = df.loc[ (df['발생장소_시도'] =='경기도') | (df['발생장소_시도'] =='강원도') | (df['발생장소_시도'] =='경상북도')]

df.head(5)
map_osm = folium.Map(location=시청_좌표, zoom_start=11)

for ix, row in df3.iterrows():

    location = (row['Y-좌표'], row['X-좌표'])

    folium.Marker(location, popup=row['피해면적_합계']).add_to(map_osm)



map_osm