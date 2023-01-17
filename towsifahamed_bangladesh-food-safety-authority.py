# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/bangladesh-food-safety-authority-restaurant-rating/bd-food-rating.csv')

df
df_area = df.area.value_counts().to_frame()

df_area['area_name']=df_area.index

df_area.reset_index(drop=True, inplace=True)

df_area.rename(columns={"area": "shop_count", "area_name": "area"}, inplace=True)

df_area
# download countries geojson file

!wget --quiet https://github.com/fahimxyz/bangladesh-geojson -O bd-postcodes.json

    

print('GeoJSON file downloaded!')
world_geo = r'bd-postcodes.json'
import folium

import matplotlib.pyplot as plt
'''world_map = folium.Map()



world_map.choropleth(



geo_data=world_geo,



data=df_area,



columns=['area', 'shop_count'],



#key_on='feature.properties.bd-',



fill_color='YlOrRd',



fill_opacity=0.7,



line_opacity=0.2,



legend_name='Shop Count'



)'''
df_temp = df['bfsa_approve_status']=='A+'

df_area_a = df[df_temp].area.value_counts().to_frame()

df_area_a['area_name']=df_area_a.index

df_area_a.reset_index(drop=True, inplace=True)

df_area_a.rename(columns={"area": "A+_shop_count", "area_name": "area"}, inplace=True)

df_area_a
df_area=df_area.merge(df_area_a,on=['area'],how = 'outer')

df_area
df_temp = df['bfsa_approve_status']=='A'

df_area_a = df[df_temp].area.value_counts().to_frame()

df_area_a['area_name']=df_area_a.index

df_area_a.reset_index(drop=True, inplace=True)

df_area_a.rename(columns={"area": "A_shop_count", "area_name": "area"}, inplace=True)

df_area=df_area.merge(df_area_a,on=['area'],how = 'outer')

df_area
df_temp = df['bfsa_approve_status']=='B'

df_area_a = df[df_temp].area.value_counts().to_frame()

df_area_a['area_name']=df_area_a.index

df_area_a.reset_index(drop=True, inplace=True)

df_area_a.rename(columns={"area": "B_shop_count", "area_name": "area"}, inplace=True)

df_area=df_area.merge(df_area_a,on=['area'],how = 'outer')

df_area
df_area.plot(kind='bar',x='area',y=['shop_count','A+_shop_count','A_shop_count','B_shop_count'],figsize=(20,10), color=['blue','green','yellow','red'])

plt.show()