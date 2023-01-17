import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import folium

import os

import bq_helper

from bq_helper import BigQueryHelper



nyc = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                               dataset_name="new_york")



query = """SELECT

 station_id,

 name,

 longitude,

  latitude,

  capacity,

  is_installed,

  is_renting,

  is_returning

FROM

  `bigquery-public-data.new_york.citibike_stations`

LIMIT 2000000;

"""



data = nyc.query_to_pandas_safe(query, max_gb_scanned=10)

data = data.rename(columns={"longitude":"lng", "latitude":"lat" }) 



#显示所有列

pd.set_option('display.max_columns', None)

#显示所有行

pd.set_option('display.max_rows', None)

#设置value的显示长度为50

pd.set_option('max_colwidth',50)

data.head(10)
data.describe()
data.isnull().sum()  #查看 null 值
data[(data['lat'].values== 0)|(data['lng'].values== 0)] #找出经度或纬度为0的站点
data.drop(data.index[9], inplace=True) #显然该数据是错误的统计数据，我们直接将其删除
from IPython.display import set_matplotlib_formats

from matplotlib import pyplot as plt



set_matplotlib_formats('retina')  # 打印高清图

plt.rcParams['figure.figsize'] = (15,10) 

fig, axes = plt.subplots(ncols=3, nrows=1)

plt.subplots_adjust(wspace =2)



sns.countplot(x='is_installed',data=data, ax=axes[0]) #对不同值的 'is_renting' 进行计数并绘图

sns.countplot(x='is_renting',data=data, ax=axes[1]) 

sns.countplot(x='is_returning',data=data, ax=axes[2]) 
g = sns.catplot(x="is_installed", y="is_returning", hue="is_renting", data=data, height=5, aspect=2.6)
data[(data['is_installed'].values== False)&(data['is_renting'].values== True)] 
g = sns.catplot(x="is_renting", y="is_returning", data=data, height=5, aspect=2.6)
#histogram for the capacity

plt.figure(1,figsize=(15,5))

data['capacity'].plot(kind='hist',bins=40)

plt.show()
g=sns.kdeplot(data[data['is_renting']==True]['capacity'],shade='True',label='is_renting',color='r') # 往外租的站点 'capacity' 分布

g=sns.kdeplot(data[data['is_renting']==False]['capacity'],shade='True',label='not_renting',color='b') # 不往外租的站点 'capacity' 分布
data.describe()
installed = np.unique(data['is_installed'])

colors = [plt.cm.tab10(i/float(len(installed)-1)) for i in range(len(installed))]



plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')



for i, is_installed in enumerate(installed):

    plt.scatter('lng', 'lat', 

                data=data.loc[data.is_installed==is_installed, :], 

                s=20, cmap=colors[i], label=str(is_installed))



plt.gca().set(xlim=(-74.116937, -73.887744), ylim=(40.635400, 40.834394),

              xlabel='lng', ylabel='lat')



plt.xticks(fontsize=12); plt.yticks(fontsize=12)

plt.title("locations vs stations ", fontsize=22)

plt.legend(fontsize=12)    

plt.show()    
df1 = data.loc[:, ['lng', 'lat']]
nyc_map = folium.Map(location=[40.655400,-73.976431],

                        zoom_start=13,

                        tiles="CartoDB dark_matter")

for i in range(len(df1)):

    lng = df1.iloc[i][0]

    lat = df1.iloc[i][1]

    folium.CircleMarker(location = [lat, lng], fill = True).add_to(nyc_map)

    

    

nyc_map
df2 = data.loc[:, ['lng', 'lat', 'capacity']]
nyc_map = folium.Map(location=[40.655400,-73.976431],

                        zoom_start=13,

                        tiles="CartoDB dark_matter")

for i in range(len(df2)):

    lng = df2.iloc[i][0]

    lat = df2.iloc[i][1]

    capacity = df2.iloc[i][2]

    radius = (df2.iloc[i][2]+55)/10

    

    if radius > 10:

        color = "#FF4500"

    else:

        color = "#008080"

    

    info = """capacity : {}<br>"""

    info = info.format(capacity

                               )

    folium.CircleMarker(location = [lat, lng], popup= info,radius = radius, color = color, fill = True).add_to(nyc_map)

    

nyc_map
nyc_map = folium.Map(location=[40.655400,-73.976431],

                        zoom_start=13,

                        tiles="cartodbpositron")

for i in range(len(df2)):

    lng = df2.iloc[i][0]

    lat = df2.iloc[i][1]

    capacity = df2.iloc[i][2]

    radius = (df2.iloc[i][2]+55)/10

    

    if radius > 10:

        color = "#FF4500"

    else:

        color = "#008080"

    

    info = """capacity : {}<br>"""

    info = info.format(capacity

                               )

    folium.CircleMarker(location = [lat, lng], popup= info,radius = radius, color = color, fill = True).add_to(nyc_map)

    

nyc_map