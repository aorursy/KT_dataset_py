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
import os
from IPython.display import HTML, display,Image
import json

import geopandas as gpd
from geopandas.tools import sjoin

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import shapely
from shapely.geometry import Point

import unicodedata
import pysal as ps


%matplotlib inline
data1 = pd.read_csv('../input/penalty_data_set_sorted_rev2.csv')




data1.head(20)
df = pd.DataFrame(data1, columns= ['LOCATION_CODE','LOCATION_DETAILS','SPEED_IND','TOTAL_NUMBER'])
df1 = df[df.SPEED_IND.notnull()]

df1.head(20)

#pd.merge(df1, on = ['LOCATION_CODE'])
#df1.set_index('LOCATION_CODE').stack()
df1_ar = df1.groupby(['LOCATION_CODE'], as_index=False).sum()
#df1_ar.set_index('LOCATION_CODE')
df1_rank = df1_ar.sort_values('TOTAL_NUMBER', ascending=False)
df1ten = df1_rank.head(10)


def highlight_column(x):
    y = 'background-color: cyan'
    df1ten_color = pd.DataFrame('', index=x.index, columns=x.columns)
    df1ten_color.iloc[:5, :2] = y
    return df1ten_color
df1ten.style.apply(highlight_column, axis=None)
df = pd.DataFrame(data1, columns= ['LOCATION_CODE','LOCATION_DETAILS','RED_LIGHT_CAMERA_IND','TOTAL_NUMBER'])
df2 = df[df.RED_LIGHT_CAMERA_IND.notnull()]
df2.head(20)
df2_ar = df2.groupby(['LOCATION_CODE'], as_index=False).sum()
df2_rank = df2_ar.sort_values('TOTAL_NUMBER', ascending=False)
df2ten = df2_rank.head(10)

def highlight_column(x):
    y = 'background-color: cyan'
    df2ten_color = pd.DataFrame('', index=x.index, columns=x.columns)
    df2ten_color.iloc[:5, :2] = y
    return df2ten_color
df2ten.style.apply(highlight_column, axis=None)
import folium
from folium.plugins import MarkerCluster
from folium.map import *
map_osm = folium.Map(location=[-33.877632, 151.082277])
map_osm.save('/tmp/map.html')




m = folium.Map(location=[-33.877632, 151.082277], tiles='Stamen Toner', zoom_start=11)
 
# SPEED Camera marker
S1="""
    <h1> </h1><br>
    GREAT WESTERN HIGHWAY, MT VICTORIA
    <p>
    <code>
        SPEED CAMERA<br>
        RANK: 1<br>
        PENALTIES: 19168
    </code>
    </p>
    """
iframeS1 = folium.IFrame(html=S1, width=350, height=150)
popupS1 = folium.Popup(iframeS1, max_width=1000)

S2="""
    <h1> </h1><br>
    STACEY STREET, BANKSTOWN
    <p>
    <code>
        SPEED CAMERA<br>
        RANK: 2<br>
        PENALTIES: 18077
    </code>
    </p>
    """
iframeS2 = folium.IFrame(html=S2, width=350, height=150)
popupS2 = folium.Popup(iframeS2, max_width=1000)

S3="""
    <h1> </h1><br>
    WOODVILLE ROAD, GRANVILLE
    <p>
    <code>
        SPEED CAMERA<br>
        RANK: 3<br>
        PENALTIES: 9703
    </code>
    </p>
    """
iframeS3 = folium.IFrame(html=S3, width=350, height=150)
popupS3 = folium.Popup(iframeS3, max_width=1000)

S4="""
    <h1> </h1><br>
    FALCON STREET, NEUTRAL BAY
    <p>
    <code>
        SPEED CAMERA<br>
        RANK: 4<br>
        PENALTIES: 8621
    </code>
    </p>
    """
iframeS4 = folium.IFrame(html=S4, width=350, height=150)
popupS4 = folium.Popup(iframeS4, max_width=1000)

S5="""
    <h1> </h1><br>
    ELIZABETH STREET, SYDNEY
    <p>
    <code>
        SPEED CAMERA<br>
        RANK: 5<br>
        PENALTIES: 8593
    </code>
    </p>
    """
iframeS5 = folium.IFrame(html=S5, width=350, height=150)
popupS5 = folium.Popup(iframeS5, max_width=1000)

folium.Marker([-33.579721, 150.222532], 
              popup=popupS1,
              icon=folium.Icon(color='blue')
             ).add_to(m)
folium.Marker([-33.879264, 151.215236], 
              popup=popupS2,
              icon=folium.Icon(color='blue')
             ).add_to(m)
folium.Marker([-33.897478, 151.099810], 
              popup=popupS3,
              icon=folium.Icon(color='blue')
             ).add_to(m)
folium.Marker([-33.895468, 151.221482], 
              popup=popupS4,
              icon=folium.Icon(color='blue')
             ).add_to(m)
folium.Marker([-33.874218, 151.216292], 
              popup=popupS5,
              icon=folium.Icon(color='blue')
             ).add_to(m)
    
    
    
# Red Light Camera marker
R1="""
    <h1> </h1><br>
    GEORGE STREET, HAYMARKET
    <p>
    <code>
        RED LIGHT CAMERA<br>
        RANK: 1<br>
        PENALTIES: 6855
    </code>
    </p>
    """
iframe1 = folium.IFrame(html=R1, width=350, height=150)
popup1 = folium.Popup(iframe1, max_width=1000)

R2="""
    <h1> </h1><br>
    STACEY STREET, BANKSTOWN
    <p>
    <code>
        RED LIGHT CAMERA<br>
        RANK: 2<br>
        PENALTIES: 6215
    </code>
    </p>
    """
iframe2 = folium.IFrame(html=R2, width=350, height=150)
popup2 = folium.Popup(iframe2, max_width=1000)

R3="""
    <h1> </h1><br>
    WOODVILLE ROAD, GRANVILLE
    <p>
    <code>
        RED LIGHT CAMERA<br>
        RANK: 3<br>
        PENALTIES: 4414
    </code>
    </p>
    """
iframe3 = folium.IFrame(html=R3, width=350, height=150)
popup3 = folium.Popup(iframe3, max_width=1000)

R4="""
    <h1> </h1><br>
    FALCON STREET, NEUTRAL BAY
    <p>
    <code>
        RED LIGHT CAMERA<br>
        RANK: 4<br>
        PENALTIES: 3965
    </code>
    </p>
    """
iframe4 = folium.IFrame(html=R4, width=350, height=150)
popup4 = folium.Popup(iframe4, max_width=1000)

R5="""
    <h1> </h1><br>
    ELIZABETH STREET, SYDNEY
    <p>
    <code>
        RED LIGHT CAMERA<br>
        RANK: 5<br>
        PENALTIES: 3930
    </code>
    </p>
    """
iframe5 = folium.IFrame(html=R5, width=350, height=150)
popup5 = folium.Popup(iframe5, max_width=1000)

folium.Marker([-33.882848, 151.204195], 
              popup=popup1,
              icon=folium.Icon(color='red')
             ).add_to(m)
folium.Marker([-33.916676, 151.041053], 
              popup=popup2,
              icon=folium.Icon(color='red')
             ).add_to(m)
folium.Marker([-33.827973, 151.004927], 
              popup=popup3,
              icon=folium.Icon(color='red')
             ).add_to(m)
folium.Marker([-33.829647, 151.213025], 
              popup=popup4,
              icon=folium.Icon(color='red')
             ).add_to(m)
folium.Marker([-33.873315, 151.209916], 
              popup=popup5,
              icon=folium.Icon(color='red')
             ).add_to(m)



m

data2 = pd.read_csv('../input/TZP2016 Employment by industry and travel zone 2011-2056.csv')
data2.head(20)
dfa = pd.DataFrame(data2, columns= ['TZ_NAME11','Industry','EMP_2016'])
dfa1 = dfa[dfa.TZ_NAME11.isin(['Mount Victoria Station'])]
dfa1



dfa1_ar = dfa1.sort_values('EMP_2016', ascending = False)
dfa1_rank = dfa1_ar.iloc[dfa1_ar['EMP_2016'].astype(float).argsort()]
dfa1_rank.head(35)


def highlight_column(x):
    y = 'background-color: cyan'
    dfa1_color = pd.DataFrame('', index=x.index, columns=x.columns)
    dfa1_color.iloc[range (29, 33), 2] = y
    return dfa1_color
dfa1_rank.style.apply(highlight_column, axis=None)
totala1 = dfa1_ar['EMP_2016'].astype(float).sum()
totala1
#Py for Mount Victoria
fig, ax = plt.subplots(figsize=(15, 7.5), subplot_kw=dict(aspect="equal"))

recipe1 = ["74.56804 Transport-Postal-Warehousing",
          "40.24344 Accommodation-Food_Services",
          "36.65257 Education-Training",
          "32.54229 Construction",
          "114.72754999999997 Rest"]

data3 = [float(x.split()[0]) for x in recipe1]
ingredients1 = [x.split()[-1] for x in recipe1]


def func(pct, allvals):
    absolute1 = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute1)


wedges, texts, autotexts = ax.pie(data3, autopct=lambda pct: func(pct, data3),
                                  textprops=dict(color="w"))

ax.legend(wedges, ingredients1,
          title="Occupation List",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")

ax.set_title("OCCUPATION AT MT VICTORIA")

plt.show()
dfb = pd.DataFrame(data2, columns= ['TZ_NAME11', 'SA2_NAME11','Industry','EMP_2016'])
dfb1 = dfb[dfb.SA2_NAME11.isin(['Sydney - Haymarket - The Rocks'])]
# dfb2 = dfb1[dfb1.TZ_NAME11.isin(['Hay'])]
# dfb2 = dfb1[(dfb1.SA2_NAME11 == 'Sydney - Haymarket - The Rocks') & (dfb1.TZ_NAME11 == 6)]

dfb1.head(20)
dfb2 = dfb1[dfb1['TZ_NAME11'].str.contains("Chinatown")]
dfb2_ar = dfb2.sort_values('EMP_2016', ascending = False)
# dfb2_group = dfb2_ar.groupby(['Industry'], as_index=False).sum()  #THIS DOESN\"T WORK, IT WILL CONCACTENATE THE STRINGS. NEED TO FIND A WAY TO ADD EMP_2016 INTO NEW COLUMN WITH FLOAT TRANSLATED. This is only to give blue highlights as well.
dfb2_ar.head(25)
dfb2_ar[['EMP_2016']] = dfb2_ar[['EMP_2016']].apply(pd.to_numeric)
dfb2_ar
dfb2_rank = dfb2_ar.sort_values('EMP_2016', ascending=False)
dfb2_rank
dfb2_group = dfb2_rank.groupby(['Industry'], as_index=False).sum()
dfb2_group
dfb2_sorted = dfb2_group.sort_values('EMP_2016', ascending=False)
dfb2_sorted

def highlight_column(x):
    y = 'background-color: cyan'
    dfb1_color = pd.DataFrame('', index=x.index, columns=x.columns)
    dfb1_color.iloc[:4, :2] = y
    return dfb1_color
dfb2_sorted.style.apply(highlight_column, axis=None)
totalb1 = dfb2_sorted['EMP_2016'].astype(float).sum()
totalb1
#Py for Haymarket
fig, ax = plt.subplots(figsize=(15, 7.5), subplot_kw=dict(aspect="equal"))

recipe2 = ["687.159 Accommodation-Food_Services",
          "175.325 Retail-Trade",
          "158.261 Professional-Scientific-Technical_Services",
          "98.3247 Health_Care-Social_Assistance",
          "308.34503 Rest"]

data4 = [float(x.split()[0]) for x in recipe2]
ingredients2 = [x.split()[-1] for x in recipe2]


def func(pct, allvals):
    absolute2 = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute2)


wedges, texts, autotexts = ax.pie(data4, autopct=lambda pct: func(pct, data4),
                                  textprops=dict(color="w"))

ax.legend(wedges, ingredients2,
          title="Occupation List",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")

ax.set_title("OCCUPATION AT HAYMARKET")

plt.show()


