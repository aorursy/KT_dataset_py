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
import pandas as pd

# get all files from ftp server

df1 = pd.read_csv("ftp://data.asc-csa.gc.ca/users/OpenData_DonneesOuvertes/pub/MOPITT/Space%20App%20Covid%2019/2019/April-Avril%202019/MOP02J-20190401-L2V18.0.3.csv")

df2 = pd.read_csv("ftp://data.asc-csa.gc.ca/users/OpenData_DonneesOuvertes/pub/MOPITT/Space%20App%20Covid%2019/2019/April-Avril%202019/MOP02J-20190402-L2V18.0.3.csv")

df3 = pd.read_csv("ftp://data.asc-csa.gc.ca/users/OpenData_DonneesOuvertes/pub/MOPITT/Space%20App%20Covid%2019/2019/April-Avril%202019/MOP02J-20190403-L2V18.0.3.csv")

df4 = pd.read_csv("ftp://data.asc-csa.gc.ca/users/OpenData_DonneesOuvertes/pub/MOPITT/Space%20App%20Covid%2019/2019/April-Avril%202019/MOP02J-20190404-L2V18.0.3.csv")

df5 = pd.read_csv("ftp://data.asc-csa.gc.ca/users/OpenData_DonneesOuvertes/pub/MOPITT/Space%20App%20Covid%2019/2019/April-Avril%202019/MOP02J-20190405-L2V18.0.3.csv")

df = pd.concat([df1,df2,df3,df4,df5])

df.drop_duplicates(keep = False, inplace = True) 

#df=pd.read_csv('./out.csv')

df.columns

#list available columns



df.index # list no of available rows
df.head(5) # first 5 records
df.tail(5) # last 5 records
import matplotlib.pyplot as plt

plt.scatter(x=df[' Longitude'], y=df['# Latitude'] , color="g")

plt.show()
from shapely.geometry import Point

import geopandas as gpd

from geopandas import GeoDataFrame





geometry = [Point(xy) for xy in zip(df[' Longitude'], df['# Latitude'])]

gdf = GeoDataFrame(df['COMixingRatio surface'], geometry=geometry)   



world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15,label='CO');





# Surface Temperature vs COMixingRatio surface

plt.plot( df['RetrievedSurfaceTemperature '],df['COMixingRatio surface'])  

plt.show()  
# uncomment below line and click save version -> save and RunAll to get the out.csv in the output dir

'''df.iloc[0:121670].to_csv('./out1.csv')

df.iloc[121671:243340].to_csv('./out2.csv')

df.iloc[243340:365010].to_csv('./out3.csv')

df.iloc[365010:486680].to_csv('./out4.csv')

df.iloc[486680:608350].to_csv('./out5.csv')

df.iloc[608350:730020].to_csv('./out6.csv')

df.iloc[730020:851690].to_csv('./out7.csv')

df.iloc[851690:973360].to_csv('./out8.csv')

df.iloc[973360:1095030].to_csv('./out9.csv')

df.iloc[1095030:1216700].to_csv('./out10.csv')'''



df[['# Latitude', ' Longitude', ' COTotalColumn', 'COMixingRatio surface']].to_csv('./out.csv')


