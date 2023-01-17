# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import mapclassify

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/pisa-2018-worldwide-ranking/pisa_ranking.csv',header=0,names=['index','Country','Avg_score'])

df.drop(columns='index',inplace= True )

# Cleaning the names 

df['Country']=df['Country'].str.lower()

df['Country']=df['Country'].str.strip()

#df.columns

df.head()
#Reading the map dataset

map_df = gpd.read_file('../input/countries-shape-files')

map_df=map_df.rename(columns={"SOVEREIGNT": "country", "geometry": "geometry"})

map_df = map_df[['country','geometry']]

map_df['country']=map_df['country'].str.lower()

map_df['country']=map_df['country'].str.strip()

map_df.head()
# changing few names to match both the dataset

map_df[map_df['country']=='macedonia'].index

map_df['country'][210]='serbia'

map_df['country'][148]='north macedonia'

map_df[map_df['country']=='czechia'].index

map_df['country'][60]='czech republic'

map_df[map_df['country']=='united republic of tanzania'].index

map_df['country'][233]='tanzania'

df[df['Country']=='united states'].index

df['Country'].iloc[37] = 'united states of america'
map_df['country'].drop_duplicates(inplace=True)

map_df.reset_index(drop=True,inplace=True)
Not_matching = []

def matching_country(name,country_list=map_df['country'].values):

    if  name not in country_list:

          Not_matching.append(name)

#checking countries that doesn't match

df['Country'].apply(lambda x: matching_country(x))

Not_matching

# since 'macau' and 'hong kong' are administrative regions, and they are present in the shape files, 

# they will removed in merging !
#merging both data

merged = map_df.set_index('country').join(df.set_index('Country'))

#filling a value of 300 for countries with no pisa score!

merged['Avg_score'].fillna(300,inplace=True)

merged.dropna().info()
fig, ax = plt.subplots(1,figsize=(15, 10))

ax.axis('off')

ax.set_title('Pisa Mathematics Ranking', fontdict={'fontsize': '35', 'fontweight' : '3'})

variable = 'Avg_score'





merged.plot(column='Avg_score', cmap='RdBu', scheme="User_Defined", 

         legend=True, edgecolor='0.8',classification_kwds=dict(bins=[300,350,400,450,500,550,600]),

         ax=ax)



leg = ax.get_legend()

ax.get_legend().set_title('Ranking')

leg.set_bbox_to_anchor((0.85, 0.6, 0.2, 0.2))



plt.show()
