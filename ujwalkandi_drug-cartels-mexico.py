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
import numpy as np

import pandas as pd

import json

from tabulate import tabulate

import matplotlib.pyplot as plt

from matplotlib import rcParams

import matplotlib

import seaborn as sns

import geopandas as gpd

from geopandas.tools import geocode

import squarify    # pip install squarify (algorithm for treemap)

from mpl_toolkits.basemap import Basemap

import folium

import plotly.express as px

import plotly.graph_objects as go
plt.figure(figsize=(26,14))

CA_map = plt.imread('../input/mehico-cartels-img/Drug Cartels - Mehico.png')

plt.imshow(CA_map, zorder=1,aspect='auto')

plt.axis('off')

plt.title('Drug Cartels Activity across Mexico',fontsize=15)

plt.show()
df = pd.read_csv('../input/drug-cartels-mehico/Drug Cartels - Mehico.csv', engine='python')

df.head()
plt.figure(figsize=(25,9))

sns.lineplot(x='Year', y='Beltran_Leyva',data=df, ci=None, label="Beltran_Leyva", color='lime', lw=3)

sns.lineplot(x='Year', y='Beltran_Leyva_Family',data=df,  ci=None, label='Beltran_Leyva_Family', color='g', lw=3)

sns.lineplot(x='Year', y='Familia',data=df,  ci=None, label='Familia',color='blue', lw=3)

sns.lineplot(x='Year', y='Golfo',data=df, label='Golfo',  ci=None, color='darkorange', lw=3)

sns.lineplot(x='Year', y='Juarez',data=df, label='Juarez', ci=None, color='magenta', lw=3)

sns.lineplot(x='Year', y='Sinaloa',data=df, label='Sinaloa',  ci=None,color='aqua', lw=3)

sns.lineplot(x='Year', y='Sinaloa_Family',data=df,  ci=None, label='Sinaloa_Family',color='maroon', lw=3)

sns.lineplot(x='Year', y='Tijuana',data=df, label='Tijuana', color='darkgray',ci=None, lw=3)

sns.lineplot(x='Year', y='Zetas',data=df, label='Zetas', ci=None, color='red', lw=3)

sns.lineplot(x='Year', y='Otros',data=df, label='Otros',color='yellow', ci=None, lw=3)

plt.legend(title='Drug Cartels', edgecolor='k')

plt.xlabel('Year',fontsize=15)

plt.ylabel('Activity',fontsize=14)

plt.title('Drug Cartels Activity in Mexico',fontsize=18)

plt.show()
fig = px.histogram(df, x="Year", y=['Golfo','Sinaloa','Zetas','Tijuana','Sinaloa_Family','Familia','Juarez','Beltran_Leyva','Beltran_Leyva_Family','Otros'])

fig.show()