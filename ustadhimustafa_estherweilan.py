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
import geopandas as gpd
import libpysal as lp
import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np
import contextily as ctx
import shapely.geometry as geom
%matplotlib inline
#  !pip install libpysal
df = pd.read_csv('/kaggle/input/estherweilandata/DT_Data_AllStores.csv'
                ,usecols=["Town",
                            'Term_WeekdayAMPeak_SpeedKmh',
                            'Term_WeekdayLunch_SpeedKmh',
                            'Term_WeekdayPMPeak_SpeedKmh',
                             'Term_Sat12_SpeedKmh',
                            'Xmas_WeekdayAMPeak_SpeedKmh',
                            'Xmas_WeekdayLunch_SpeedKmh',
                            'Xmas_WeekdayPMPeak_SpeedKmh',
                            'Summer_WeekdayAMPeak_SpeedKmh',
                            'Summer_WeekdayLunch_SpeedKmh',
                            'Summer_WeekdayPMPeak_SpeedKmh',
                            'Summer_Sat12_SpeedKmh'])
listings = pd.read_csv('/kaggle/input/estherweilanlocation/uktowns.csv')
df.columns
data = df.merge(listings, how='left')
data.head()
data = data.rename(columns = {
    'Long':"Longitude",
    'Lat':"Latitude"
})
data.head()
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(data.Longitude, data.Latitude))
data.head()
gdf.head()
import seaborn as sns
sns.set()

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# We restrict to Europe
ax = world[world.name == 'United Kingdom'].plot(
    color='white', edgecolor='black')
# We can now plot our ``GeoDataFrame``.
gdf.plot(ax=ax, color='red')

plt.show()
# median_speed = gpd.sjoin(gdf[['Summer_WeekdayAMPeak_SpeedKmh', 'geometry']], gdf, op='within')\
#                   .groupby('index_right').Summer_WeekdayAMPeak_SpeedKmh.median()
# gdf['median_pri'] = median_speed.values

pd.isnull(gdf).sum()
gdf = gdf.dropna(how='any')
sns.set()

ax = world[world.name == 'United Kingdom'].plot(
    color='white', edgecolor='black')
gdf.plot(column='Term_WeekdayAMPeak_SpeedKmh',ax=ax)

sns.set()


ax = world[world.name == 'United Kingdom'].plot(
    color='white', edgecolor='black')
gdf.plot(column='Term_WeekdayAMPeak_SpeedKmh', scheme='Quantiles', k=5, cmap='GnBu', legend=True, ax=ax)

y = gdf['Term_WeekdayAMPeak_SpeedKmh']

y.median()
yb = y > y.median()
sum(yb)
# yb = y > y.median()
# labels = ["0 Low", "1 High"]
# yb = [labels[i] for i in 1*yb] 
# df['yb'] = yb

gdf.assign(yb=pd.cut(gdf['Term_WeekdayAMPeak_SpeedKmh'], bins = [0, y.median(), 1000],right=False, labels = ['Low','High']))
# fig, ax = plt.subplots(figsize=(12,10), subplot_kw={'aspect':'equal'})
# gdf['yb'].plot(cmap='binary', edgecolor='grey', legend=True, ax=ax)
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('Term_WeekdayAMPeak_SpeedKmh ~ C(Town)', data=gdf).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table
def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']

    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])

    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])

    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov

anova_table(aov_table)