import pandas as pd

import numpy as np

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt



data = pd.read_csv('../input/globalterrorismdb_0616dist.csv', encoding='ISO-8859-1', low_memory=False)
data.shape
data.head()
data.get_dtype_counts()
#Create an array of the regions

regions = set(data['region'].unique())



print('Number of unique regions:', len(regions))
# Loop through each region.

for region in regions:

    # Section out the region data.

    region_data = data[data['region'] == region]

    # Find each regions min and max coordinates.

    min_lat = region_data['latitude'].min() - 5

    max_lat = region_data['latitude'].max() + 5

    min_lon = region_data['longitude'].min() - 5

    max_lon = region_data['longitude'].max() + 5

    # Create a map from each min and max coordinate.

    m = Basemap(projection='merc',llcrnrlat=min_lat,urcrnrlat=max_lat,\

                llcrnrlon=min_lon,urcrnrlon=max_lon,lat_ts=20,resolution='c')

    m.drawcoastlines()

    m.drawcountries()

    x, y = m(region_data['longitude'].values, region_data['latitude'].values)

    m.fillcontinents(zorder=0)

    m.scatter(x, y, color='r')

    plt.title('Region: ' + str(region))

    plt.show()

    # Create borders and the sea.

    # display the terrorist attacks.
data['region'].value_counts()
year_attacks = data['iyear'].value_counts().sort_index()

plt.plot(year_attacks.index, year_attacks)

plt.xlabel('Year')

plt.ylabel('Number of attacks')

plt.show()



# Loop through each region.

for region in regions:

    # Find the number off attacks by year for each region

    region_year_attacks = data[data['region'] == region]['iyear'].value_counts()

    # Sort it by the year so that it draws properly

    region_year_attacks = region_year_attacks.sort_index()

    plt.plot(region_year_attacks.index, region_year_attacks)

    

# Set up the legend, labels, and finally show.

plt.legend(regions)

plt.xlabel('Year')

plt.ylabel('Number of Attacks')

plt.show()
data['extended'].value_counts(normalize=True)
import seaborn as sns



sns.distplot(data['longitude'].dropna())

plt.ylabel('Distribution of Attacks.')

plt.show()

sns.distplot(data['latitude'].dropna())

plt.ylabel('Distribution of Attacks.')

plt.show()
data['vicinity'].value_counts(normalize=True)
print(data['crit1'].value_counts())

print(data['crit2'].value_counts())

print(data['crit3'].value_counts())
data['doubtterr'].value_counts(normalize=True)
data['success'].value_counts(normalize=True)
data['suicide'].value_counts(normalize=True)
data['attacktype1'].value_counts(normalize=True).sort_index()
data['guncertain1'].value_counts(normalize=True)
(data['nperps'] > 0).value_counts(normalize=True)
sns.distplot(data[(data['nperps'] > 0) & (data['nperps'] < 100)]['nperps'].dropna())

plt.xlabel('Number of Perpetrators')

plt.ylabel('Distribution')

plt.show()
sns.distplot(data[(data['nperpcap'] > 0) & (data['nperpcap'] < 100)]['nperpcap'].dropna())

plt.show()
data['claimed'].value_counts(normalize=True)
data['claimmode'].value_counts(normalize=True)
data[data['compclaim'] >= 0]['compclaim'].value_counts(normalize=True)
sns.distplot(data['nkill'].dropna())

plt.show()
sns.distplot(data['nkill'].dropna(), kde=False)

plt.show()

print(data['nkill'].describe())
sns.distplot(data['nwound'].dropna())

plt.show()

print(data['nwound'].describe())
print(data['property'].value_counts())
print(data['ishostkid'].value_counts())
# Take out all unkowns, groupby and give the percentagees.

data[data['INT_LOG'] >=0].groupby('region')['INT_LOG'].value_counts(normalize=True)
data[data['INT_IDEO'] >=0].groupby('region')['INT_IDEO'].value_counts(normalize=True)
data[data['INT_ANY'] >=0].groupby('region')['INT_ANY'].value_counts(normalize=True)