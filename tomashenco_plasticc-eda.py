# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math
from astropy.coordinates import SkyCoord

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/training_set.csv')
train_metadata = pd.read_csv('../input/training_set_metadata.csv')
train_metadata.head()
train_metadata.describe()
train_data.head()
target_counts = train_metadata.groupby(by='target')['object_id'].count().sort_values(ascending=False)
plt.figure(figsize=(15, 8))
plt.bar(target_counts.index.astype(str), target_counts.values)
plt.xlabel('astro class')
plt.ylabel('occurences in training data')
plt.show()
plt.figure(figsize=(20, 8))
plt.subplot(121, projection='aitoff')
plt.scatter(x=(train_metadata['ra']-180)*math.pi/180, y=train_metadata['decl']*math.pi/180, alpha=0.1)
plt.title('Equatorial coordinates')
plt.subplot(122, projection='aitoff')
plt.scatter(x=(train_metadata['gal_l']-180)*math.pi/180, y=train_metadata['gal_b']*math.pi/180, alpha=0.1)
plt.title('Galactic coordinates')
plt.show()
ddf_metadata = train_metadata[train_metadata['ddf'] == 1].copy()
wfd_metadata = train_metadata[train_metadata['ddf'] != 1].copy()
plt.figure(figsize=(20, 8))
plt.subplot(121, projection='aitoff')
plt.scatter(x=(ddf_metadata['ra']-180)*math.pi/180, y=ddf_metadata['decl']*math.pi/180, alpha=0.1)
plt.title('Equatorial coordinates')
plt.subplot(122, projection='aitoff')
plt.scatter(x=(ddf_metadata['gal_l']-180)*math.pi/180, y=ddf_metadata['gal_b']*math.pi/180, alpha=0.1)
plt.title('Galactic coordinates')
plt.show()
plt.figure(figsize=(20, 5))
plt.barh(y=['DDF', 'WFD'], 
        width=[ddf_metadata['object_id'].count(), wfd_metadata['object_id'].count()])
plt.show()
train_metadata['distmod'].isna().sum()
train_metadata[train_metadata['distmod'].isna()].head(10)
train_metadata[train_metadata['distmod'].isna()]['hostgal_photoz'].describe()
train_metadata['is_milky_way'] = train_metadata['hostgal_photoz'] == 0
outside_sources = train_metadata.loc[~train_metadata['is_milky_way']].copy()
max_distance = outside_sources['distmod'].max()
outside_sources['scaled_distance'] = outside_sources['distmod'] / max_distance

plt.figure(figsize=(20, 8))
plt.gray()
plt.subplot(121, projection='aitoff')
plt.scatter(x=(outside_sources['ra']-180)*math.pi/180, y=outside_sources['decl']*math.pi/180, c=outside_sources['scaled_distance'])
plt.title('Equatorial coordinates')
plt.subplot(122, projection='aitoff')
plt.scatter(x=(outside_sources['gal_l']-180)*math.pi/180, y=outside_sources['gal_b']*math.pi/180, c=outside_sources['scaled_distance'])
plt.title('Galactic coordinates')
plt.show()
outside_sources['mwebv_scaled'] = outside_sources['mwebv'] / outside_sources['mwebv'].max()
s = [4 for i in range(len(outside_sources['mwebv']))]

plt.figure(figsize=(20, 8))
plt.gray()
plt.subplot(121, projection='aitoff')
plt.scatter(x=(outside_sources['ra']-180)*math.pi/180, y=outside_sources['decl']*math.pi/180, c=outside_sources['mwebv_scaled'], s=s)
plt.title('Equatorial coordinates')
plt.subplot(122, projection='aitoff')
plt.scatter(x=(outside_sources['gal_l']-180)*math.pi/180, y=outside_sources['gal_b']*math.pi/180, c=outside_sources['mwebv_scaled'], s=s)
plt.title('Galactic coordinates')
plt.show()
milky_way_sources = train_metadata.loc[train_metadata['is_milky_way']].copy()

plt.figure(figsize=(20, 8))
plt.subplot(121, projection='aitoff')
plt.scatter(x=(milky_way_sources['ra']-180)*math.pi/180, y=milky_way_sources['decl']*math.pi/180, alpha=0.1)
plt.title('Equatorial coordinates')
plt.subplot(122, projection='aitoff')
plt.scatter(x=(milky_way_sources['gal_l']-180)*math.pi/180, y=milky_way_sources['gal_b']*math.pi/180, alpha=0.1)
plt.title('Galactic coordinates')
plt.show()
milky_way_ddf_count = len(train_metadata[(train_metadata['ddf']==1) & (train_metadata['is_milky_way'])])
outside_ddf_count = len(train_metadata[(train_metadata['ddf']==1) & (~train_metadata['is_milky_way'])])

plt.figure(figsize=(15, 5))
plt.barh(y=['Milky Way', 'Outside'], width=[milky_way_ddf_count, outside_ddf_count])
plt.show()
# Convert from Modified Julian Date to normal date
# Unix = (MJD−40587)×86400
train_data['date'] = pd.to_datetime((train_data['mjd'] - 40587) * 86400, unit='s')
light_curve_615 = train_data[train_data['object_id']==615]
light_curve_615['passband'].value_counts()
bands = {0: 'b', 1: 'g', 2: 'r', 3: 'm', 4: 'y', 5: 'k'}
plt.figure(figsize=(15, 8))
for band, color in bands.items():
    plt.errorbar(x=light_curve_615[light_curve_615['passband']==band]['mjd'], 
                 y=light_curve_615[light_curve_615['passband']==band]['flux'], 
                 yerr=light_curve_615[light_curve_615['passband']==band]['flux_err'], 
                 fmt='o', color=color)
plt.plot()
