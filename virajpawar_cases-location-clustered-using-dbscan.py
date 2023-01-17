# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.cluster import DBSCAN

from shapely.geometry import MultiPoint

from geopy.distance import great_circle

data = pd.read_csv("../input/coronavirusdataset/route.csv")
data.head()
# define map colors

land_color = '#f5f5f3'

water_color = '#cdd2d4'

coastline_color = '#f5f5f3'

border_color = '#bbbbbb'

meridian_color = '#f5f5f3'

marker_fill_color = '#cc3300'

marker_edge_color = 'None'

import pandas as pd, numpy as np, matplotlib.pyplot as plt

from datetime import datetime as dt

from mpl_toolkits.basemap import Basemap

%matplotlib inline

# create the plot

fig = plt.figure(figsize=(20, 10))

ax = fig.add_subplot(111, facecolor = '#ffffff', frame_on = False)





m = Basemap(projection='kav7', lon_0 = 0, resolution = 'c', area_thresh = 10000)

m.drawmapboundary(color = border_color, fill_color = water_color)

m.drawcoastlines(color = coastline_color)

m.drawcountries(color = border_color)

m.fillcontinents(color = land_color, lake_color = water_color)

m.drawparallels(np.arange(-90., 120., 30.), color = meridian_color)

m.drawmeridians(np.arange(0., 420., 60.), color = meridian_color)



x, y = m(data['longitude'].values, data['latitude'].values)

m.scatter(x, y, s=8, color=marker_fill_color, edgecolor=marker_edge_color, alpha=1, zorder=3)
## Cases clustered with parameters as minimum 3 cases in a 5 km radius.



data['latitude_longitude'] = data[['latitude', 'longitude']].apply(tuple, axis=1)

coords_mor = data.as_matrix(columns=['latitude','longitude'])

coords = coords_mor

kms_per_radian = 6371.0088

epsilon = 1/kms_per_radian

db = DBSCAN(eps = epsilon, min_samples = 3, algorithm = 'ball_tree', metric='haversine').fit(np.radians(coords))

cluster_labels = db.labels_

num_clusters = len(set(cluster_labels))

clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])

print('Number of clusters: {}'.format(num_clusters))

clusters_ = clusters.reset_index() 

clusters_ = clusters_.rename(columns={0:'lat_long'})



cluster_data = pd.DataFrame()



for i in range(len(clusters_)):

    cl1 = pd.DataFrame(clusters_.lat_long.iloc[i])

    cl1 = cl1.rename(columns = {0:'latitude',1:'longitude'}) 

    cl1['cluster'] = i

    cl1['latitude_longitude'] = cl1[['latitude', 'longitude']].apply(tuple, axis=1)

    cluster_data = cluster_data.append(cl1)



cluster_data_ = cluster_data.drop_duplicates(subset='latitude_longitude')
cluster_data_.head(100)
cluster_data_.dtypes
## Cases clustered
import numpy as np

import matplotlib.pyplot as plt



x = cluster_data_['latitude'].values

y = cluster_data_['longitude'].values

# c = db.labels_

plt.scatter(x, y, alpha=0.5, c = np.array(cluster_data_.cluster) )

plt.show()
clustered_data = pd.merge(cluster_data_, data,on='latitude_longitude')
clustered_data.head(100)
clustered_data.cluster.value_counts()
max_cases_location = clustered_data.loc[clustered_data['cluster'] == 0]
max_cases_location.city.value_counts()
## City with maximum cases: Jung-gu 
max_cases_location.head(100)
import numpy as np

import matplotlib.pyplot as plt

# type = data_cluster_for_centroid_start.province.tolist()



x = max_cases_location['latitude_x'].values

y = max_cases_location['longitude_x'].values

# c = db.labels_

plt.scatter(x, y, alpha=0.5 )

plt.show()