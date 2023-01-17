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
## All columns
''' 'date_time', 'site_name', 'posa_continent', 'user_location_country',
    'user_location_region', 'user_location_city',
    'orig_destination_distance', 'user_id', 'is_mobile', 'is_package',
    'channel', 'srch_ci', 'srch_co', 'srch_adults_cnt', 'srch_children_cnt',
    'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id',
    'is_booking', 'cnt', 'hotel_continent', 'hotel_country', 'hotel_market',
    'hotel_cluster'
'''
## My opinion - mandatory columns
'''
srch_destination_id, is_booking, hotel_cluster, srch_adults_cnt
'''
NUMBER_OF_ROWS = 1000 # Train data is too big, get some rows
train_df = pd.read_csv('/kaggle/input/expedia-hotel-recommendations/train.csv', nrows=NUMBER_OF_ROWS)

# Aggregation data
groupby1 = train_df.groupby(['srch_destination_id', 'hotel_cluster'])['is_booking'].agg(['count'])
groupby1.head()
# Convert single index dataframe
single_index_df = groupby1.reset_index(level=[0,1])
single_index_df
# Collect hotel_cluster as list
def list_2_str(items):
    if (items is None) or (len(items) <= 0):
        return ''
    result = ''
    for item in items:
        result = result + str(item) + ','
    return result[:(len(result) - 1)]

total_count_of_hotel_cluster = 0 
destination_id_n_cluster_list = dict()
for index, row in single_index_df.iterrows():
    srch_destination_id = row['srch_destination_id']
    hotel_cluster = row['hotel_cluster']
    
    hotel_clusters = list()
    if srch_destination_id in destination_id_n_cluster_list:
        hotel_clusters = destination_id_n_cluster_list[srch_destination_id]
    hotel_clusters.append(hotel_cluster)
    total_count_of_hotel_cluster += 1
    destination_id_n_cluster_list[srch_destination_id] = hotel_clusters

destination_id_n_clusters = dict()
for key, value in destination_id_n_cluster_list.items():
    str_value = list_2_str(value)
    destination_id_n_clusters[key] = str_value
# Convert dict_list to dataframe
final_df = pd.DataFrame(destination_id_n_clusters.items(), columns=['srch_destination_id', 'hotel_clusters'])
final_df.head()
NUMBER_OF_ROWS = 5000 # Train data is too big, get some rows
test_df = pd.read_csv('/kaggle/input/expedia-hotel-recommendations/train.csv', nrows=NUMBER_OF_ROWS, usecols=['srch_destination_id'])
test_df.head()
merged_test_df = test_df.merge(final_df, how = 'left')
merged_test_df[['hotel_clusters']] = merged_test_df[['hotel_clusters']].fillna(value = 'NA')
merged_test_df.head(10)
import matplotlib.pyplot as plt

plt.scatter(merged_test_df['srch_destination_id'].values, merged_test_df['hotel_clusters'].values)
plt.show()
# Aggregation data
NUMBER_OF_ROWS = 5000 # Train data is too big, get some rows
train_df = pd.read_csv('/kaggle/input/expedia-hotel-recommendations/train.csv', nrows=NUMBER_OF_ROWS)
test_df = pd.read_csv('/kaggle/input/expedia-hotel-recommendations/test.csv', nrows=NUMBER_OF_ROWS)
k_nearest_train_points = train_df[['srch_destination_id']]
k_nearest_train_labels = train_df[['hotel_cluster']]
k_nearest_test_points = test_df[['srch_destination_id']]
from sklearn.neighbors import KNeighborsClassifier

k_nearest_classifier = KNeighborsClassifier(n_neighbors = 100)
k_nearest_classifier.fit(k_nearest_train_points, k_nearest_train_labels)
k_nearest_result = k_nearest_classifier.predict(k_nearest_test_points)
plt.scatter(k_nearest_test_points, k_nearest_result, s=50, alpha=0.5)
plt.show()
NUMBER_OF_ROWS = 5000 # Train data is too big, get some rows
train_df = pd.read_csv('/kaggle/input/expedia-hotel-recommendations/train.csv', nrows=NUMBER_OF_ROWS)
test_df = pd.read_csv('/kaggle/input/expedia-hotel-recommendations/test.csv', nrows=NUMBER_OF_ROWS)

agg_cluster_X = train_df[['srch_destination_id']]
agg_cluster_Y = train_df[['hotel_cluster']]
agg_cluster_destination_id = test_df[['srch_destination_id']]
from sklearn.cluster import AgglomerativeClustering

agglomerativeCluster = AgglomerativeClustering(n_clusters = 100, linkage = 'ward')
agglomerativeCluster.fit(agg_cluster_X, agg_cluster_Y)
agg_recommend_hotel = agglomerativeCluster.fit_predict(agg_cluster_destination_id)
plt.scatter(agg_cluster_destination_id, agg_recommend_hotel, s = 50, alpha=0.5)
plt.show()