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
neighbor_info_df = pd.read_csv('../input/2019-2nd-ml-month-with-kakr-neighbor-info/neighbor_info.csv')

print(neighbor_info_df.shape)

neighbor_info_df.head()
neighbor_1km = neighbor_info_df[(neighbor_info_df['data'] == 'train') & (neighbor_info_df['distance'] <= 0.5)]

neighbor_1km['neighbor_price_log'] = np.log1p(neighbor_1km['neighbor_price'])

neighbor_1km_stat = neighbor_1km.groupby('id').agg({

    'neighbor_id': 'count',

    'distance': ['min','max','mean','median','std','skew'],

    'neighbor_price_log': ['mean'],

    'neighbor_sqft_living': ['min','max','mean','median','std','skew'],

    'neighbor_sqft_lot': ['min','max','mean','median','std','skew'],

    'neighbor_bedrooms': ['min','max','mean','median','std','skew'],

    'neighbor_bathrooms': ['min','max','mean','median','std','skew'],

    'neighbor_grade': ['min','max','mean','median','std','skew'],

    'neighbor_view': ['min','max','mean','median','std','skew'],

    'neighbor_condition': ['min','max','mean','median','std','skew'],

}).reset_index()



neighbor_1km_stat.columns = [

    'id','neighbor_1km_count',

    'nb_1km_distance_min','nb_1km_distance_max','nb_1km_distance_mean','nb_1km_distance_median','nb_1km_distance_std','nb_1km_distance_skew',

    'nb_1km_price_mean',

    'nb_1km_sqft_living_min','nb_1km_sqft_living_max','nb_1km_sqft_living_mean','nb_1km_sqft_living_median','nb_1km_sqft_living_std','nb_1km_sqft_living_skew',

    'nb_1km_sqft_lot_min','nb_1km_sqft_lot_max','nb_1km_sqft_lot_mean','nb_1km_sqft_lot_median','nb_1km_sqft_lot_std','nb_1km_sqft_lot_skew',

    'nb_1km_bedrooms_min','nb_1km_bedrooms_max','nb_1km_bedrooms_mean','nb_1km_bedrooms_median','nb_1km_bedrooms_std','nb_1km_bedrooms_skew',

    'nb_1km_bathrooms_min','nb_1km_bathrooms_max','nb_1km_bathrooms_mean','nb_1km_bathrooms_median','nb_1km_bathrooms_std','nb_1km_bathrooms_skew',

    'nb_1km_grade_min','nb_1km_grade_max','nb_1km_grade_mean','nb_1km_grade_median','nb_1km_grade_std','nb_1km_grade_skew',

    'nb_1km_view_min','nb_1km_view_max','nb_1km_view_mean','nb_1km_view_median','nb_1km_view_std','nb_1km_view_skew',

    'nb_1km_condition_min','nb_1km_condition_max','nb_1km_condition_mean','nb_1km_condition_median','nb_1km_condition_std','nb_1km_condition_skew',

]



print(neighbor_1km_stat.shape)

neighbor_1km_stat.head()
neighbor_1km_stat.to_csv('neighbor_1km_stat.csv', index=False)
neighbor_3km = neighbor_info_df[(neighbor_info_df['data'] == 'train') & (neighbor_info_df['distance'] <= 1.5)]

neighbor_3km['neighbor_price_log'] = np.log1p(neighbor_3km['neighbor_price'])

neighbor_3km_stat = neighbor_3km.groupby('id').agg({

    'neighbor_id': 'count',

    'distance': ['min','max','mean','median','std','skew'],

    'neighbor_price_log': ['mean'],

    'neighbor_sqft_living': ['min','max','mean','median','std','skew'],

    'neighbor_sqft_lot': ['min','max','mean','median','std','skew'],

    'neighbor_bedrooms': ['min','max','mean','median','std','skew'],

    'neighbor_bathrooms': ['min','max','mean','median','std','skew'],

    'neighbor_grade': ['min','max','mean','median','std','skew'],

    'neighbor_view': ['min','max','mean','median','std','skew'],

    'neighbor_condition': ['min','max','mean','median','std','skew'],

}).reset_index()



neighbor_3km_stat.columns = [

    'id','neighbor_3km_count',

    'nb_3km_distance_min','nb_3km_distance_max','nb_3km_distance_mean','nb_3km_distance_median','nb_3km_distance_std','nb_3km_distance_skew',

    'nb_3km_price_mean',

    'nb_3km_sqft_living_min','nb_3km_sqft_living_max','nb_3km_sqft_living_mean','nb_3km_sqft_living_median','nb_3km_sqft_living_std','nb_3km_sqft_living_skew',

    'nb_3km_sqft_lot_min','nb_3km_sqft_lot_max','nb_3km_sqft_lot_mean','nb_3km_sqft_lot_median','nb_3km_sqft_lot_std','nb_3km_sqft_lot_skew',

    'nb_3km_bedrooms_min','nb_3km_bedrooms_max','nb_3km_bedrooms_mean','nb_3km_bedrooms_median','nb_3km_bedrooms_std','nb_3km_bedrooms_skew',

    'nb_3km_bathrooms_min','nb_3km_bathrooms_max','nb_3km_bathrooms_mean','nb_3km_bathrooms_median','nb_3km_bathrooms_std','nb_3km_bathrooms_skew',

    'nb_3km_grade_min','nb_3km_grade_max','nb_3km_grade_mean','nb_3km_grade_median','nb_3km_grade_std','nb_3km_grade_skew',

    'nb_3km_view_min','nb_3km_view_max','nb_3km_view_mean','nb_3km_view_median','nb_3km_view_std','nb_3km_view_skew',

    'nb_3km_condition_min','nb_3km_condition_max','nb_3km_condition_mean','nb_3km_condition_median','nb_3km_condition_std','nb_3km_condition_skew',

]



print(neighbor_3km_stat.shape)

neighbor_3km_stat.head()
neighbor_3km_stat.to_csv('neighbor_3km_stat.csv', index=False)
neighbor_5km = neighbor_info_df[(neighbor_info_df['data'] == 'train') & (neighbor_info_df['distance'] <= 2.5)]

neighbor_5km['neighbor_price_log'] = np.log1p(neighbor_5km['neighbor_price'])

neighbor_5km_stat = neighbor_5km.groupby('id').agg({

    'neighbor_id': 'count',

    'distance': ['min','max','mean','median','std','skew'],

    'neighbor_price_log': ['mean'],

    'neighbor_sqft_living': ['min','max','mean','median','std','skew'],

    'neighbor_sqft_lot': ['min','max','mean','median','std','skew'],

    'neighbor_bedrooms': ['min','max','mean','median','std','skew'],

    'neighbor_bathrooms': ['min','max','mean','median','std','skew'],

    'neighbor_grade': ['min','max','mean','median','std','skew'],

    'neighbor_view': ['min','max','mean','median','std','skew'],

    'neighbor_condition': ['min','max','mean','median','std','skew'],

}).reset_index()



neighbor_5km_stat.columns = [

    'id','neighbor_5km_count',

    'nb_5km_distance_min','nb_5km_distance_max','nb_5km_distance_mean','nb_5km_distance_median','nb_5km_distance_std','nb_5km_distance_skew',

    'nb_5km_price_mean',

    'nb_5km_sqft_living_min','nb_5km_sqft_living_max','nb_5km_sqft_living_mean','nb_5km_sqft_living_median','nb_5km_sqft_living_std','nb_5km_sqft_living_skew',

    'nb_5km_sqft_lot_min','nb_5km_sqft_lot_max','nb_5km_sqft_lot_mean','nb_5km_sqft_lot_median','nb_5km_sqft_lot_std','nb_5km_sqft_lot_skew',

    'nb_5km_bedrooms_min','nb_5km_bedrooms_max','nb_5km_bedrooms_mean','nb_5km_bedrooms_median','nb_5km_bedrooms_std','nb_5km_bedrooms_skew',

    'nb_5km_bathrooms_min','nb_5km_bathrooms_max','nb_5km_bathrooms_mean','nb_5km_bathrooms_median','nb_5km_bathrooms_std','nb_5km_bathrooms_skew',

    'nb_5km_grade_min','nb_5km_grade_max','nb_5km_grade_mean','nb_5km_grade_median','nb_5km_grade_std','nb_5km_grade_skew',

    'nb_5km_view_min','nb_5km_view_max','nb_5km_view_mean','nb_5km_view_median','nb_5km_view_std','nb_5km_view_skew',

    'nb_5km_condition_min','nb_5km_condition_max','nb_5km_condition_mean','nb_5km_condition_median','nb_5km_condition_std','nb_5km_condition_skew',

]



print(neighbor_5km_stat.shape)

neighbor_5km_stat.head()
neighbor_5km_stat.to_csv('neighbor_5km_stat.csv', index=False)
nearest_neighbor = neighbor_info_df[(neighbor_info_df['data'] == 'train') & (neighbor_info_df['distance'] <= 4)]

nearest_neighbor = nearest_neighbor.sort_values(['id','grade_diff','sqft_living_diff','sqft_living15_diff',

                                                 'bathrooms_diff','distance','bedrooms_diff','view_diff','sqft_lot_diff',

                                                 'condition_diff','waterfront_diff'])

nearest_neighbor['nb_order'] = nearest_neighbor.groupby(['id']).cumcount() + 1

print(nearest_neighbor.shape)

nearest_neighbor.head()
nearest_5_neighbor = nearest_neighbor[nearest_neighbor['nb_order'] <= 5].reset_index(drop=True)

nearest_5_neighbor['neighbor_price_log'] = np.log1p(nearest_5_neighbor['neighbor_price'])



nearest_5_neighbor_stat = nearest_5_neighbor.groupby('id').agg({

    'neighbor_id': 'count',

    'distance': ['min','max','mean','median','std','skew'],

    'neighbor_price_log': ['mean'],

    'neighbor_sqft_living': ['min','max','mean','median','std','skew'],

    'neighbor_sqft_lot': ['min','max','mean','median','std','skew'],

    'neighbor_bedrooms': ['min','max','mean','median','std','skew'],

    'neighbor_bathrooms': ['min','max','mean','median','std','skew'],

    'neighbor_grade': ['min','max','mean','median','std','skew'],

    'neighbor_view': ['min','max','mean','median','std','skew'],

    'neighbor_condition': ['min','max','mean','median','std','skew'],

}).reset_index()



nearest_5_neighbor_stat.columns = [

    'id','n_5_nb_count',

    'n_5_nb_distance_min','n_5_nb_distance_max','n_5_nb_distance_mean','n_5_nb_distance_median','n_5_nb_distance_std','n_5_nb_distance_skew',

    'n_5_nb_price_mean',

    'n_5_nb_sqft_living_min','n_5_nb_sqft_living_max','n_5_nb_sqft_living_mean','n_5_nb_sqft_living_median','n_5_nb_sqft_living_std','n_5_nb_sqft_living_skew',

    'n_5_nb_sqft_lot_min','n_5_nb_sqft_lot_max','n_5_nb_sqft_lot_mean','n_5_nb_sqft_lot_median','n_5_nb_sqft_lot_std','n_5_nb_sqft_lot_skew',

    'n_5_nb_bedrooms_min','n_5_nb_bedrooms_max','n_5_nb_bedrooms_mean','n_5_nb_bedrooms_median','n_5_nb_bedrooms_std','n_5_nb_bedrooms_skew',

    'n_5_nb_bathrooms_min','n_5_nb_bathrooms_max','n_5_nb_bathrooms_mean','n_5_nb_bathrooms_median','n_5_nb_bathrooms_std','n_5_nb_bathrooms_skew',

    'n_5_nb_grade_min','n_5_nb_grade_max','n_5_nb_grade_mean','n_5_nb_grade_median','n_5_nb_grade_std','n_5_nb_grade_skew',

    'n_5_nb_view_min','n_5_nb_view_max','n_5_nb_view_mean','n_5_nb_view_median','n_5_nb_view_std','n_5_nb_view_skew',

    'n_5_nb_condition_min','n_5_nb_condition_max','n_5_nb_condition_mean','n_5_nb_condition_median','n_5_nb_condition_std','n_5_nb_condition_skew',

]



print(nearest_5_neighbor_stat.shape)

nearest_5_neighbor_stat.head()
nearest_5_neighbor_stat.to_csv('nearest_5_neighbor_stat.csv', index=False)
nearest_10_neighbor = nearest_neighbor[nearest_neighbor['nb_order'] <= 10].reset_index(drop=True)

nearest_10_neighbor['neighbor_price_log'] = np.log1p(nearest_10_neighbor['neighbor_price'])



nearest_10_neighbor_stat = nearest_10_neighbor.groupby('id').agg({

    'neighbor_id': 'count',

    'distance': ['min','max','mean','median','std','skew'],

    'neighbor_price_log': ['mean'],

    'neighbor_sqft_living': ['min','max','mean','median','std','skew'],

    'neighbor_sqft_lot': ['min','max','mean','median','std','skew'],

    'neighbor_bedrooms': ['min','max','mean','median','std','skew'],

    'neighbor_bathrooms': ['min','max','mean','median','std','skew'],

    'neighbor_grade': ['min','max','mean','median','std','skew'],

    'neighbor_view': ['min','max','mean','median','std','skew'],

    'neighbor_condition': ['min','max','mean','median','std','skew'],

}).reset_index()



nearest_10_neighbor_stat.columns = [

    'id','n_10_nb_count',

    'n_10_nb_distance_min','n_10_nb_distance_max','n_10_nb_distance_mean','n_10_nb_distance_median','n_10_nb_distance_std','n_10_nb_distance_skew',

    'n_10_nb_price_mean',

    'n_10_nb_sqft_living_min','n_10_nb_sqft_living_max','n_10_nb_sqft_living_mean','n_10_nb_sqft_living_median','n_10_nb_sqft_living_std','n_10_nb_sqft_living_skew',

    'n_10_nb_sqft_lot_min','n_10_nb_sqft_lot_max','n_10_nb_sqft_lot_mean','n_10_nb_sqft_lot_median','n_10_nb_sqft_lot_std','n_10_nb_sqft_lot_skew',

    'n_10_nb_bedrooms_min','n_10_nb_bedrooms_max','n_10_nb_bedrooms_mean','n_10_nb_bedrooms_median','n_10_nb_bedrooms_std','n_10_nb_bedrooms_skew',

    'n_10_nb_bathrooms_min','n_10_nb_bathrooms_max','n_10_nb_bathrooms_mean','n_10_nb_bathrooms_median','n_10_nb_bathrooms_std','n_10_nb_bathrooms_skew',

    'n_10_nb_grade_min','n_10_nb_grade_max','n_10_nb_grade_mean','n_10_nb_grade_median','n_10_nb_grade_std','n_10_nb_grade_skew',

    'n_10_nb_view_min','n_10_nb_view_max','n_10_nb_view_mean','n_10_nb_view_median','n_10_nb_view_std','n_10_nb_view_skew',

    'n_10_nb_condition_min','n_10_nb_condition_max','n_10_nb_condition_mean','n_10_nb_condition_median','n_10_nb_condition_std','n_10_nb_condition_skew',

]



print(nearest_10_neighbor_stat.shape)

nearest_10_neighbor_stat.head()
nearest_10_neighbor_stat.to_csv('nearest_10_neighbor_stat.csv', index=False)
nearest_20_neighbor = nearest_neighbor[nearest_neighbor['nb_order'] <= 20].reset_index(drop=True)

nearest_20_neighbor['neighbor_price_log'] = np.log1p(nearest_20_neighbor['neighbor_price'])



nearest_20_neighbor_stat = nearest_20_neighbor.groupby('id').agg({

    'neighbor_id': 'count',

    'distance': ['min','max','mean','median','std','skew'],

    'neighbor_price_log': ['mean'],

    'neighbor_sqft_living': ['min','max','mean','median','std','skew'],

    'neighbor_sqft_lot': ['min','max','mean','median','std','skew'],

    'neighbor_bedrooms': ['min','max','mean','median','std','skew'],

    'neighbor_bathrooms': ['min','max','mean','median','std','skew'],

    'neighbor_grade': ['min','max','mean','median','std','skew'],

    'neighbor_view': ['min','max','mean','median','std','skew'],

    'neighbor_condition': ['min','max','mean','median','std','skew'],

}).reset_index()



nearest_20_neighbor_stat.columns = [

    'id','n_20_nb_count',

    'n_20_nb_distance_min','n_20_nb_distance_max','n_20_nb_distance_mean','n_20_nb_distance_median','n_20_nb_distance_std','n_20_nb_distance_skew',

    'n_20_nb_price_mean',

    'n_20_nb_sqft_living_min','n_20_nb_sqft_living_max','n_20_nb_sqft_living_mean','n_20_nb_sqft_living_median','n_20_nb_sqft_living_std','n_20_nb_sqft_living_skew',

    'n_20_nb_sqft_lot_min','n_20_nb_sqft_lot_max','n_20_nb_sqft_lot_mean','n_20_nb_sqft_lot_median','n_20_nb_sqft_lot_std','n_20_nb_sqft_lot_skew',

    'n_20_nb_bedrooms_min','n_20_nb_bedrooms_max','n_20_nb_bedrooms_mean','n_20_nb_bedrooms_median','n_20_nb_bedrooms_std','n_20_nb_bedrooms_skew',

    'n_20_nb_bathrooms_min','n_20_nb_bathrooms_max','n_20_nb_bathrooms_mean','n_20_nb_bathrooms_median','n_20_nb_bathrooms_std','n_20_nb_bathrooms_skew',

    'n_20_nb_grade_min','n_20_nb_grade_max','n_20_nb_grade_mean','n_20_nb_grade_median','n_20_nb_grade_std','n_20_nb_grade_skew',

    'n_20_nb_view_min','n_20_nb_view_max','n_20_nb_view_mean','n_20_nb_view_median','n_20_nb_view_std','n_20_nb_view_skew',

    'n_20_nb_condition_min','n_20_nb_condition_max','n_20_nb_condition_mean','n_20_nb_condition_median','n_20_nb_condition_std','n_20_nb_condition_skew',

]



print(nearest_20_neighbor_stat.shape)

nearest_20_neighbor_stat.head()
nearest_20_neighbor_stat.to_csv('nearest_20_neighbor_stat.csv', index=False)