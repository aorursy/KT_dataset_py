# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



pd.options.display.max_rows = 10000

pd.options.display.max_columns = 10000

pd.options.display.max_colwidth = 1000



import os

import gc

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train_copy = train.copy()

train_copy['data'] = 'train'



# remove outlier

train_copy = train_copy[~((train_copy['sqft_living'] > 12000) & (train_copy['price'] < 3000000))].reset_index(drop=True)

    

test_copy = test.copy()

test_copy['data'] = 'test'

test_copy['price'] = np.nan



data = pd.concat([train_copy, test_copy], sort=False).reset_index(drop=True)

data = data[train_copy.columns]

data.head()
def haversine_array(lat1, lng1, lat2, lng2): 

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2)) 

    AVG_EARTH_RADIUS = 6371 # in km 

    lat = lat2 - lat1 

    lng = lng2 - lng1 

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2 

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d)) 

    return h
print(data['lat'].min(), data['lat'].max(), data['long'].min(), data['long'].max())



haversine_array(data['lat'].min(), data['long'].min(), data['lat'].max(), data['long'].max())
from tqdm import tqdm_notebook as tqdm



neighbor_df = pd.DataFrame()

lat2 = data['lat'].values

long2 = data['long'].values



for i, row in tqdm(data.iterrows(), total=data.shape[0]):

    lat1 = np.array(row['lat'])

    long1 = np.array(row['long'])

    dist_arr = haversine_array(lat1, long1, lat2, long2)

    tmp_df = pd.DataFrame({'id': np.tile(np.array([row['id']]), data.shape[0]), 'neighbor_id': data['id'], 'distance':dist_arr})

    tmp_df = tmp_df[tmp_df['distance'] <= 4]

    tmp_df = tmp_df[tmp_df['id'] != tmp_df['neighbor_id']]

    neighbor_df = neighbor_df.append(tmp_df.copy())

    del tmp_df

    gc.collect()

    

print(neighbor_df.shape)

neighbor_df.head()
#neighbor_df.to_csv('neighbor.csv', index=False)
data_df = data.rename(index=str, columns={'id': 'neighbor_id'})

neighbor_info_df = neighbor_df.merge(data_df[['neighbor_id','sqft_living','sqft_living15','sqft_lot','sqft_lot15',

                                              'bedrooms','bathrooms','grade','waterfront','view','condition',

                                              'data','price']], on='neighbor_id')

neighbor_info_df.columns = ['id','neighbor_id','distance','neighbor_sqft_living','neighbor_sqft_living15',

                            'neighbor_sqft_lot','neighbor_sqft_lot15','neighbor_bedrooms','neighbor_bathrooms',

                            'neighbor_grade','neighbor_waterfront','neighbor_view','neighbor_condition',

                            'data','neighbor_price']

neighbor_info_df = neighbor_info_df.sort_values(['id','neighbor_id']).reset_index(drop=True)

print(neighbor_info_df.shape)

neighbor_info_df.head()
neighbor_info_df = neighbor_info_df.merge(data[['id','sqft_living','sqft_living15','sqft_lot','sqft_lot15',

                                                'bedrooms','bathrooms','grade','waterfront','view','condition',

                                                'price']], on='id').reset_index(drop=True)

print(neighbor_info_df.shape)

neighbor_info_df.head()
cols = ['sqft_living','sqft_living15','sqft_lot','sqft_lot15','bedrooms','bathrooms','grade','waterfront','view','condition']



for col in cols:

    neighbor_info_df[col + '_diff'] = abs(neighbor_info_df[col] - neighbor_info_df['neighbor_' + col])



print(neighbor_info_df.shape)

neighbor_info_df.head()
neighbor_info_df.to_csv('neighbor_info.csv', index=False)