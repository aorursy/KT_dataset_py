import pandas as pd

import numpy as np

import os
# safe downcast

def sd(col, max_loss_limit=0.001, avg_loss_limit=0.001, na_loss_limit=0, n_uniq_loss_limit=0, fillna=0):

    """

    max_loss_limit - don't allow any float to lose precision more than this value. Any values are ok for GBT algorithms as long as you don't unique values.

                     See https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_decimal_values_in_[0,_1]

    avg_loss_limit - same but calculates avg throughout the series.

    na_loss_limit - not really useful.

    n_uniq_loss_limit - very important parameter. If you have a float field with very high cardinality you can set this value to something like n_records * 0.01 in order to allow some field relaxing.

    """

    is_float = str(col.dtypes)[:5] == 'float'

    na_count = col.isna().sum()

    n_uniq = col.nunique(dropna=False)

    try_types = ['float16', 'float32']



    if na_count <= na_loss_limit:

        try_types = ['int8', 'int16', 'float16', 'int32', 'float32']



    for type in try_types:

        col_tmp = col



        # float to int conversion => try to round to minimize casting error

        if is_float and (str(type)[:3] == 'int'):

            col_tmp = col_tmp.copy().fillna(fillna).round()



        col_tmp = col_tmp.astype(type)

        max_loss = (col_tmp - col).abs().max()

        avg_loss = (col_tmp - col).abs().mean()

        na_loss = np.abs(na_count - col_tmp.isna().sum())

        n_uniq_loss = np.abs(n_uniq - col_tmp.nunique(dropna=False))



        if max_loss <= max_loss_limit and avg_loss <= avg_loss_limit and na_loss <= na_loss_limit and n_uniq_loss <= n_uniq_loss_limit:

            return col_tmp



    # field can't be converted

    return col





def reduce_mem_usage_sd(df, deep=True, verbose=False, obj_to_cat=False):

    numerics = ['int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2

    for col in df.columns:

        col_type = df[col].dtypes



        # collect stats

        na_count = df[col].isna().sum()

        n_uniq = df[col].nunique(dropna=False)

        

        # numerics

        if col_type in numerics:

            df[col] = sd(df[col])



        # strings

        if (col_type == 'object') and obj_to_cat:

            df[col] = df[col].astype('category')

        

        if verbose:

            print(f'Column {col}: {col_type} -> {df[col].dtypes}, na_count={na_count}, n_uniq={n_uniq}')

        new_na_count = df[col].isna().sum()

        if (na_count != new_na_count):

            print(f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost na values. Before: {na_count}, after: {new_na_count}')

        new_n_uniq = df[col].nunique(dropna=False)

        if (n_uniq != new_n_uniq):

            print(f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost unique values. Before: {n_uniq}, after: {new_n_uniq}')



    end_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2

    percent = 100 * (start_mem - end_mem) / start_mem

    if verbose:

        print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))

    return df



def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2 # just added 

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    percent = 100 * (start_mem - end_mem) / start_mem

    print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))

    return df
# Load Dataset

def load_dataset():

    path = '../input/ashrae-energy-prediction/'

    train = pd.read_csv(path+'train.csv')

    test  = pd.read_csv(path+'test.csv')

    weather_train = pd.read_csv(path+'weather_train.csv')

    weather_test  = pd.read_csv(path+'weather_test.csv')

    building_metadata = pd.read_csv(path+'building_metadata.csv')

    return train, test, weather_train, weather_test, building_metadata



%time raw_train, raw_test, raw_weather_train, raw_weather_test, raw_building_metadata = load_dataset()
# Copy each table, to compare whether the transformed values macth with the original ones.

train = raw_train.copy()

test  = raw_test.copy()

weather_train = raw_weather_train.copy()

weather_test  = raw_weather_test.copy()

building_metadata = raw_building_metadata.copy()
# Implement `reduce_mem_usage`

train = reduce_mem_usage(train)

test = reduce_mem_usage(test)

weather_train = reduce_mem_usage(weather_train)

weather_test = reduce_mem_usage(weather_test)

building_metadata = reduce_mem_usage(building_metadata)
# Check whether the transformed values exactly match with the original ones.

table_names = ['train','test','weather_train','weather_test', 'building_metadata']

raw_tables = [ raw_train, raw_test, raw_weather_train, raw_weather_test, raw_building_metadata ]

transformed_tables = [ train, test, weather_train , weather_test, building_metadata]





for table_name, raw_table, transformed_table in zip( table_names, raw_tables, transformed_tables  ):

    print('<' + table_name.upper() +'>')

    for column in list(raw_table):

        print(f"Columns Name : {column}")

        if np.mean( raw_table[column] == transformed_table[column] ) == 1:

            print("\tperfectly matches with the originals")

        else:

            print("\tnot perfectly matches with the originals")

            print("\tBad Transformation : {:.2f}%".format(100 * (1 -np.mean( raw_table[column] == transformed_table[column] ) )))

    print('='*100)
# Copy each table, to compare whether the transformed values macth with the original ones.

train = raw_train.copy()

test  = raw_test.copy()

weather_train = raw_weather_train.copy()

weather_test  = raw_weather_test.copy()

building_metadata = raw_building_metadata.copy()
# Implement `reduce_mem_usage_sd`

train = reduce_mem_usage_sd(train, verbose=True)

print('='*80)

test = reduce_mem_usage_sd(test, verbose=True)

print('='*80)

weather_train = reduce_mem_usage_sd(weather_train, verbose=True)

print('='*80)

weather_test = reduce_mem_usage_sd(weather_test, verbose=True)

print('='*80)

raw_building_metadata = reduce_mem_usage_sd(building_metadata, verbose=True)
# Check whether the transformed values exactly match with the original ones.

table_names = ['train','test','weather_train','weather_test', 'building_metadata']

raw_tables = [ raw_train, raw_test, raw_weather_train, raw_weather_test, raw_building_metadata ]

transformed_tables = [ train, test, weather_train , weather_test, building_metadata]





for table_name, raw_table, transformed_table in zip( table_names, raw_tables, transformed_tables  ):

    print('<' + table_name.upper() +'>')

    for column in list(raw_table):

        print(f"Columns Name : {column}")

        if np.mean( raw_table[column] == transformed_table[column] ) == 1:

            print("\tperfectly matches with the originals")

        else:

            print("\tnot perfectly matches with the originals")

            print("\tBad Transformation : {:.2f}%".format(100 * (1 -np.mean( raw_table[column] == transformed_table[column] ) )))

    print('='*100)