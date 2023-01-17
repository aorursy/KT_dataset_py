# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_i = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

train_t = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train = pd.merge(left=train_t, right=train_i, on='TransactionID', how='left')



print(train.shape)
del train_i, train_t

gc.collect()
print(train.info(memory_usage='deep'))
# Check memory usage by type

for dtype in ['float','int','object']:

    selected_dtype = train.select_dtypes(include=[dtype])

    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()

    mean_usage_mb = mean_usage_b / 1024**2

    print('Average memory usage for {} columns: {:03.2f} MB'.format(dtype, mean_usage_mb))
def mem_usage(pandas_obj):

    if isinstance(pandas_obj, pd.DataFrame):

        usage_b = pandas_obj.memory_usage(deep=True).sum()

    else:

        usage_b = pandas_obj.memory_usage(deep=True)

    

    usage_mb = usage_b / 1024**2

    return "{:03.2f} MB".format(usage_mb)
int_types = ['uint8', 'int8', 'int16', 'int32']

for it in int_types:

    print(np.iinfo(it))
train_int = train.select_dtypes(include=['int'])

converted_int = train_int.apply(pd.to_numeric, downcast='unsigned')
print('`int` type: ')

print('Before:',mem_usage(train_int))

print('After:',mem_usage(converted_int))
compare_ints = pd.concat([train_int.dtypes, converted_int.dtypes], axis=1)

compare_ints.columns = ['before', 'after']

compare_ints.apply(pd.Series.value_counts, axis=0)
train_float = train.select_dtypes(include=['float'])

converted_float = train_float.apply(pd.to_numeric, downcast='float')



print('`float` type:')

print('Before:', mem_usage(train_float))

print('After:', mem_usage(converted_float))
### 

compare_floats = pd.concat([train_float.dtypes, converted_float.dtypes], axis=1)

compare_floats.columns = ['before', 'after']

compare_floats.apply(pd.Series.value_counts, axis=0)
train_obj = train.select_dtypes(include=['object']).copy()

max_uniq = max(train_obj.apply(pd.Series.nunique))

print('The max unique values in `object` columns is:', max_uniq)

print('The corresponding percentage is: {:02.2f}%'.format(max_uniq/train_obj.shape[0]*100))
converted_obj = train_obj.apply(pd.Series.astype, dtype='category')



print('`object` type: ')

print('Before:',mem_usage(train_obj))

print('After:',mem_usage(converted_obj))
optimized_train = train.copy()

optimized_train[converted_int.columns] = converted_int

optimized_train[converted_float.columns] = converted_float

optimized_train[converted_obj.columns] = converted_obj



print('Before memory reduction:', mem_usage(train))

print('After memory reduction:', mem_usage(optimized_train))
del train_int, train_float, train_obj

del converted_int, converted_float, converted_obj

gc.collect()