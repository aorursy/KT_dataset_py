# Import libraries

import pandas as pd

import numpy as np

import gc



pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
# Loading data

train_tr = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col = 'TransactionID')

train_id = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', index_col = 'TransactionID')



test_tr = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col = 'TransactionID')

test_id = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv', index_col = 'TransactionID')



# Join train and test datasets

train_df = train_tr.join(train_id)

test_df = test_tr.join(test_id)



# Removing datasets that we don't need anymore

del train_id

del train_tr

del test_id

del test_tr



gc.collect()



print(train_df.shape)

print(test_df.shape)
train_df.head()
train_df.info()
test_df.info()
# Check memory usage of different features

# train_df.memory_usage()
# You can use these commands to see datatypes description

print(np.iinfo('int16'))

print(np.finfo('float64'))
# First I will select only numeric columns

num_cols = [col for col in train_df.columns.values if str(train_df[col].dtype) != 'object']
# To fullfill my curiocity, I'll create small dataframe with minimum and maximum values

types_df = pd.DataFrame({'Col': num_cols, 

              'min': [train_df[col].min() for col in num_cols],

              'max': [train_df[col].max() for col in num_cols],

              'dtype': [str(train_df[col].dtype) for col in num_cols]})



types_df['dtype_min'] = types_df['dtype'].map({'int64': np.iinfo('int64').min, 'float64': np.finfo('float64').min})

types_df['dtype_max'] = types_df['dtype'].map({'int64': np.iinfo('int64').max, 'float64': np.finfo('float64').max})

types_df.sample(20)
def reduce_size(dataset):

    for col in dataset.columns.values:

        if str(dataset[col].dtype) == 'object':

            # Change object to category if needed

#             dataset[col] = dataset[col].astype('category')

            continue

        elif str(dataset[col].dtype)[:3] == 'int':                    

            dataset[col] = pd.to_numeric(dataset[col], downcast = 'integer')

        else:   

            dataset[col] = pd.to_numeric(dataset[col], downcast = 'float')

        

    return dataset
train_df = reduce_size(train_df)

test_df = reduce_size(test_df)
train_df.info()
test_df.info()