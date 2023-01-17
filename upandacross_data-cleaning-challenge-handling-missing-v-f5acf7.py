# modules we'll use
import numpy as np
import pandas as pd
import socket

# set seed for reproducibility
np.random.seed(0) 
print(socket.gethostname())
if socket.gethostname() == 'bruntu':
    sf_permits = pd.read_csv('Building_Permits.csv', low_memory=False)
else:
    sf_permits = pd.read_csv('../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv')
if socket.gethostname() == 'bruntu':
    nfl_data = pd.read_csv('NFL Play by Play 2009-2016 (v3).csv', low_memory=False)
else:
    nfl_data = pd.read_csv('../input/building-permit-applications-data/Building_Permits.csv')
# look at a few rows of the nfl_data file. I can see a handful of missing data already!
nfl_data.sample(5)
sf_permits.sample(10)
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
print('{:.2%} of nfl data is missing'.format(total_missing/total_cells))

# your turn! Find out what percent of the sf_permits dataset is missing
print('{:.2%} of building permit application data is missing'\
      .format(sf_permits.isnull().sum().sum()/np.product(sf_permits.shape)))
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
count_rows = len(sf_permits)
my_turn_cols = 'Street Number Suffix|Zipcode'.split('|')
sns, zc = [(key, value / count_rows) 
       for key, value 
       in sf_permits[my_turn_cols].isnull().sum().items()]
print('{:>20s} is {:.2%} NaN and probably should be dropped'.format(*sns))
print('{:>20s} is  {:.2%} NaN and probably a data entry error'.format(*zc))

count_rows = len(sf_permits)
col_null_count = sf_permits.isnull().sum().sort_values(ascending=False, inplace=False)

print('Columns with no Nan columns:')

# don't bother dividing by count_rows
print('\n'.join(['\t{:.2%} {:<s}'.format(pct, col)
                 for col, pct
                 in col_null_count[col_null_count == 0].items()]))

print('Columns with Nan columns by descending percentage:')

# dividing by count_rows for percent
print('\n'.join(['\t{:.2%} {:<s}'.format(pct / count_rows, col) 
                           for col, pct
                 in col_null_count[col_null_count > 0].items()]))

# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
print('bad idea, {} rows left after dropna'.format(sf_permits.dropna().__len__()))
# Now try removing all the columns with empty values. Now how much of your data is left?
colsB4 = sf_permits.columns.__len__()
drop_count = sf_permits.dropna(axis=1).columns.__len__()
print('{} cols after dropna, losing {}'.format(drop_count,
                                               colsB4 - drop_count))

# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0)
# only replace columns that have some NaN's
subset_sf_permits = sf_permits[col_null_count[col_null_count > 0].keys()].sample(10)

# no Nan's left
subset_sf_permits.fillna(method='bfill', axis=1).isna().sum()
