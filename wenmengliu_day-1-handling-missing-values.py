# import libraries
import pandas as pd 
import numpy as np

# load data
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv('../input/building-permit-applications-data/Building_Permits.csv')

# set seed for reproducibility
np.random.seed(0)
# look at a few rows of the nfl_data flie. I can see a handful of missing 
# data already!
nfl_data.sample(5)
# look at any missing data at a couple rows from the sf_permits 
sf_permits.sample(5)
# get the number of missing data points per column
missing_values_count_nfl = nfl_data.isnull().sum()
missing_values_count_sf = sf_permits.isnull().sum()

# look at the missing points in the first ten columns
missing_values_count_nfl[0:10]
missing_values_count_sf[0:10]
# how many total missing values do we have?
total_cells_nfl = np.product (nfl_data.shape)
total_missing_nfl = missing_value_count.sum()

# percent of nfl data that is missing
(total_missing_nfl/total_cells_nfl)*100
total_cells_sf = np.product(sf_permits.shape)
total_missing_sf = missing_values_count_sf.sum()

# percent of sf data that is missing
(total_missing_sf/total_cells_sf)*100
# look at the missing points in the first ten columns
nfl_data['TimeSecs'].head(10)
sf_permits['Zipcode'].head(10)
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis = 1)
columns_with_na_dropped.head()
# just how much data did we lost?
print('COlumns in original dataset : %d \n' % nfl_data.shape[1])
print('COlumns with na dropped: %d' % columns_with_na_dropped.shape[1])
sf_columns_with_na_dropped = sf_permits.dropna(axis = 1)
sf_columns_with_na_dropped.head()
# how many data did we lose in sf_permits?
print('Columns in original sf_permits dataset: %d\n' % sf_permits.shape[1])
print('Columns in na dropped sf_permits dataset: %d\n' % sf_columns_with_na_dropped.shape[1])
# Get a small subset of the NFL dataset
subset_nfl_data =nfl_data.loc[:,'EPA':'Season'].head()
subset_nfl_data
# replace all NA'S with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column,then replace 
# all the remaining na's with 0
subset_nfl_data.fillna(method = 'bfill',axis=0).fillna(0)
sf_permits.fillna(method='bfill', axis = 0).fillna(0)