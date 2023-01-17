# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 
# look at a few rows of the nfl_data file. I can see a handful of missing data already!
nfl_data.sample(5)
# your turn! Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?
sf_permits.sample(10)
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
# The shape attribute for numpy arrays returns the dimensions of the array. If Y has  n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n.
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permits dataset is missing
missing_value_count1 = sf_permits.isnull().sum()
missing_value_count1[0:10]



total_cells = np.product(sf_permits.shape)
total_missing = missing_value_count.sum()
(total_missing/total_cells)*100
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
# axis : {0 or ‘index’, 1 or ‘columns’}, default 0. Determine if rows or columns which contain missing values are removed. 0, or ‘index’ : Drop rows which contain missing values. 1, or ‘columns’ : Drop columns which contain missing value.
# how : {‘any’, ‘all’}, default ‘any’, Determine if row or column is removed from DataFrame, when we have at least one NA or all NA. ‘any’ : If any NA values are present, drop that row or column. ‘all’ : If all values are NA, drop that row or column.
columns_with_na_dropped.head()
# Dataframe.head() Return the first n rows. It is useful for quickly testing if your object has the right type of data in it.
# Parameters: n : int, default 5. Number of rows to select.
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits.dropna()

# Now try removing all the columns with empty values. Now how much of your data is left?
dropped = sf_permits.dropna(axis = 1)
dropped.head()
print ("Columns in sf_permits %s \n" %sf_permits.shape[1])
print ("Colmuns in the dropped dataset %s \n" %dropped.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
sf_permits.head()
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then replacing any remaining NaN's with 0
filled_with_0=sf_permits.fillna(0)
filled_with_0.head()
filled_with_before=sf_permits.fillna(method = 'bfill', axis=0)
filled_with_before.head()