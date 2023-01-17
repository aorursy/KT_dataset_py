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

# your code goes here :)
sf_permits.sample(5)
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permits dataset is missing
sf_permits_missing_values= sf_permits.isnull().sum()
sf_permits_missing_values
sf_permits_total_values= np.product(sf_permits.shape)
#sf_permits_total_values
total_missing_sf_permits=sf_permits_missing_values.sum()
#total_missing_sf_permits
perecentage_of_missing_sf_permits=(total_missing_sf_permits/sf_permits_total_values)*100
#perecentage_of_missing_sf_permits

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
sf_permits_missing_values['Zipcode']

sf_permits_missing_values['Street Number Suffix']
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
#The shape attribute for numpy arrays returns the dimensions of the array.
#If nfl_data has n rows and m columns, then nfl_data.shape is (n,m). So nfl_data.shape[0] is n and nfl_data.shape[1] is m for a 2D array
nfl_data.shape
#rows
nfl_data.shape[0]
#columns
nfl_data.shape[1]
#This is out of range and returns an error
#nfl_data.shape[2]

# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
dropped_na_sf_permits=sf_permits.dropna(axis=1)
#The number of columns with na that were dropped
dropped_na_sf_permits
# just how much data did we lose?
print("Columns in original SF PERMITS dataset: %d \n" %sf_permits.shape[1])
print("Columns with na's dropped in SF PERMITS : %d" %dropped_na_sf_permits.shape[1])
# Now try removing all the columns with empty values. Now how much of your data is left?

# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
sf_permits.fillna(method = 'bfill', axis=0).fillna(0)
# comes directly after it and then replacing any remaining NaN's with 0