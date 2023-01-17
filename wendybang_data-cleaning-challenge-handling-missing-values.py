# modules we'll use
import pandas as pd   # both pandas and numpy are very useful libraries in Python (this is my first time using Python)
import numpy as np

# read in all our data
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 
# look at a few rows of the nfl_data file. I can see a handful of missing data already!
nfl_data.sample(5) # sample as in R
# your turn! Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?

sf_permits.sample(5)
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()   # .isnull() returns a boolean value 

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape) # .shape returns the dimension of the data # .product just multiply the number of rows and the number of columns
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permits dataset is missing
sf_missing_counts = sf_permits.isnull().sum()   # number of missing values in each column of sf_permits 
sf_missing_total = sf_missing_counts.sum()      # sum them up to get the total number of missing values in sf_permits
data_total = np.product(sf_permits.shape)       # total number of the whole data
sf_missing_total/data_total*100                 # percentage of missing data
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1]) # data.shape[0] is the number of rows and ...[1] is the number of columns
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1]) # %d will format a number for display.
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits.dropna()
# Now try removing all the columns with empty values. Now how much of your data is left?
sf_dropna_columns = sf_permits.dropna(axis=1)
print("Columns in the original data: %d \n" % sf_permits.shape[1])  # number of colunms in sf_permits
print("Columns after dropping na columns: %d" % sf_dropna_columns.shape[1])  # number of columns after dropping columns containing missing values
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head() # data.loc[]: Purely label-location based indexer for selection by label. [:,] any row, [ ,'a':'b'] columns a through b 
subset_nfl_data
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0) # `ffill` does the opposite direction # axis = 0 requires column by column fillna
# if `axis=1` with method=`backfill` then it will use its posterior value (value in its later column) to replace na
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then replacing any remaining NaN's with 0
sf_permits.fillna(method='bfill',axis=0).fillna(0)


# Look back at the Zipcode column in the sf_permits dataset
number_of_missing_zip = sf_permits['Zipcode'].isnull().sum() # There are 1716 missing values in Zipcode column.
number_of_rows = sf_permits.shape[0]  # there are 198900 rows 
number_of_missing_zip/number_of_rows*100 # less than 1% missing values in Zipcode column, so probably we could fill the missing values in a reasonable way
# Take a look at columns in the data -- try to find the most related columns to Zipcode column -- from which we could impute the missing valueds
sf_permits.columns # now columns next to Zipcode should be considered next
sf_permits.loc[1:20, 'Supervisor District':'Location'] # First thought would be 'Location' column
loc1 = sf_permits.loc[2,'Location']  
loc1 # '(37.7946573324287, -122.42232562979227)' the type of loc1 is str -- couldn't be used for calculation -- needs to be converted to array
type(loc1)
loc2 = sf_permits.loc[3,'Location']

d1 = np.linalg.norm(np.array(eval((loc1)))-np.array(eval(loc2)))  # euclidean distance between loc1 and loc2 which share the same Zipcode
d1
loc3 = sf_permits.loc[1,'Location']
np.linalg.norm(np.array(eval((loc1)))-np.array(eval(loc3)))  # loc1 and loc3 has different Zipcode, so their distance is more likely to be greater than that 
                                                             # between loc1 and loc2. Let's check more.
sf_permits['Location'].isnull().sum() # there are 1700 missing values in column 'Location' -- recall that there are 1716 missing values in column 'Zipcode'
sf_permits_new = sf_permits[pd.notnull(sf_permits['Location'])]

sf_permits_new.shape #(197200, 43)
sf_permits.shape #(198900, 43)

sf_permits_new['Zipcode'].isnull().sum() # there are only 16 missing values in 'Zipcode' column in the new dataset
                                         # there are 1716 missing values in 'Zipcode' the original dataset
                                         # Unfortunetely, for most of missing values, we couldn't impute them by 'Location' since they both missing
# But let's first impute 16 zipcodes with available locations -- give them the zipcode of their nearest locations
def impute(df, location_to_compare, column, column_to_impute, threshold=0.007):
          b = df[location_to_compare,column]
          for i in range(1,sf_permit_new.shape[0]) and i!=location_to_compare:
                    a = df[i,column]
                    if np.linalg.norm(np.array(eval(a))-np.array(eval(b))) <= threshold:
                           df[location_to_compare,column_to_impute] = df[i,column_to_impute]
                           break;

sf_permits_new[sf_permits_new['Zipcode'].isnull()]