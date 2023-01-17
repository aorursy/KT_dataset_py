import numpy as np

import pandas as pd



# Reading the data set

data = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")
# look at a few rows of the data file. I can see a handful of missing data already!

data.head()
# get the number of missing data points per column

missing_value_count = data.isnull().sum()



# look at the # of missing points in the first ten columns

missing_value_count[0:10]
# how many total missing values do we have?

total_cells = np.product(data.shape)

total_missing = missing_value_count.sum()



(total_missing/total_cells)*100
# look at the # of missing points in the first ten columns

missing_value_count
# remove all the rows that contain a missing value

data.dropna()
# remove all columns with at least one missing value

column_with_na_dropped = data.dropna(axis=1)

column_with_na_dropped.head()
# just how much data did we lose?

print('Columns in original dataset: %d \n' % data.shape[1])

print("Columns with na's dropped: %d \n" % column_with_na_dropped.shape[1])
# look at a few rows of the data file.

data.head()
# replace all NA's with 0

data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 

# then replace all the reamining na's with 0

data.fillna(method='bfill', axis=0).fillna(0)