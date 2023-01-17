# import the pandas module

import pandas as pd
import matplotlib.pyplot as plt

# copy data from a file called heathrow-2015.csv and store it in pandas as a dataset called heathrow_2015_data

heathrow_2015_data = pd.read_csv("../input/ldsedexcel/heathrow-2015.csv")

# show the first 6 records from the data

heathrow_2015_data.head(6)
# check the datatypes

# use describe for any fields you are going to investigate and filter out or replace any unusable values (such as 'tr')
# find the means and standard deviations of the fields grouped by wind direction

# create box plots grouped by wind direction

# create a dataset for a different location or year and repeat the analysis
# communicate the result