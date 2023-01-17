# Zillow’s Home Value Prediction (Zestimate)

# Python environment: information required to understand database

# I hope you enjoy this first part of the code
import numpy as np # linear algebra: work with large dataset 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# just to make sure you're collecting data ...
train_df = pd.read_csv("../input/properties_2016.csv")

print(train_df.describe()) # describing the data (without NaNs)

# conclusions

# more than 2.98 billion of houses

# A lot of value is missed  but few values are missed in  bathroomcnt,bedroomcnt,
