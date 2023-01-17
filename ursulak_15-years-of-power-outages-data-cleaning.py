# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import datetime

# set seed for reproducibility
np.random.seed(0)
# loading data
power_outages_data = pd.read_csv("../input/Grid_Disruption_00_14_standardized - Grid_Disruption_00_14_standardized.csv")

power_outages_data.sample(10)

# check data type of cloumn
power_outages_data["Time Event Began"].dtype
# check unique values to find out missmatched types
power_outages_data_copy = power_outages_data.copy()
power_outages_data_copy['Time Event Began'].unique()
power_outages_data_copy.count()
# apply some rules to get rid of the values Midnight, Evening and 12:00 noon // it could be discussed if Evening should convert to null or to an aproximate time for evening as I do here
power_outages_data_copy.loc[(power_outages_data_copy['Time Event Began'] == 'Midnight', 'Time Event Began')] = "12:00 a.m."
power_outages_data_copy.loc[(power_outages_data_copy['Time Event Began'] == 'Evening', 'Time Event Began')] = "06:00 p.m."
power_outages_data_copy.loc[(power_outages_data_copy['Time Event Began'] == '12:00 noon', 'Time Event Began')] = "12:00 p.m."
power_outages_data_copy['Time Event Began'].unique()
# Changing 'Time Event Began' values to Datetime and put them in a new column
power_outages_data_copy["Time Event Began Parsed"] = pd.to_datetime(power_outages_data_copy["Time Event Began"], infer_datetime_format=True, errors='coerce').dt.time
power_outages_data_copy.sample(10)
# Checking the data loss by parsing
power_outages_data_copy.count()
# drop the old column
power_outages_data_copy = power_outages_data_copy.drop("Time Event Began", 1)
power_outages_data_copy.sample(10)
# a similiar handling for the 'Time of Restoration'
power_outages_data_copy['Time of Restoration'].unique()
power_outages_data_copy["Time of Restoration Parsed"] = pd.to_datetime(power_outages_data_copy["Time of Restoration"], infer_datetime_format=True, errors='coerce').dt.time
power_outages_data_copy.sample(10)
power_outages_data_copy = power_outages_data_copy.drop("Time of Restoration", 1)
power_outages_data_copy.sample(10)
# rename the columns
power_outages_data_copy = power_outages_data_copy.rename(columns={'Time of Restoration Parsed': 'Time of Restoration', 'Time Event Began Parsed': 'Time Event Began'})
power_outages_data_copy.sample(10)
power_outages_data_copy.to_csv("power_outages_data_copy.csv")