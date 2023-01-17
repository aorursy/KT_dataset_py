# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
donations = pd.read_csv("../input/Donations.csv")

# Getting donor ID's from the dataset 'Donations'
donors = donations["Donor ID"].value_counts()

# Getting donor ID's list who donated number of times
donor_ids = []
for donor_index in range(len(donors)):
    if donors[donor_index] > 1:
        donor_ids.append(donors.index[donor_index])

# Importing datetime module required for time conversions(date to nanoseconds)
from datetime import datetime

# Getting the day interval list of a particular donor ID
donor_donations_day_interval = []
for donor_index in range(len(donor_ids)):
    temp = donations[donations["Donor ID"] == donor_ids[donor_index]]
    # Getting all time objectives of donations of a donor
    time_objs = temp["Donation Received Date"].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    
    sorted_datetime = time_objs.sort_values()
    #print(sorted_datetime)
    
    # Get a list of day intervals of multiple donations of a donor
    donations_day_interval = [donor_ids[donor_index]]
    for z in range(len(sorted_datetime)):
        if z < len(sorted_datetime)-1:
            td = np.timedelta64(sorted_datetime.values[z+1] - sorted_datetime.values[z], 'ns')
            days = td.astype('timedelta64[D]')
            donations_day_interval.append(days / np.timedelta64(1, 'D'))
    donor_donations_day_interval.append(donations_day_interval)
    
print(donor_donations_day_interval)

