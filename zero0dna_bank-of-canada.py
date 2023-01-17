# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from datetime import datetime

from datetime import timedelta

import random



# record the start and end date of the two datasets

start_ds1 = datetime.strptime('1960-01-01', '%Y-%m-%d')

end_ds1 = datetime.strptime('1995-12-31', '%Y-%m-%d')

start_ds2 = datetime.strptime('1980-01-01', '%Y-%m-%d')

end_ds2 = datetime.strptime('2018-12-31', '%Y-%m-%d')



num_days = (end_ds2 - start_ds1).days + 1

###############################################################

date_list = [(start_ds1 + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(num_days)]

date_list_ds1 = [(start_ds1 + timedelta(days=x)).strftime('%Y-%m-%d') for x in range((end_ds1-start_ds1).days+1)]

date_list_ds2 = [(start_ds2 + timedelta(days=x)).strftime('%Y-%m-%d') for x in range((end_ds2-start_ds2).days+1)]



ds1 = {k: round(random.uniform(0, 0.2), 3) for k in date_list_ds1}

ds2 = {k: round(random.uniform(0, 0.2), 3) for k in date_list_ds2}

###############################################################



# create a new empty dataset to store the results

ds3 = {}



# set the current date to 1960-01-01

cur_date = datetime.strptime('1960-01-01', '%Y-%m-%d')



# loop through 1960-01-01 to 2018-12-31

for i in range(num_days):

    # convert date to string

    cur_date_str = cur_date.strftime('%Y-%m-%d')



    # save yields data to ds3 when they are only in ds1

    if cur_date < start_ds2 and cur_date_str in ds1:

        ds3[cur_date_str] = ds1[cur_date_str]

        

    # averaging the yields data if they are in both ds1 and ds2

    elif cur_date <= end_ds1 and cur_date_str in ds1 and cur_date_str in ds2:

        ds3[cur_date_str] = ds1[cur_date_str] if ds1[cur_date_str] == ds2[cur_date_str] else (ds1[cur_date_str] + ds2[cur_date_str])/2

    

    # save yields data to ds3 when they are only in ds2

    elif cur_date <= end_ds2 and cur_date_str in ds2:

        ds3[cur_date_str] = ds2[cur_date_str]

        

    cur_date += timedelta(days=1)



###############################################################

for d in date_list:

    y1 = str(round(ds1[d] * 100, 3)) + '%' if d in ds1 else '---'

    y2 = str(round(ds2[d] * 100, 3)) + '%' if d in ds2 else '---'

    y3 = str(round(ds3[d] * 100, 3)) + '%'

    

    print(d,' ', 

          y1 + '  ',

          y2 + '  ', 

          y3)
ds4 = {'1960-01-01': 0.134,

       '1960-01-02': 0.137,

       '1985-01-31': 0.134,

       '1985-02-01': None,

       '1995-12-30': 0.105,

       '1995-12-31': None}

# create a variable to store the previous data

prev_value = 0



# loop through the dataset

for k in ds4.keys():

    # replace data with previous if it's missing

    if ds4[k] == None:

        ds4[k] = prev_value

    # set the current data as previous

    prev_value = ds4[k]

    ################################

    print(k, ' ', ds4[k])
import math



# create a new empty dataset to store the results

ds5 = {}



# create a variable to record the sum of yields

total = 0



# loop through the dataset

prev_value = ('1960-01-01', ds3['1960-01-01'])

for k, val in ds3.items():

    # from the second yield data, calculate the difference and save it to d5

    if k == '1960-01-01':

        continue

    change = val - prev_value[1]

    ds5[prev_value[0] + '~' + k] = change 

    prev_value = (k, val)

    # add the change to the sum variable

    total = total + change



# calculate mean from sum

mean = total / len(ds5)



# create a variable to record the variance of yields

variance = 0



# loop through the yields change dataset to calculate the variance

for val in ds5.values():

    variance = variance + (val - mean) ** 2

variance = variance / len(ds5)



# calculate standard deviation from variance

std = math.sqrt(variance)



# calculate high and low end of 3 sigma

high = mean + std*3

low = mean - std*3



# create a variable to store the results

ds6 = {}



# loop through the yields change dataset to filter out data that are out of 3 sigma range

# and store the results in ds6

for k, val in ds5.items():

    if val < high and val > low:

        ds6[k] = val
# create a new list to store the monthly changes

monthly_change = []



# record the current month number

cur_month = 1



# record average value of previous month

prev_avg = 0



# record the number and sum of data for current month 

cur_num = 0

cur_total= 0



# loop through the yields dataset

for k, val in ds3.items():

    # get the month of current data point

    month = datetime.strptime(k, '%Y-%m-%d').month

    

    # if we are in a new month then:

    if cur_month != month:

        # calculate the average of current month

        cur_avg = cur_total / cur_num

        # calculate the change from previous month to current month and save it to the list

        monthly_change.append(cur_avg - prev_avg)

        # set previous month average to current month average

        prev_avg = cur_avg

        # update the record of month number

        cur_month = 1 if cur_month == 12 else cur_month + 1 

        # reset number and sum of data for current month

        cur_total = 0

        cur_num = 0

    # add value of yield to sum     

    cur_total = cur_total + val

    # increment the data count 

    cur_num = cur_num + 1

    

print(len(monthly_change), len(ds3))

    

    

        
from datetime import datetime

from datetime import timedelta

import random



# record the start and end date of the two datasets

start_ds1 = datetime.strptime('1960-01-01', '%Y-%m-%d')

end_ds1 = datetime.strptime('1995-12-31', '%Y-%m-%d')

start_ds2 = datetime.strptime('1980-01-01', '%Y-%m-%d')

end_ds2 = datetime.strptime('2018-12-31', '%Y-%m-%d')



# total number of days from 1960-01-01 to 2018-12-31

num_days = (end_ds2 - start_ds1).days + 1



# create a new empty dataset to store the results

ds3 = {}



# set the current date to 1960-01-01

cur_date = datetime.strptime('1960-01-01', '%Y-%m-%d')



# loop through 1960-01-01 to 2018-12-31

for i in range(num_days):

    # convert date to string

    cur_date_str = cur_date.strftime('%Y-%m-%d')



    # save yields data to ds3 when they are only in ds1

    if cur_date < start_ds2 and cur_date_str in ds1:

        ds3[cur_date_str] = ds1[cur_date_str]

        

    # averaging the yields data if they are in both ds1 and ds2

    elif cur_date <= end_ds1 and cur_date_str in ds1 and cur_date_str in ds2:

        ds3[cur_date_str] = ds1[cur_date_str] if ds1[cur_date_str] == ds2[cur_date_str] else (ds1[cur_date_str] + ds2[cur_date_str])/2

    

    # save yields data to ds3 when they are only in ds2

    elif cur_date <= end_ds2 and cur_date_str in ds2:

        ds3[cur_date_str] = ds2[cur_date_str]

        

    cur_date += timedelta(days=1)
# create a variable to store the previous data

prev_value = 0



# loop through the dataset

for k in ds3.keys():

    # replace data with previous if it's missing

    if ds3[k] == None:

        ds3[k] = prev_value

    # set the current data as previous

    prev_value = ds3[k]
import math



# create a new empty dataset to store the results

ds4 = {}



# create a variable to record the sum of yields

total = 0



# loop through the dataset

prev_value = ('1960-01-01', ds3['1960-01-01'])

for k, val in ds3.items():

    # from the second yield data, calculate the difference and save it to ds4

    if k == '1960-01-01':

        continue

    change = val - prev_value[1]

    ds4[prev_value[0] + '~' + k] = change 

    prev_value = (k, val)

    # add the change to the sum variable

    total = total + change



# calculate mean from sum

mean = total / len(ds4)



# create a variable to record the variance of yields

variance = 0



# loop through the yields change dataset to calculate the variance

for val in ds4.values():

    variance = variance + (val - mean) ** 2

variance = variance / len(ds4)



# calculate standard deviation from variance

std = math.sqrt(variance)



# calculate high and low end of 3 sigma

high = mean + std*3

low = mean - std*3



# create a variable to store the results

ds5 = {}



# loop through the yields change dataset to filter out data that are out of 3 sigma range

# and store the results in ds6

for k, val in ds4.items():

    if val < high and val > low:

        ds5[k] = val
# create a new list to store the monthly changes

monthly_change = []



# record the current month number

cur_month = 1



# record average value of previous month

prev_avg = 0



# record the number and sum of data for current month 

cur_num = 0

cur_total= 0



# loop through the yields dataset

for k, val in ds3.items():

    # get the month of current data point

    month = datetime.strptime(k, '%Y-%m-%d').month

    

    # if we are in a new month then:

    if cur_month != month:

        # calculate the average of current month

        cur_avg = cur_total / cur_num

        # calculate the change from previous month to current month and save it to the list

        monthly_change.append(cur_avg - prev_avg)

        # set previous month average to current month average

        prev_avg = cur_avg

        # update the record of month number

        cur_month = 1 if cur_month == 12 else cur_month + 1 

        # reset number and sum of data for current month

        cur_total = 0

        cur_num = 0

    # add value of yield to sum     

    cur_total = cur_total + val

    # increment the data count 

    cur_num = cur_num + 1