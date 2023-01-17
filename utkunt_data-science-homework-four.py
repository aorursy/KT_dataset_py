# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
mydata = pd.read_csv('../input/fifa19/data.csv')

mydata.info() # basic information
 # building dataframe from dictionaries

team = ['fenerbahce', 'besiktas', 'galatasaray']

player = ['24','21', '22']

list_label = ['team', 'player']

list_col = [team,player]

zipped_listed = list(zip(list_label,list_col))

dict_file = dict(zipped_listed)

data_frame01 = pd.DataFrame(dict_file)

print(data_frame01)
# add new column and assing value for each one

data_frame01['founding_date'] = ['1907', '1903', '1905']

print(data_frame01)
# add new columnd and assign different values

data_frame01['punishment'] = 0

print(data_frame01)
# plotting data

mydata01 = mydata.loc[:,['Dribbling','BallControl']]

mydata01.plot(kind='scatter',x='Dribbling',y='BallControl', alpha=0.3)
# convert string into time

time_list = ['1986-08-09','1986-01-31']

datetime_object = pd.to_datetime(time_list) # convert str to time series

print(type(time_list[0])) # type of time_list

print(type(datetime_object)) # type of datetime_object

newdata = mydata.head()

timelist = ['2019-01-01','2019-01-02','2018-01-31','2014-09-13','2010-09-09']

datetime_object = pd.to_datetime(timelist)

newdata['Date'] = datetime_object

newdata = newdata.set_index("Date")

newdata
print(newdata.loc['2018-01-31'])
# resample data by year

newdata.resample('A').mean()
# resample data by month

newdata.resample('M').mean()
newdata.resample('M').interpolate('linear')