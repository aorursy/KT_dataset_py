# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/DJIA_table.csv')
data.info()
data.head()

data.tail ()
print(type('Date')) # As you can see date is string

# however we want it to be datetime object

datetime_object = pd.to_datetime(data.Date)

print(datetime_object)
data.head()
data["new_date"] = datetime_object

# lets make date as index

data2 = data.set_index("new_date")

data2 
print(data2.loc["2008-08-08"])
databy_year = data2.resample("A").mean() # in order to see the mean of years
databy_year.head ()
data1 = databy_year.loc[:,["High","Low"]] # line plot of the mean of the years

data1.plot()

data1.plot(subplots = True)

plt.show()
databy_month = data2.resample("M").mean()
databy_month.head ()
data3 = databy_month.loc[:,["High","Low"]]  # line plot of the mean of the months

data3.plot()
