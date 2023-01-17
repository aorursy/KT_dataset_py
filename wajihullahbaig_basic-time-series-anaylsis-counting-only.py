# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
import numpy as np
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

inputData=pd.read_csv(r"../input/BreadBasket_DMS.csv",parse_dates=["Date"], index_col="Date")
inputData.dtypes
inputData.columns
print("Data shape:",inputData.shape)
print(inputData.head(10))
print(inputData.index)
#Lets check some items for 2016
startDate,endDate = "2016-10","2016-12"
selectedItems = inputData[startDate:endDate].loc[:,['Time','Item']]

itemName = "Jam"
#Lets get a particular item as a subset
subSet = selectedItems.loc[selectedItems["Item"] == itemName]
# Lets make an item count on time
counts = subSet.Time.value_counts()

# We have a series and we need to have a dataframe - Needed for plot
plotData = pd.DataFrame(counts,index = counts.index)
plotData.sort_index(inplace=True)
show = 10
take = plotData.head(show)
x = take.index
y = take.Time # This has now the count
fig = plt.figure(figsize = (15,5))
ax = fig.gca()
plt.stem(x,y)
plt.xlabel('Time span',fontsize=10)
plt.ylabel('Count',fontsize=10)
ax.tick_params(labelsize=10)
plt.title('Item sold during a day (1 year culmination) - only {} entries shown'.format(show),fontsize=20)
plt.grid()
plt.ioff()
plt.show()

groupeData = subSet.groupby(subSet.index).size()
monthlyAverages = groupeData.groupby(pd.TimeGrouper(freq="M")).count()
monthlyAverages.plot(kind='bar')
plt.xlabel('Date span',fontsize=17)
plt.ylabel('Count',fontsize=17)
plt.show()

