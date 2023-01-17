# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
inputData=pd.read_csv(r"../input/BreadBasket_DMS.csv")
mergedDateTime = inputData.Date +' '+inputData.Time
inputData.index = (pd.to_datetime(mergedDateTime))
inputData.drop(["Time","Date","Transaction"],axis=1,inplace=True)
startDate,endDate = "2016-10","2017-02"
itemName = "Coffee"
subset = inputData[startDate:endDate]
# Group the data on hourly spans
group = pd.DataFrame({"ItemCount":subset.groupby([subset.index.map(lambda t: t.hour),"Item"]).size()}).reset_index();

# Now lets find the probabilities
# Note this is an hourly probability - so we only consider items falling within
# the hour span
itemProbabilities = group[group["Item"] == itemName]
itemProbabilities.rename(index=int, columns={"level_0": "Hour"},inplace=True)
itemProbabilities.drop(["Item"],axis= 1,inplace=True)
total = np.float(itemProbabilities["ItemCount"].sum())  
itemProbabilities["ItemCount"]=itemProbabilities["ItemCount"].apply(lambda v:(v /total)) # ItemCount will now hold probabilities

itemProbabilities.head()
fig = plt.figure(figsize = (15,5))
ax = fig.gca()
x = itemProbabilities["Hour"]
y = itemProbabilities["ItemCount"]
plt.stem(x,y, markerfmt='go',label="Item Probabilities")
plt.legend()
plt.xlabel('Time span',fontsize=10)
plt.ylabel('Probability',fontsize=10)
ax.tick_params(labelsize=10)
plt.title('Probability of {} sold during the day'.format(itemName),fontsize=20)
plt.grid()
plt.ioff()
plt.show()

