# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #data visualization libraray
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt#for visualization graph 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataframe = pd.read_csv("../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv")
#Read the data frame
dataframe.head() 
#This head function is used for the demonstration of 5 top data
dataframe.head(10)#10 top data is coming from the dataset file
dataframe.info() #The colon and their type information will be coming
dataframe.corr()
#correlation map 
f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(dataframe.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
plt.show()

dataframe.head(10)
dataframe.columns

dataframe.Low.plot(kind = 'line', color = 'black',label = 'Low',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
dataframe.Close.plot(color="purple",label="Close",linewidth=1,grid=True,linestyle="-.")
plt.legend(loc="upper right")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
#plt.zlabel("Z-Axis")
plt.title("Bitcoin Line Plot")
plt.show()
dataframe.plot(kind="scatter",x="High",y="Close",alpha=0.7,color="green")
plt.xlabel("High")
plt.ylabel("Close")
plt.title("Bitcoin High and Close Fluctiation Plot")

dataframe.Low.plot(kind="hist",bins=60,figsize=(20,20))
plt.show()
dataframe.High.plot(kind="hist",bins=60,figsize=(20,20))
plt.show()
dataframe.Close.plot(kind="hist",bins=40)
plt.clf()
data =pd.read_csv("../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv")
series = data['Low']
print(type(series))
data_frame = data[["Close"]]
print(type(data_frame))
x = data['Close'] >5
data[x]
data[np.logical_and(data["Close"]>20,data["High"]>130)]
data[(data["Low"]>1200) & (data["High"]>130)]