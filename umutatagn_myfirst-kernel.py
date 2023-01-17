# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data= pd.read_csv('/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
data.head(10)
#Correlation map
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
plt.show()
data.corr()
data.columns
data.High.plot(kind='line',color='g',label='high',linewidth=1,alpha=0.5,grid=True,linestyle=':')
data.Low.plot(color='r',label='Low',linewidth=1,alpha=0.5,linestyle='-.',grid=True)
plt.legend('upper right')
plt.title('Bitcoin')
plt.show()
#Scanner plot x=High y=Low
data.plot(kind='scatter',x='High',y='Low',alpha=0.5,color='b')
plt.xlabel=('High')
plt.ylabel=('Low')
plt.title=('High-Low Scatter plot')
plt.show()
data.Low.plot(kind='hist',bins=50,figsize=(7,7))
plt.show()
x=data['Low']>19640
data[x]
data[np.logical_and(data['Low']>19640,data['High']>=19666)]
for index,value in data[['Low'][0:1]].iterrows():
    print(index,':',value)
