# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib.pyplot as plt
import seaborn as sns #visualization tool
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
data.head()

data.columns
data.info()
data.corr()
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),annot=True,linewidth=.5,fmt='.2f',ax=ax)
data.Open.plot(kind='line',color='g',label='Speed',linewidth=1,alpha=0.5,grid=True,linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot of Open Attribute')
plt.show()

axx = data['Open'].plot.hist(figsize=(12,6),fontsize=14,bins=50,color='gray')
data.plot.scatter(x='High',y='Low',figsize=(12,6),alpha=0.5,title='Bitcoin Values')
print(data['High'][0:4000000])

k = data['Open'] > 0
data[k]
data[np.logical_and(data['High']>4000,data['Low']<4000)]
data[(data['High']>2000) & (data['Low']<2000)]
for index,value in enumerate(data['Open'][0:20]):
    print(index," : ",value)
for index,value in data[['Open']][0:5].iterrows():
    print(index," : ",value)