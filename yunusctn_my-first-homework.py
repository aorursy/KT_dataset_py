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
import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns #visulation
data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
data
data.info()

data.describe()
data.corr()

data.columns
#corelation map

f, ax =plt.subplots(figsize=(20,20))

sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt= '.1f', ax=ax) #burada birbiri ile ilişkisi olan birimler. 1 pozitif ilişki 0 ilişki yok.
data.Deaths.plot(kind='line', color='b', label='Deaths', Linewidth=1, alpha=0.5, grid=True, Linestyle =':')

#data.Confirmed.plot(color='green', label='Confirmed', Linewidth=1, alpha=0.5, grid=True, Linestyle ='--')

data.Recovered.plot(color='green', label='Recovered', Linewidth=1, alpha=0.5, grid=True, Linestyle ='--')

plt.xlabel('x noktası (Ölen)')

plt.ylabel('y noktası (iyileşen)')

plt.legend()

plt.show()
#scatter plot

data.plot(kind='scatter', x='Deaths', y='Recovered', alpha=0.5, color='red')

plt.xlabel('Ölen')

plt.ylabel('iyileşen')

plt.title('Died and Recovered Scatter Plot')

#plt.scatter(data.Deaths, data.Recovered, color='yellow', alpha=0.5)
#histogram

data.Deaths.plot(kind='hist', bins=10, figsize=(10,3)) #histogram is not appropriate for this.
num2 = ['ölü var'if i > 0 else 'ölü yok' for i in data.Deaths]

print(num2)

data1=data.head()

data2=data.tail()

data_conc=pd.concat([data1, data2], axis=0, ignore_index=True)

data_conc
data1=data.head()

data2=data.tail()

data_conc=pd.concat([data1, data2], axis=1,)

data_conc
data.dtypes
data.info()  #missing data
data["Country/Region"].value_counts(dropna =True)

data["Deaths"].value_counts(dropna=False)