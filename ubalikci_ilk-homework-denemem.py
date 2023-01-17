# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/deneme1.csv")

data.corr()
data.info()
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True, linewidths=.9, fmt='.5f',ax=ax)
plt.show()
data.columns
data.Yuksek.plot(kind='line', color='g',label = 'Yüksek',linewidth=1,alpha = 0.9,grid = True,linestyle = ':')
data.Dusuk.plot(kind='line', color='r',label = 'Düşük',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Zaman')              # label = name of label
plt.ylabel('Değer')
plt.title('Yüksek Düşük Grafiği')            # title = title of plot
plt.show()
data.plot(kind='scatter', x='Yuksek', y='Dusuk',alpha = 0.5,color = 'red')
plt.xlabel('Yüksek')              # label = name of label
plt.ylabel('Düşük')
plt.title('Yüksek Düşük Scatter Plot ile')            # title = title of plot
plt.show()
data.Yuksek.plot(kind='hist',bins=50,figsize=(18,18))
plt.show()
data.Dusuk.plot(kind='hist',bins=100,figsize=(18,18),color='r')
plt.show()

seriler=data['Dusuk']
print(type(seriler))
datafreym=data[['Dusuk']]
print(type(datafreym))


x=data['Dusuk']>5
data[x]
liste=[10,72,44,15,32]
for i in liste:
    print('i=',i)
print(' ')
dictionary = {'spain':'madrid','france':'paris'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')
for index,value in data[['Dusuk']][0:10].iterrows():
    print(index," : ",value)