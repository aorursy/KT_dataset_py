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
data = pd.read_csv('../input/master.csv')

data.info()
data.corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

data.columns
data.year.plot(kind = 'line', color = 'g',label = 'year',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.suicides_no.plot( color = 'r',label = 'suicides_no',linewidth=1,alpha = 0.5,grid = True,linestyle = '--')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()

data.plot(kind='scatter', x='suicides_no', y='year',alpha = 0.5,color = 'red')
plt.xlabel('suicides_no')              # label = name of label
plt.ylabel('year')
plt.title('Sex Age Scatter Plot') 
data.suicides_no.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()