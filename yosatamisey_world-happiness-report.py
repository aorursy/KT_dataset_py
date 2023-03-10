# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/2017.csv')
data.info()
data.head(10)
data.tail(10)
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.columns
data.Family.plot(kind='line', color = 'r',label = 'Family', linewidth = 1, alpha = 0.5, grid = True, linestyle = ':')
data.Freedom.plot(kind = 'line', color = 'g', label = 'Freedom', linewidth = 1, alpha = 0.5, grid = True, linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='Family', y='Freedom',alpha = 0.5,color = 'red')
plt.xlabel('Family')              # label = name of label
plt.ylabel('Freedom')
plt.title('Family - Freedom Scatter Plot')            # title = title of plot
# Histogram
# bins = number of bar in figure
data.Freedom.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
# Histogram
# bins = number of bar in figure
data.Family.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
