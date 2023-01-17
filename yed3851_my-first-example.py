# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#read data

data = pd.read_csv('../input/WorldCups.csv')
#looking data information

data.info()
#proportional table between parameters

data.corr()
#correlation map

f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(data.corr(), annot=True, linewidths=.4, fmt= '.1f',ax=ax)

plt.show()
#how many data dou you want see?

data.head(5)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.GoalsScored.plot(kind = 'line', color = 'g',label = 'GoalsScored',linewidth=3,alpha = 0.5,grid = True,linestyle = ':')

data.MatchesPlayed.plot(color = 'r',label = 'MatchesPlayed',linewidth=3, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper left')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = GoalsScored, y = MatchesPlayed

data.plot(kind='scatter', x='GoalsScored', y='MatchesPlayed',alpha = 0.5,color = 'red')

plt.xlabel('GoalsScored')              # label = name of label

plt.ylabel('MatchesPlayed')

plt.title('GoalsScored MatchesPlayed Scatter Plot')            # title = title of plot
# Histogram

# bins = number of bar in figure

data.GoalsScored.plot(kind = 'hist',bins = 50,figsize = (12,12))

# clf() = cleans it up again you can start a fresh

data.GoalsScored.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
#pandas

series = data['GoalsScored']       

print(type(series))

data_frame = data[['GoalsScored']] 

print(type(data_frame))
# 1 - Filtering Pandas data frame

x = data['GoalsScored']>100     

data[x]
# 2 - Filtering pandas with logical_and

data[np.logical_and(data['GoalsScored']>120, data['MatchesPlayed']>40 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['GoalsScored']>100) & (data['MatchesPlayed']>50)]