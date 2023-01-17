# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
d = pd.read_csv("../input/electric-motor-temperature/pmsm_temperature_data.csv")

d.info()
d.corr()
#correlation map

f,ax = plt.subplots(figsize=(20, 10))

sns.heatmap(d.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
d.head()
d.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

d.motor_speed.plot(kind = 'line', color = 'b',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

d.torque.plot(color = 'r',label = 'Torque',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = Speed, y = Torque

d.plot(kind='scatter', x='motor_speed', y='torque',alpha = 0.8,color = 'red')

plt.xlabel('Speed')              # label = name of label

plt.ylabel('Torque')

plt.title('Scatter Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

d.motor_speed.plot(kind = 'hist',bins = 50,figsize = (10,10))

plt.title('Histogram Plot')

plt.show()
d[np.logical_and(d['motor_speed']>1.5, d['torque']>0.720 )]