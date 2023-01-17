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
data = pd.read_csv("../input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")




data2 = data.tail(400)
f,ax = plt.subplots(figsize=(6, 6))

sns.heatmap(data2.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data2.Low.plot(kind = 'line', color = 'g',label = 'Low',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data2.Close.plot(color = 'r',label = 'Close',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = High, y = Volume_(BTC)

data2.plot(kind='scatter', x='High', y='Volume_(BTC)',alpha = 0.5,color = 'red')

plt.xlabel('High')              # label = name of label

plt.ylabel('Volume_(BTC)')

plt.title('BTC CORR')            # title = title of plot
# Histogram

# bins = number of bar in figure

data2.Close.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
# 1 - Filtering Pandas data frame

x = data['Volume_(BTC)']>800     # There are only 3 pokemons who have higher defense value than 200

data[x]
# 2 - Filtering pandas with logical_and

# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100

data[(data['Volume_(BTC)']>800) & (data['High']>3000)]
threshold = sum(data2["Volume_(BTC)"])/len(data2["Volume_(BTC)"])
print(threshold)
threshold = sum(data2.Volume_(BTC))/len(data2.Volume_(BTC))

data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]

data.loc[:10,["speed_level","Speed"]] # we will learn loc more detailed later