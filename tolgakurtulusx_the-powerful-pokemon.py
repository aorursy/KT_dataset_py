# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/pokemon.csv') #we read data
data.info() #we information data
data.corr() #we see corr. This just like multiplication level
#We see graphic data

f,ax = plt.subplots(figsize=(18, 18)) #18 18 write places. It is shape box settings

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax) #annot = True is writing in the box like 0.2

plt.show()
data.head() #This is the firt five data showing
data.columns
# LINE PLOT 

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

#kind = Line, Histogram, Bar ....

data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # x label = name of label

plt.ylabel('y axis')              # y label = name of label

plt.title('Line Plot')            # title = title of plot

plt.show()
# SCATTER PLOT 

# x = attack, y = defense

data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'blue')

plt.xlabel('Attack')              # label = name of label

plt.ylabel('Defence')

plt.title('Attack Defense Scatter Plot')            # title = title of plot

plt.show()
# HISTOGRAM

# bins = number of bar in figure

data.Speed.plot(kind = 'hist',bins = 75,figsize = (12,12))

plt.show()
# 1 - Filtering Pandas data frame

x = data['Defense']>200     # There are only 3 pokemons who have higher defense value than 200

data[x]
# 2 - Filtering Pandas data frame

x = data['Attack']>180     # There are only 2 pokemons who have higher attack value than 180

data[x]
# 3 - Filtering Pandas data frame

x = data['HP']>180     # There are only 3 pokemons who have higher hp value than 180

data[x]
# 4 - Filtering Pandas data frame

x = data['Speed']>150     # There are only 2 pokemons who have higher Speed value than 150

data[x]
# 5 - Filtering Pandas data frame

x = data['Legendary'] == True     # There are a lot of pokemons who have Legendary is True

data[x]
# 6 - Filtering Pandas My best pokemon kind

x = data['Type 1'] == 'Ghost' # There are a lot of pokemons who have Legendary is True

data[x]
# 7 - Filtering Pandas other My best pokemon kind

x = data['Type 1'] == 'Psychic' # There are a lot of pokemons who have Legendary is True

data[x]
# 8 - Filtering Pandas I wonder dark poke

x = data['Type 1'] == 'Dark' # There are a lot of pokemons who have Legendary is True

data[x]
# 9 - 1. Way

# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100

data[np.logical_and(data['Defense']>200, data['Attack']>100 )]
# 10 - 2. Way - I want to find the powerful pokemon and The Best Pokemon is Arceus

# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['Defense']>100) & (data['Attack']>100) & (data['HP']>100) & (data['Speed']>100)]