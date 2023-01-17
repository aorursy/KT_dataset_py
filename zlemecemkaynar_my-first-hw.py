# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/world-happiness/2019.csv')

data.head(25)

#first 25 data- 25 happiest country
data.info()

data.columns=['Overall_rank','Country_or_region','Score','GPD_per_capita','Social_support','Healt_life_expectancy','Freedom','Generosity','Corruption']
#correlation map

f,ax = plt.subplots(figsize=(11, 11))

sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt= '.1f',ax=ax)

plt.show()
# Line Plot

data.Generosity.plot(kind = 'line', color = 'b',label = 'Generosity',linewidth=2,alpha = 1,grid = True,linestyle = ':')

data.Healt_life_expectancy.plot(color = 'black',label = 'Healt Life Exp',linewidth=1, alpha = 1,grid = True,linestyle = '-')

plt.legend(loc='upper right')    

plt.xlabel('Generosity')              

plt.ylabel('Healt Life Expectancy')

plt.title('Generosity and Healt Life Expectancy values') 

plt.show()
# Scatter Plot 

# x = GPD_per_capita, y = Social_support

data.plot(kind='scatter', x='GPD_per_capita', y='Social_support',alpha = 0.8,color = 'blue')

plt.xlabel('GPD per capita')              # label = name of label

plt.ylabel('Social support')

plt.title('GPD per capita vs Social support Scatter Plot') 

plt.show()
data.Score.plot(kind = 'hist',bins = 50,figsize = (12,8))

plt.show()
# cleans it up again you can start a fresh

#data.Score.plot(kind = 'hist',bins = 50)

#plt.clf()
#A new dictionary- Country or region vs score values

dictionary={'Finland':'7,769','Denmark':'7.600','Norway':'7.554','Iceland':'7.494','Netherlands':'7.488','Switzerland':'7.480'}

print(dictionary.keys())

print(dictionary.values())

#add new item on dictionary

dictionary['Sweden']='7,343'

print(dictionary)
#delete item on dictionary

del dictionary['Denmark']

print(dictionary)
print('Iceland' in dictionary)

print('Turkey' in dictionary)
series = data['Score']        # data['Defense'] = series

print(type(series))

data_frame = data[['Score']]  # data[['Defense']] = data frame

print(type(data_frame))
# 1 - Filtering Pandas data frame

x = data['Score']>7.5     # There are 3 countries which have higher score value than 7

data[x]


y=data['Score']<3.3 # There are 4 countries which have lower score value than 4

data[y]
#There are 7 countries have higher freedom value than 0,3 and have lower score value than 4

data[(data['Freedom']>0.3) & (data['Score']<4)]

#There are only 2 countries have higher freedom value than 0,3 and have lower score value than 4

data[(data['Freedom']<0.3) & (data['Score']>6)]
#first four value and index of items

for index,value in data[['Score']][0:3].iterrows():

    print(index," : ",value)