# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/2017.csv")
data.info()
data.corr() #It gives data about ratio of column headings
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)

plt.show()
data.head(10)
data.columns
# Line Plot

# linewidth = width of line, alpha = opacity, linestyle = sytle of line

data.Family.plot(kind = "line",color = "g",label = "Family",linewidth = 1,alpha = 0.9,grid = True,linestyle = ":" )

data.Freedom.plot(color = "r",label = "Freedom",linewidth = 1, alpha = 0.9,grid = True,linestyle = '-.')

plt.legend(loc = "upper right")     # legend = puts label into plot

plt.xlabel("x axis")              # label = name of label

plt.ylabel("y axis")

plt.title("Line Plot")            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind = "scatter", x = "Family", y = "Freedom",alpha = 0.5,color = "red")

plt.xlabel("Family")              # label = name of label

plt.ylabel("Freedom")

plt.title("Family Freedom Scatter Plot")            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.Family.plot(kind = 'hist',bins = 40,figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.Family.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
#create dictionary and look its keys and values

dictionary = {"spain" : "madrid","italy" : "rome"}

print(dictionary.keys())

print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dictionary["spain"] = "barcelona"    # update existing entry

print(dictionary)

dictionary["france"] = "paris"       # Add new entry

print(dictionary)

del dictionary["spain"]              # remove entry with key 'spain'

print(dictionary)

print("france" in dictionary)        # check include or not

dictionary.clear()                   # remove all entries in dict

print(dictionary)
# In order to run all code you need to take comment this line

# del dictionary         # delete entire dictionary     

print(dictionary)       # it gives error because dictionary is deleted
data = pd.read_csv("../input/2017.csv")
series = data["Happiness.Score"]        # data['Happiness.Score'] = series

print(type(series))

data_frame = data[["Happiness.Score"]]  # data[['Happiness.Score']] = data frame

print(type(data_frame))
# Comparison operator

print(3 > 2)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
# 1 - Filtering Pandas data frame

x = data["Happiness.Score"] > 7.0    # There are 12 countries which have higher happiness score than 7.0

data[x]
# 2 - Filtering pandas with logical_and

# There are 9 countries which have higher happiness score than 7.0 and higher family than 1.5

data[np.logical_and(data["Happiness.Score"] > 7.0, data["Family"] > 1.5)]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data["Happiness.Score"] > 7.0) & (data["Family"] > 1.5)]
# Stay in loop if condition( i is not equal 5) is true

i = 0

while i != 5 :

    print('i is: ',i)

    i +=1 

print(i,' is equal to 5')
# Stay in loop if condition( i is not equal 5) is true

lis = [1,2,3,4,5]

for i in lis:

    print('i is: ',i)

print("")



# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(lis):

    print(index," : ",value)

print("")   



# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {"spain" : "madrid","france" : "paris"}

for key,value in dictionary.items():

    print(key," : ",value)

print("")



# For pandas we can achieve index and value

for index,value in data[["Happiness.Score"]][0:1].iterrows():

    print(index," : ",value)
