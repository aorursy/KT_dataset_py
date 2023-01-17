# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

import sqlite3 as sql

import numpy as np



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
db = sql.connect('../input/soccer/database.sqlite')

data_full = pd.read_sql_query("select * from Player_Attributes", db)

data_full.columns
db = sql.connect('../input/soccer/database.sqlite')

data = pd.read_sql_query("select * from Player_Attributes limit 5000", db)

data.columns
data.info()
data.head()
data.tail(10)
#correlation map

f,ax = plt.subplots(figsize=(25, 25))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()


# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

f,ax = plt.subplots(figsize=(18, 6))

data.loc[:,"shot_power"].plot(kind = 'line', color = 'g',label = 'shot_power',linewidth=1,alpha = 0.5,grid = True)

data.loc[:,"long_shots"].plot(color = 'r',label = 'long_shots',linewidth=1, alpha = 0.5,grid = True,linestyle = ':')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('number of samples')              # label = name of label

plt.ylabel('scale')

plt.title('Line Plot')            # title = title of plot

plt.show()
#Filtering pandas with logical_and

rightmedium=data_full[(data_full.preferred_foot=="right")&(data_full.attacking_work_rate=="medium")]

righthigh=data_full[(data_full.preferred_foot=="right")&(data_full.attacking_work_rate=="high")]

leftmedium=data_full[(data_full.preferred_foot=="left")&(data_full.attacking_work_rate=="medium")]

lefthigh=data_full[(data_full.preferred_foot=="left")&(data_full.attacking_work_rate=="high")]

print(righthigh.size)

print(rightmedium.size)

print(leftmedium.size)

print(lefthigh.size)
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

f,ax = plt.subplots(figsize=(18, 8))

#plt.scatter(rightmedium.shot_power,rightmedium.long_shots,label = "right attacking medium",alpha = 0.50)

plt.scatter(righthigh.shot_power[0:5000],righthigh.long_shots[0:5000],label = "right attacking high",alpha = 0.50)

#plt.scatter(leftmedium.shot_power,leftmedium.long_shots,label = "left attacking medium",alpha = 0.50)

plt.scatter(lefthigh.shot_power[0:5000],lefthigh.long_shots[0:5000],label = "left attacking high",alpha = 0.50)



plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('Shot Power')              # label = name of label

plt.ylabel('Long shots')

plt.title('Scatter Plot')            # title = title of plot

plt.show()
plt.hist(rightmedium.potential,bins=30)

plt.xlabel("medium potential")

plt.ylabel("frequency")

plt.title("hist")

plt.show()
dictionary=data.to_dict('series')

print(dictionary.keys())
dictionary["passing_accuracy"]=40.0,50.8,55.5 # Add new entry

print(dictionary.keys())

print(dictionary["passing_accuracy"][2])
print("curve" in dictionary) # check include in or not

del dictionary["curve"]   # remove entry with key "curve"

print(dictionary.keys())

print("curve" in dictionary) # check include in or not) # check include in or not
dictionary.clear() 

print(dictionary)
del dictionary  

#print(dictionary)  # it gives error because dictionary is deleted
dictionary=data.to_dict('series')

alist=[]

i=0

while i!=5000:

    dummy=np.around(np.random.uniform(1.0, 100.0) ,decimals=1)

    alist.append(dummy) 

    #print(alist)

    i+=1

dictionary["passing_accuracy"]=alist

#print(alist)

#print(dictionary["passing_accuracy"])

len(dictionary["passing_accuracy"])

print(i,"finish")







blist=[]

for value in dictionary['passing_accuracy']:

    if value>50:

        blist.append(value)



dictionary['passing_accuracy'].clear() #I have removed all entries in passing_accuracy key.

print("empty", dictionary['passing_accuracy'])

dictionary['passing_accuracy']=blist #I have entered values which are bigger than 50.

print("length of entries",len(blist)) #length of entries

print("Values are bigger than 50", dictionary['passing_accuracy'])
