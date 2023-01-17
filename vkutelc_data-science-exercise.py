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
myData = pd.read_csv('../input/data.csv')
myData.info() 
myData.head() # first 5 objects
myData.tail() # last 5 objects
myData.describe() #only numeric feature
myData.columns
myData.dtypes
myData.loc[:10,"Name":"Overall"] # write 0-10 line from Name columns to Overall columns 
#filtering

filtering_data = myData.Nationality == "Brazil"

filtering_data2 = myData.Age < 25

filtering_data3 = myData.Overall > 75

myData[filtering_data & filtering_data2 & filtering_data3]
myData.Overall.mean() # mean of Overall
myData.corr()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(myData.corr(), annot=True, linewidths=.10, fmt= '.1f',ax=ax)

plt.show()
# Line Plot

myData.SprintSpeed.plot(kind='line',color='blue',label='SprintSpeed',grid='True',linewidth=0.5,alpha = 0.5,linestyle=':')

myData.ShotPower.plot(color='red',label='ShotPower',grid='True',linewidth=0.5,alpha = 0.5,linestyle='-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = overall, y = potential

myData.plot(kind='scatter', x='Overall', y='Potential',alpha = 0.5,color = 'green', grid='True')

plt.xlabel('Overall')              # label = name of label

plt.ylabel('Potential')

plt.title('Overall-Potential Scatter Plot')            # title = title of plot

plt.show()
myData.Age.plot(kind = 'hist',bins = 64,figsize = (14,14))

plt.show()
#DİCTİONARY

dictionary = {'Turkey':'Fenerbahce','Spain':'Real Madrid','England':'Liverpool','Germany':'Dortmund','Italy':'Milan','Portugal':'Sporting Lizbon'}

print(dictionary.keys())

print(dictionary.values())
dictionary['France'] =  "PSG" # add

print(dictionary)

dictionary['Portugal'] = "Benfica" # update 

print(dictionary)

dictionary['Russia'] =  "Cska Moscow" # add

print(dictionary)

del dictionary['Russia']

print(dictionary)

print('Russia' in dictionary)
# LOOP

i=0

while i != 100:

    print('i is :',i)

    i = i+2

print(i ,' is equal to 100')    
print(dictionary)

for key,value in dictionary.items():

    print(key," : ",value)

print('')
for index,value in myData[['Age']][5:10].iterrows():

    print(index," : ",value)
def first_fun():

    """return defined t tuble"""

    t = (2,4,6,8)

    return t

a,b,c,d = first_fun()

print(a,b,c,d)
a = 8

def first():

    y = 10+a

    return y 

print(first())

    

 
def first():

    def second():

        a=2019

        b=1996

        c=a-b

        return c

    return second()*3

print(first())
def first(r,pi=3.14):

    """circumference"""

    result = 2*pi*r

    return result

circle = first(5)

toInt = int(circle) #double to int

print(toInt)

def f(*args):

    for i in args:

        print(i)

f(2,4,6,8,10)

age = lambda x: 2019-x

print(age(1996))
