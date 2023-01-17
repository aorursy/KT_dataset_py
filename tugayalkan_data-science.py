# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  #Visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read_csv ile hazır datasetimizi data isimli dataframeimize aktarıyoruz

data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv") 

data.info()
data.corr()  #ürünler arası corelation ı verir
#corelation Map.  ~ Eğer 2 feature arasında corelation 1 ise bunlar birbiri ile doğru orantılıdır(Pozitif corelation). (mesela, bir evin oda sayısı artarsa fiyatı artar)

f,ax = plt.subplots(figsize = (12,12))

sns.heatmap(data.corr(), annot=True, linewidths = 5, fmt = '.1f',ax = ax) 

plt.show() # uyarı verdiği için ekledik, eklemesek de aşağıdaki görseli elde edebiliriz

# seaborn corelasyon verisini alıyor || annot=True demek yazan sayıların gözükmesini sağlar || linewidths aradaki çizgi kalınlğı, 

#fmt precision ı 1 olsun yani 0dan sonra 1 rakam yazsın || ax figurun size ını belirler
data.head(10)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# has the same task as the followinng code block



#plt.scatter(data.Attack, data.Defense,color='red',alpha=0.5)

#plt.show()

#scatter plot 

#x = attack, y = defense



data.plot(kind = 'scatter', x ='Attack', y = 'Defense', alpha = 0.5, color = 'red')

plt.xlabel('Attack')

plt.ylabel('Defence')

plt.title('Attack Defense Scatter Plot')

#plt.show()
# Histogram

#bins = number of bar in figure



data.Speed.plot(kind = 'hist', bins = 50, figsize = (10,10))

plt.show()
#clf() = cleans it up again you can start a fresh

data.Speed.plot(kind = 'hist', bins = 50)

plt.clf()

# We can't see plot due to clf()
#create dictionary and look its keys and values

dictionary = {'spain':'madrid','usa':'vegas'}

print(dictionary.keys())

print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dictionary['spain'] = "barcelona"  # update existing entry

print(dictionary)

dictionary['france'] = "paris"     # add new entry

print(dictionary)

del dictionary['spain']            # remove entry with key 'spain'

print(dictionary)

print('france' in dictionary)      # check include or not

dictionary.clear()                 # remove all entries in dict

print(dictionary)

 

# In order to run all code you need to take comment this line

# del dictionary         # delete entire dictionary     

print(dictionary)       # it gives error because dictionary is deleted
data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")
series = data['Defense']   # data['Defense'] = series

print(type(series))

data_frame = data[['Defense']]

print(type(data_rame))
# Comparison operator

print(3 > 2)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
# 1 - Filtering Pandas data frame

x = data['Defense'] > 200 # There are only 3 pokemons who have higher defense value than 200

data[x]
# 2 - Filtering pandas with logical_and

# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100

data[np.logical_and(data['Defense']>200, data['Attack']>100 )]
# This is also same with provious code line. Therefore we can alse use '&' for filter ring

data[(data['Defense']>200) & (data['Attack']>100)]
# Stay in loop if condition( i is not equal 5) is true

i=0

while i != 5:

    print('i is: ',i)

    i +=1

print(i,' is equal to 5')
# Stay in loop if condition( i is not equal 5) is true

lis = [1,2,3,4,5]

for i in lis:

    print('i is: ',i)

print('')



# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(lis):

    print(index," : ", value)

print('')



# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.



dictionary = {'spain':'madrid','france':'paris'}

for key, value in dictionary.items():

    print(key," : ", value)

print('')



# For pandas we can achieve index and value

for index,value in data[['Attack']][0:1].iterrows():

    print(index," : ",value)
