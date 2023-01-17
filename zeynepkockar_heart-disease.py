import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 
import pandas as pd

data=pd.read_csv('../input/heart.csv')
#convert and copy

liste1=[data.chol]

liste2=[data.cp]

array1=np.array(liste1)

array2=np.array(liste2)

#vertical

array3=np.vstack((array1,array2))

print(array3)

#horizontal

array4=np.hstack((array1,array2))

print(array4)
data.info()
data.corr()

data.describe()
print(data.loc[:3,"sex"])
#fıltrering pandas data frame

filtre1=data.chol>200

filtre_data=data[filtre1]

print(filtre_data)
#correlation map

f,ax = plt.subplots(figsize=(17, 17))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

data.head(10)
data.tail(10)
x=2

def f():

    y=x*data.ca

    return y

print (x) #global scope

print (f()) #local scope

ageincrease= lambda age:age*3

print(ageincrease(data.age))
#practice ıterators

name="dataıteam"

it=iter(name)

print(*it)
#zip() lists

list1=[data.age]

list2=[data.cp]

z=zip(list1,list2)

print(z)

zlist=list(z)

print(zlist)

ax=sns.swarmplot(x='cp',y='trestbps',data=data)

plt.xlabel('cp')

plt.ylabel('trestbps')

plt.show