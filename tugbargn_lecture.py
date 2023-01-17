



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns #visualization tool



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        
data = pd.read_csv("../input/pokemon-challenge/pokemon.csv")

data.info()

data.corr()#feature arasında correlation verir 1->positive 0->negative
#correlationmap

f,ax = plt.subplots(figsize = (18,18))#figsize otomatik olarak da belirleniyor

sns.heatmap(data.corr(),annot = True, linewidths =.5,fmt = ".1f",ax = ax)

plt.show()
data.head(10)
data.columns
data.Speed.plot(kind='line',color ='g',label="Speed",linewidth = 1,alpha = 0.5,grid = True,linestyle = ":")

data.Defense.plot(color='r',label='Defense',linewidth=1,alpha=0.5,grid=True,linestyle='-.')

plt.legend(loc='upper right') #puts label into plot

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show() # Text(0.5, 1.0, 'Line Plot') altta u yazının çıkmasını önler
#Scatter Plot

# x = attack y=defense

#plt.scatter(data.Attack,data.Defense,color='red',alpha=0.5)

data.plot(kind='scatter',x='Attack',y='Defense',alpha=0.5,color='red') #üstteki alternatifi

plt.xlabel("Attack")

plt.ylabel("Defense")

plt.title("Attack Defense Plot")
#histogram

#bins = number of bar in figure , çubuk sayısı

data.Speed.plot(kind="hist",bins = 50,figsize=(15,15))

plt.show()
#clf() = cleans it up again you can start a fresh

data.Speed.plot(kind = "hist",bins = 50)

plt.clf()
dictionary ={'spain': 'madrid','usa':'vegas'} #create dictionary

print(dictionary.keys())

print(dictionary.values())
dictionary['spain']= 'barcelona'

print(dictionary)

dictionary['france']='paris'

print(dictionary)

del dictionary['spain'] #remove

print(dictionary)

print('france'in dictionary) #check include or not

dictionary.clear() #remove all

print(dictionary)
#del dictionary

print(dictionary)
#pandas

data = pd.read_csv("../input/pokemon-challenge/pokemon.csv") #csv->comma seperates value

#data.head()
series = data['Defense']

print(type(series))

data_frame = data[['Defense']] #dataframe böyle oluşturuluyor

print(type(data_frame))
print(3>2)

print(True or False)
#filtering pandas data frame

x = data['Defense']>200

data[x] #true durumlarını yazdırır
data['Defense']>200 #bütün durumları yazar
data[np.logical_and(data['Defense']>200,data['Attack']>100)]#istediğimiz koşullara göre filtreledik
data[(data['Defense']>200) & (data['Attack']>100)] #üsttekinin alternatifi
i = 0

while i != 5:

    print("i is :",i)

    i = i+1

print("i is equal to 5")
lis = [1,2,3,4,5]

for i in lis:

    print("i is :",i)

print(" ")



for index,value in enumerate(lis):

    print(index," : ",value)

print(" ")



dictionary = {"spain":"madrid","france":"paris"}

for key,value in dictionary.items():

    print(key," : ",value)

print(" ")



for index,value in data[['Attack']][0:1].iterrows():

    print(index," : ",value)