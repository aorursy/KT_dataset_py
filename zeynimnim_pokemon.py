#indentation error: tab boslugu birakma , dogru mu yazdin yazcaklarini kontrol et.



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns #visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

           

        

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
data
data.info()
#correlation map : iki feature arasindaki oranti 1 ise bunlar dogru orantilidir.

#Featurlar arasindaki iliskiyi gosterir correlation

#Bir evin oda sayisiyla fiyati pozitif correlated. Birbiriyle dogru orantilidir.



data.corr()



#ozelliklerin correlation katsayisi 0 sa iliski yok, 0-1 ise pozitif corr. var eksi 1 ile 0 arasi ise 

#negatif corr.  var
#correlation renkli gosterelim ehehe

f,ax = plt.subplots(figsize = (10,10))

sns.heatmap( data.corr(), annot = True, linewidths =.5, fmt = '.1f',ax=ax)

plt.show() #bunu dersen ustte cikan siyah yazi cikmaz
data.head(10)
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

# kind`a histogram line scatter dan birini yazarsin

data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Defense.plot(kind = 'line',color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc ='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()

#Her x ekseni pokemonlar, y ekseni de pokemonlarin hizi ve gucu.

#Asagidaki grafikte iki feature pokemonlara gore gosterilmis ama iki feature arasi iliski yok yani.
#Simdi scatter ile iki feature arasindaki iliskiyi gorelim



data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')

plt.xlabel('Attack')              # label = name of label

plt.ylabel('Defence')

plt.title('Attack Defense Scatter Plot')            # title = title of plot

#plt.show koysam alttaki siyah yazi gider

#Asagidaki grafige bir line fit edersem bu iki ozelligin dogru orantili oldugunu gorurum.

data.columns
#Yukardaki scatterin aynisini kisa yolla yapalim

plt.scatter(data.Attack, data.Defense, color = "red",alpha = 0.5)

plt.show()
# Histogram

# bins = number of bar in figure x ekseninde pokemonlarin sahip oldugu hizlar var

# bin cubuk sayisi bar sayisidir

# bin fazla olunca grafik daha ayrintili oluyor, hangi hiz degeri frekansi ne hepsini gosteriyor

data.Speed.plot(kind = 'hist',bins = 50,figsize = (15,15))

plt.show()

#plt.clf() grafigi yok eder temizler out u :p
dictionary = { 'Spain' : 'Madrid', 'USA' : 'Vegas'}

print(dictionary.keys())

print(dictionary.values())



#update existing entry

dictionary['Spain'] = "Barcelona"

print(dictionary)

#add entry

dictionary['Turkey'] = "Istanbul"

print(dictionary)

#delete entry

del dictionary['Spain']

print(dictionary)

#check include or not

print('Spain' in dictionary)

#clean all entries

dictionary.clear()

print(dictionary)

#dictionary degiskenini tamamen bellekten silmek

del dictionary

print(dictionary)

x = 3

print(x)
data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')

data
#series veya dataframe olusturursun. Series in ve dataframenin methodlari farklidir. 

#Hangisini kullacagina karar vermelisin.



series = data['Defense']

print(type(series))

data_frame = data[['Defense']]

print(type(data_frame))
#defansi 200 den buyuk olan pokemonlar

# Filtering Pandas Dara Frame

x = data['Defense'] > 200

data[x]
#Filtering pandas with logical and, yani iki tane sart olsun ehehe

data[np.logical_and(data['Defense'] > 200 , data['Attack'] > 100)]

# true and true olmalii

#zeynep mantigi

x = (data['Defense'] > 200) & (data['Attack'] > 100)

data[x]
i = 0

while (i < 5):

    print("i : ",i)

    i += 1

print("you are a disaster")



#Farkli data structurlarinda while i nasil kullaniriz bunu gorelim

#list data yapisinda while



liste = [1,2,3,4,5]

for each in liste:

    print(each)

print("honey honey where is money?")
#List kullanirken boyle for u kullanacagiz

for index, value in enumerate(liste):

    print("index: ",index,"and value: ",value)

print("hehe bittii")

#Dictionary kullanirken boyle for u kullanacagiz

dictionary =  {'Zeynep':20 , 'Yigit':24}

for key, value in dictionary.items():

    print("index: ",key,"and value: ",value)

print("your love is my cure")
#1. 2. ve 3. pokemonun atack degerlerini istiyorum

for index, value in data[['Attack']][0:3].iterrows():

    print(index+1,". pokemon ",value)

    
def tuble_x():

    """ return defined t tuble """

    t = (1,2,3)

    return t

a,b,c = tuble_x()

print(a,b,c)
#builtin hazir kullanilan degiskenler tipler vs otomati kgelen seyler

import builtins

dir(builtins)
def f(*args):

    for each in args:

        print(each)

        

f(1)

f(1,2,3,4)
#Dictionary tarzi icin ise farkli bir kullanimi mevcut **kwargs

""" print key and value of dictionary"""

def f(**kwargs):

    for key,value in kwargs.items():

        print(key,"  ",value)







f(country ='Spain', capital = 'Madrid', population = 6000000)
#USER DEFINED F.

def square(x):

    #buraya print koyunca basmiyor demek ki lamdayi yani kisa yolu seciyor

    return x**2



#LAMBDA F. Fonksiyon yazmanin kisa ve kolay yolu

square = lambda x: x**2



toplam = lambda x,y,z: x + y + z 

print(toplam(2,5,7))
liste = [1,2,3]

y = map(lambda x:x**2,liste) 

print(list(y))



def s(x, y = 2):



    return x**2

#zip() methodu cok kullanilir

#zip list

#2 list ziplenir ve tek list yapilir

list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1, list2)

z_list = list(z)

print(z_list)



un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip) #unzip return tuple

print(un_list1)

print(un_list2)

print(type(un_list2)) #unzip yapinca tuple ye ceviriyor ama biz onu list e cevirebiliriz

print(type(list(un_list2)))





#example of list comprehension. !!COK ONEMLI

num1 = [1,2,3]

num2 = [each + 1 for each in num1]

print(num2)
#condiitonals on iteration !! COK ONEMLI

num1 = [5,10,15]

num2 = [i**2 if i == 10 else i - 5 if i < 7 else i + 5 for i in num1]

print(num2)
#Pokemonlari hizli veya yavas diye siniflandiralim, ortalama hiz esik degerimiz olacak.

#len fonksiyonu item sayisini dondurur.

threshold = sum(data.Speed)/len(data.Speed)

print("threshold ", threshold)

data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]

data.loc[:10,["speed_level","Speed"]]

data['speed_level'] #sanirim cift tirnak degil galiba
data.Speed[:10]

# :)

data.shape #kac satir kac sutun var datada
#POKEMONLARIN TYPE LARI VAR, ATES POKUMONU BUZ POKEMONU FILAN

#SIMDI HER TIPTEN KAC POKEMON VAR ONA BAKALIMM

#TYPE BIR ATTRIBUTE ADI

print(data['Type 1'].value_counts(dropna = False)) # nan value varsa onlar da sayilacak
data.describe #sadece nuemrik degerleri verir, boolean tipler ifilan kaldirir
data_new = data.head(5)

data_new
#Melted Data Ornekleri

melted = pd.melt(frame = data_new,id_vars = 'Name',value_vars = ['Attack','Defense'])

melted

#cozemedim buradaki sorunu. Neysem. Meltingi yapamadim :) Defence yazmisim ehehe
#simdi de unmelted yapalim

melted.pivot(index = 'Name', columns = 'variable', values = 'value')
#Iki adet daraframe`i birlestirecegiz.



data1 = data.head()

data2 = data.tail()

#axis = 0 yaparsan satira eklenir 2. data yani x ekseninde bir artis olmaz :)

conc_data_row = pd.concat([data1,data2], axis = 0, ignore_index = True )

conc_data_row

data1 = data['Attack'].head()

data2 = data['Defense'].head()

conc_data_col = pd.concat([data1,data2], axis = 1)

conc_data_col

#data['Attack'] bu bir dataframe ornegidir
data.dtypes
#object -> categorical , int -> float donusumleri yapalim :

data['Type 1'] = data['Type 1'].astype('category')

data['Speed'] = data['Speed'].astype('float')

data.dtypes
data.info()
data["Type 2"]
data.columns

data["Type 2"].value_counts(dropna = False)

#386 tane Type2 degeri olmayan pokemon varmis

#dropna = False yazmazsak Nan sayisi gosterilmez
#Type2 si nan olan pokemonlari datamizdan silelim, ugrasamam nan man :p

data1 = data

data1['Type 2'].dropna(inplace = True)



assert 1 == 1 # dogru oldugu icin hicbir sey dondurmez
assert 1 == 2 # yanlis oldugu icin bize hata verecek
assert data['Type 2'].notnull().all() # bir sey dondurmuyor cunku hepsi not null olmasi durumu digru

#not null olmayanlari sildiydik.
data['Type 2'].fillna('empty', inplace = True)
assert data['Type 2'].notnull().all() 
assert data.columns[1] == 'Name'
data.dtypes
# np ye dikkat !!!

assert data.Speed.dtypes == np.int64