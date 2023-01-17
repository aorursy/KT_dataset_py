# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/winequality-red.csv')
data.info()
data.head()
data.tail(10)
#Korelasyon 1e yakınsa aynı yönlü iyi ilişki

           #-1e yakınsa zıt yönlü iyi ilişki

data.corr
data.columns
#value_counts  Seçtiğimiz kolonda ilgisiz değer var mı bize gösterir

data.sulphates.value_counts()
f,ax=plt.subplots(figsize=(15,15))

sns.heatmap(data.corr(),annot=True,linewidth=.5,fmt='.1f',ax=ax)

plt.show()
#data.density.plot(kind='line',color='b',label='Density',linewidth=1,alpha=0.5,grid=True,linestyle=':')

#data.alcohol.plot(color='g',label='Alcohol',linewidth=1,alpha=0.5,grid=True,linesytle=':.')

#plt.legend(loc='upper right')

#plt.xlabel('x axis')

#plt.ylabel('y axis')

#plt.title('Line Plot')

#plt.show()

data.alcohol.plot(kind = 'line', color = 'pink',label = 'Alcohol',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.quality.plot(color = 'b',label = 'Quality',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()









data.plot(kind='scatter',x='sulphates',y='chlorides',alpha=0.5,color='purple')

plt.xlabel('Sulphates')

plt.ylabel('Chlorides')

plt.title('Sulphates-Chlorides Scatter Plot')

#y eksenindeki değerler o değere sahip kaç adet veri olduğu

#x eksenindeki değerler hangi değere sahip oldukları

data.sulphates.plot(kind='hist',bins=50,figsize=(10,10))



#plt.clf()=grafiği siler
dictionary={'France':'Burgonya','Italy':'Sicilya','America':'California'}

print(dictionary.keys())

print(dictionary.values())
dictionary['France']="Bordeaux"

print(dictionary)

dictionary['Italy']="Venice"

print(dictionary)

del dictionary['America']

print(dictionary)

print('France' in dictionary)

dictionary.clear()

print(dictionary)
series=data['quality']

print(type(series))

data_frame=data[['sulphates']]

print(type(data_frame))
print(5>1)

print(2==3)

print(True and False)
# tek filtere

x=data['quality']>7

data[x] 
#2 - filtre pandasda logical_and

#data[np.logical_and(data['quality']>7,data['sulphates']>0.8)]

#üstteki ve alttaki aynı anlam

data[(data['quality']>7)&(data['sulphates']>0.8)]
i=0

while i !=5:

    print('i is:',i)

    i+=1

print(i,'is equal to 5')
lis={1,2,5,6,7,9,3}

for i in lis:

    print('i is : ',i)

print('')

for index,value in enumerate(lis):

    print(index,":",value)

print('')
dictionary={'France':'Burgonya','Italy':'Sicilya','America':'California'}

for key,value in dictionary.items():

    print(key,":",value)

print('')



for index,value in data[['sulphates']][0:1].iterrows():

    print(index,":",value)
#Tuble

def tuble_ex():

    t=(1,2,3)

    return t

a,b,c =tuble_ex()

print(a,b,c)

    
#SCOPE

import builtins

dir(builtins)
def f(*args):

    for i in args:

        print(i)

f(1)

f(1,3,5,7)

def f(**kwargs):

    for key,value in kwargs.items():

        print(key,"",value)

f(country='turkey',capital='izmir',population=125648)

#LAMBDA FUNCTION

kare=lambda x:x**2

print(kare(4))
#ANONYMOUS FUNCTION

#lambda fonksiyon gibi ama birden  fazla argüment alabilir.

number_list=[1,3,4]

y=map(lambda x:x**2,number_list)

print(list(y))
#ITERATORS

name="gamze"

it=iter(name)

print(next(it))

print(*it)
liste1=[1,3,5,7,9]

liste2=[0,2,4,6,8]

z=zip(liste2,liste1)

print(z)

z_list=list(z)

print(z_list)
un_zip=zip(*z_list)

un_list1,un_list2=list(un_zip)

print(un_list1)

print(un_list2)

print(type(un_list2))
data.shape
data.describe()

data.boxplot(column='alcohol',by='quality')

plt.show()

#yuvarlak işareti olanlar outlierslarımız(aykırı değer)

#en yukardaki yuvarlak fazla abartılmış gibi olduğunu söyler

#TIDY DATA

#melt() function=bu datadan yeni data oluşturmak  

data_new=data.head(7)

data_new
#id_vars=değişmesini istemediğimiz kolon adı

#value_vars = yeni oluşmasını istediğimiz datadan ortaya çıkacak olan 

#değerlerin hangi kolonlara göre şekilleneceği

melted=pd.melt(frame=data_new,id_vars='quality',value_vars=['alcohol','sulphates'])

melted
#melted.pivot(index='quality',columns='variable',values='value')
 #CONCATENATING ,

    #dikey concat

data1=data.head()

data2=data.tail()

conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True)

conc_data_row
#yatay concat

#axis=1 sütunları yanyana birleştir

data1=data['quality'].head()

data2=data['alcohol'].head()

conc_data_col=pd.concat([data1,data2],axis=1)

conc_data_col
data.dtypes
data.info()
#missing value

#data1=data

#data1['quality'].dropna(inplace=True)
assert data['quality'].notnull().all()
#data['quality'].fillna('empty',inplace=True)=datayı emptyle doldur
country=["Turkey","Italy"]

race=["Turkish","Italian"]

list_label=["country","race"]

list_col=[country,race]

zipped=list(zip(list_label,list_col))

data_dict=dict(zipped)

df=pd.DataFrame(data_dict)

df
df["city"]=["Istanbul","Venice"]

df
df["income"]=0

df
data1=data.loc[:,["density","alcohol","quality"]]

data1.plot()
data1.plot(subplots=True)

plt.show()
data1.plot(kind="hist",y="alcohol",bins=50,range=(7,16),normed=True)

#normed=normalizasyon
data.columns
fig,axes=plt.subplots(nrows=2,ncols=1)

data1.plot(kind="hist",y="alcohol",bins=50,range=(0,250),normed=True,ax=axes[0])

data1.plot(kind="hist",y="alcohol",bins=50,range=(0,250),normed=True,ax=axes[1],cumulative=True)

plt.savefig('graph.png')

plt
time_list=["1998-10-05","1995-04-04"]

print(type(time_list[1]))

datetime_object=pd.to_datetime(time_list)

print(type(datetime_object))
data2=data.head()

date_list=["1995-02-12","1995-03-12","1995-04-12","1998-02-12","1999-02-12"]

datetime_object=pd.to_datetime(date_list) 

data2["date"]=datetime_object

data2=data2.set_index("date")

data2
print(data2.loc["1995-02-12"])

print(data2.loc["1995-02-12":"1999-02-12"])

#bu tarihler arasında olan indisteki eleman bilgilerine ulaştık.
data2.resample("A").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean()
data.head()
#data["chlorides"][2]

data.loc[2,["chlorides"]]
data[["chlorides","pH"]]
data.loc[1:10,"density":"quality"]
boolean=data.quality>7

data[boolean]
first_filter=data.alcohol>12

second_filter=data.quality>7

data[first_filter & second_filter]
data.quality[data.alcohol>12]
def div(n):

    return n/2

data.alcohol.apply(div)
data["total acidity ratio"]=data['fixed acidity']+data['volatile acidity']+data['citric acid']

data.head(9)
data.columns
print(data.index.name)

data.index.name="index_number"

data.head()
data3=data.copy()

data3.index=range(1,1600,1)

data3.head()
dic={"threatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,23,58],"age":[15,26,45,32]}

df=pd.DataFrame(dic)

df
df.pivot(index='threatment',columns='gender',values='response')
df2=df.set_index(["threatment","gender"])

df2
df2.unstack(level=1)
df3=df2.swaplevel(0,1)

df3
df
pd.melt(df,id_vars="threatment",value_vars=["age","response"])
df.groupby("threatment").mean()
df.groupby("threatment").age.max()
df.groupby("age").max()
df.groupby("threatment")[["age","response"]].min()
df.info()