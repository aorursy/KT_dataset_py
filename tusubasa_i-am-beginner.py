# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/avocado.csv')
data.corr()
data.head(15)
data.tail(15)
#correlation map
f,ax=plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
data.columns=[each.lower() for each in data.columns] #columnları kucuk harflerle yenıden yazar
data.columns=[each.split()[0]+"_"+each.split()[1] if len(each.split())>1 else each for each in data.columns]
data.info()
data.columns
#lineplot
data.total_bags.plot(kind="line", color="g", label="total_bags", linewidth=1, alpha=0.5, grid=True, linestyle=":")
data.small_bags.plot(kind="line", color="r", label="small_bags", linewidth=1, alpha=0.5, grid=True, linestyle="-.")
data.averageprice.plot(kind="line", color="r", label="averageprice", linewidth=1, alpha=0.5, grid=True, linestyle="-.")
data.total_volume.plot(kind="line", color="r", label="total_volume", linewidth=1, alpha=0.5, grid=True, linestyle="-.")
data.large_bags.plot(kind="line", color="r", label="large_bags", linewidth=1, alpha=0.5, grid=True, linestyle="-.")
data.xlarge_bags.plot(kind="line", color="r", label="xlarge_bags", linewidth=1, alpha=0.5, grid=True, linestyle="-.")
data.year.plot(kind="line", color="r", label="year", linewidth=1, alpha=0.5, grid=True, linestyle="-.")
plt.legend(loc="upper right")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("lineplot")
plt.show()
#scatter plot
data.plot(kind="scatter", x="total_bags", y="small_bags", alpha=0.5, color="r")
plt.xlabel("total_bags")
plt.ylabel("small_bags")
plt.title("Bags scatter Plot")
plt.show()
#scatter plot
data.plot(kind="scatter", x="year", y="small_bags", alpha=0.5, color="r")
plt.xlabel("year")
plt.ylabel("small_bags")
plt.title("Bags scatter Plot")
plt.show()
data.total_bags.plot(kind="hist", bins=200, figsize=(15,15))
data.small_bags.plot(kind="hist", bins=200, figsize=(15,15))
data.averageprice.plot(kind="hist", bins=200, figsize=(15,15))
data.total_volume.plot(kind="hist", bins=200, figsize=(15,15))
data.large_bags.plot(kind="hist", bins=200, figsize=(15,15))
data.xlarge_bags.plot(kind="hist", bins=200, figsize=(15,15))
#plt.clf()# figuru siliyor
#dictionary oluşturma ve değerlerine bakma
dictionary={"spain":"madrid", "usa":"vegas"}
print(dictionary.keys())
print(dictionary.values())
dictionary["spain"]="barcelona" #güncelleme
print(dictionary)
dictionary["france"]="paris" #ekleme
print(dictionary)
del dictionary["spain"] #silme
print(dictionary)
print("france" in dictionary) #key var mı kontrol et
dictionary.clear() #dicitonary komple temizlenir
print(dictionary)
series=data["total_bags"] #vector
print(type(series))
data_frame=data[["total_bags"]] #liste
print(type(data_frame))
print(3>2)
print(3<2)
print(3==2)
print(3!=2)
print(True and False)
print(True or False)
x=data["total_bags"]==0
data[x]
data[np.logical_and(data["total_bags"]<20, data["small_bags"]<20)]
data[(data["averageprice"]<2) & (data["total_bags"]<10)]
# while loop
i=0
while i!=5:
    print("i is: ", i)
    i+=1 
print(i, " 5'e eşit")
#for loop
lis=[1,2,3,4,5]
for i in lis:
    print("i is: ",i)
print("")

for index, value in enumerate(lis):
    print(index," : ", value)
print("")

dictionary1={"spain":"madrid", "france":"paris", "usa":"vegas"}
for key,value in dictionary1.items():
    print(key," : ", value)
print("")

for index,value in data[["total_bags"]][0:2].iterrows():
    print(index," : ", value)
print("")
#lambda fonkiyon  yazmanın hızlı yolu
square=lambda x: x**2
print(square(25))

tot=lambda x,y,z:x+y+z
print(tot(1,2,3))
#anonim fonksiyon
number_list=[1,2,3]
y=map(lambda x:x**2, number_list)
print(list(y))
#iteratör örneği
name="dursun"
it=iter(name)
print(next(it))
print(*it)
#zip örneği

list1=[1,2,3,4]
list2=[5,6,7,8]
z=zip(list1,list2)
print(z)
z_list=list(z)
print(z_list)

un_zip=zip(*z_list)
un_list1,un_list2=list(un_zip)#unzip tuple a cevirir
print(un_list1)
print(un_list2)
print(type(un_list2))
print(type(list(un_list1)))
#list comprehension örneği
num1=[1,2,3]
num2=[i+1 for i in num1]
print(num2)

#conditionals on iterable
num1=[5,10,15]
num2=[i**2 if i==10 else i-5 if i<7 else i+5 for i in num1]
print(num2)
threshold=sum(data.total_bags)/len(data.total_bags)
print(threshold)
data["seviye"]=["high" if i>threshold else "low" for i in data.total_bags]
data.loc[:50, ["seviye", "total_bags"]]
print(data.describe())
print(data.region.unique())
pl.figure(figsize=(12,5))
pl.title("Distribution Price")
ax = sns.distplot(data["averageprice"], color = 'r')
pl.show()
def tuble_ex():
    """return defined t tuble"""
    t=(1,2,3)
    return t
a,b,c=tuble_ex()
print(a,b,c)
def f(a, b=1,c=2):
    y=a+b+c
    return y
print(f(5))
print(f(5,4,3))
print("")

def a(*args):
    for i in args:
        print (i)
a(1)
print("")

def a1(**kwargs):
    """print key and value of dictionary"""
    for key,value in kwargs.items():
        print(key," : ",value)
a1(country="spain", capital="madrid", population=500245)
x=2
a=3
def f():
    x=3
    return x
def g():
    x=a*3
    return x
print(x)
print(f())
print(g())
import builtins
dir(builtins)

#nested func (içiçe fonksiyonlar)
def square():
    """return swuare of value"""
    def add():
        """add two local variable"""
        x=2
        y=3
        z=x+y
        return z
    return add()**2
print(square())
data.shape
print(data["region"].value_counts(dropna=False))
print(data["total_bags"].value_counts(dropna=False))
print(data["type"].value_counts(dropna=False))
data.info()
data.columns
data.boxplot(column="total_bags", by="type")
data.boxplot(column="4046", by="type")
data.boxplot(column="year", by="type")
plt.show()
data_new=data.head(50)
data_new
melted=pd.melt(frame=data_new, id_vars="date",value_vars=["total_bags", "small_bags",])
melted
melted.pivot(index="date", columns="variable", values="value")
data.tail(50)
data1=data.head()
data2=data.tail()
conc_data_row=pd.concat([data1,data2], axis=0, ignore_index=True)
conc_data_row


data3=data["date"].head(10)
data4=data["total_bags"].head(10)
conc_data_col=pd.concat([data3,data4],axis=1)
conc_data_col
data.dtypes
data["date"]=data["date"].astype("str")
data.dtypes
data["total_bags"].dropna(inplace=True)
assert data["total_bags"].notnull().all()
country=["Turkey","Spain"]
population=["125","520"]
list_label=["country","population"]
list_col=[country,population]
zipped=list(zip(list_label, list_col))
data_dict=dict(zipped)
df=pd.DataFrame(data_dict)
df
df["capital"]=["Ankara","Madrid"]
df
df["income"]=0
df
data5=data.loc[:,["large_bags","small_bags","total_volume"]]
data5.plot()
plt.show()
data.plot(subplots=True)
plt.show()
data.plot(kind="scatter",x="total_volume",y="total_bags")
plt.show()

data.plot(kind="hist", y="small_bags", bins=50, range=(0,1250),normed=True)
fig,axes=plt.subplots(nrows=2,ncols=1)
data.plot(kind="hist", y="small_bags", bins=50,range=(0,1250), normed=True,ax=axes[0])
data.plot(kind="hist", y="small_bags", bins=50,range=(0,1250), normed=True,ax=axes[1],cumulative=True)
plt.savefig("graph.png")
plt
data.describe()
data.dtypes
time_list=["2018-05-13","2018-04-26"]
print(type(time_list[1]))
datetime_object=pd.to_datetime(time_list)
print(type(datetime_object))
#data=data.set_index("date")# datetime index .. timeserise için bunu yapmaya çok uğraştım
#data.index=pd.to_datetime(data.index)
data.resample("M").mean()
data.resample("A").mean()

data.columns
data.head()
data.index.dtype
data.columns

print(data.loc["2015-12-06"])
print(data.loc["2015-12-27":"2015-12-13"])
#data.drop(["unnamed:_0"],axis=1,inplace=True)
data=data.set_index("#")
data.head()
data.drop(["seviye"],axis=1,inplace=True)
data.head()
data.loc[1,["4046"]]
print(type(data["4046"])) #series
print(type(data[["4046"]])) #dataframe
boolean=data.total_bags<200
data[boolean]
fil1=data.total_bags<100
fil2=data.small_bags<200
data[fil1&fil2]
data.small_bags[data.total_bags<200]
def div(n):
    return 2*n
data.total_bags.apply(div)
data.total_bags.apply(lambda n:n*2)
data["totals"]=data.total_bags+data.total_volume
data.head()
print(data.index.name)
data.index.name="index_name"
data.head()
data.head()
data.index=range(1,18250,1)
data5=data.copy()
data5.index=range(100,18349,1)
data5.head()
data.loc[2:15,"4046":"4770"]
data.loc[15:2:-1,"4046":"4770"]
data.loc[2:10,"4046":]
data[["4046","4225"]]
data = pd.read_csv('../input/avocado.csv')
data.head()
data10=data.set_index(["region","type"])
data10.head(150)
dic={"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df=pd.DataFrame(dic)
df
df.pivot(index="treatment",columns="gender", values="response")
df1=df.set_index(["treatment","gender"])
df1
df1.unstack(level=0)
df2=df1.swaplevel(0,1)
df2
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df
df.groupby("treatment").mean()
df.groupby("treatment").age.mean()
df.groupby("treatment")[["age","response"]].mean()
data["4046"][1]
