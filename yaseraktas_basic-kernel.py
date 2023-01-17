# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        import seaborn as sns

        

        

        

 



# Any results you write to the current directory are saved as output.
data1=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')

data2=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
data1.columns

data1.columns=['id','title','cast','crew']

data2=data2.merge(data1,on='id')
data2.columns
data2.head(5)
x=data2['vote_average']>8.2 

data2[x]
yenidata=data2[np.logical_and(data2['vote_average']>8, data2['vote_count']>2000, )]

yenidata.sort_values("budget")
plt.figure(figsize=(12,12))

plt.barh(yenidata['original_title'].head(10),yenidata['budget'].head(10),color='g')

plt.gca().invert_yaxis()

plt.xlabel("Budget")

plt.ylabel("Popular Film Name")

plt.title("Popular Movies")
yenidata2=data2[np.logical_and(data2['vote_average']>8, data2['vote_count']>2000, )]

yenidata2.sort_values("revenue")
plt.figure(figsize=(12,12))

plt.barh(yenidata2['original_title'].head(10),yenidata2['revenue'].head(10),color='r')

plt.gca().invert_yaxis()

plt.xlabel("revenue")

plt.ylabel("Popular Film Name")

plt.title("REVENUE")
a=yenidata2["revenue"].mean()

a
b=yenidata2["budget"].mean()

b
good_movies=yenidata2.copy()



def winmoney(z,b=b,a=a):

    sayi1=z['revenue']

    sayi2=z['budget']

    return sayi1-sayi2



good_movies['win'] = good_movies.apply(winmoney, axis=1)



good_movies=good_movies.sort_values('win', ascending=False)

good_movies[['original_title', 'vote_count', 'vote_average','revenue','budget','win']].head(10)

yenidata2.describe()

 
yenidata2["genres"]
training = yenidata2["genres"][95]

len(training.split(","))

training.split(",")[1].split(":")[1].split('"')[1]
film1 = yenidata2[yenidata2["original_title"]=="The Dark Knight"]

film2 = yenidata2[yenidata2["original_title"]=="Interstellar"]

pd.concat([film1,film2])
data2.columns
plt.subplots(figsize=(12,10))

list1=[]

for i in yenidata2['genres']:

    list1.extend(i)

ax=pd.Series(list1).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('summer_r',10))

for i, v in enumerate(pd.Series(list1).value_counts()[:10].sort_values(ascending=True).values): 

    ax.text(.8, i, v,fontsize=12,color='white',weight='bold')

ax.patches[9].set_facecolor('r')

plt.title('Top Genres')

plt.show()
def tuble_training():

    

    t=('yaser',"mucahit","aktas")

    

    return t

v,b,n = tuble_training()



print(v,b,n)
b=25 #global value

def g():

    

    b=5 #local value

    return b

print(b)

print("")

print(g())
import builtins

dir(builtins)
def kare():

   

    def toplama():

        a=7

        b=8

        d=a+b

        return d

    return toplama()**2

print (kare())
def f(a,b=1,c=2):



    z=a+b+c

    return z

print(f(8))

print(f(8,7,6))
def f (*args):

    for i in args:

        print(i)

f(1)

print("")

f(1,2,3,4,5,6)
def f (**kwargs):



    for key,value in kwargs.items():

        print(key,"",value)

f(country='turkey',city='istanbul',numara=343434)
toplama=lambda x,y,z: x+y+z

print(toplama(1,2,3))



bölme=lambda x,y : x/y

print(bölme(4,2))
training_list=['yaser','aktas']

a=map(lambda x:x+x,training_list)

print(list(a))

example="TURKEY"

it=iter(example)

print(next(it))

print(*it)
list1=["yaser",'mucahit',"aktas"]

list2=[1,2,3,4]

z=zip(list1,list2)

print(z)

z_list=list(z)

print(z_list)
un_zip=zip(*z_list)

un_list1,un_list2=list(un_zip)

print(un_list1)

print(un_list2)

print(type(list(un_list1)))
d1=['yaser',"aktas","mucahit"]

d2=[i+i for i in d1]

print(d2)
num1=[5,8,17]

num2=[i/2 if i==5 else i+5 if i<9 else i+10  for i in num1]

print(num2)
threshold= sum(data2.vote_average)/len(data2.vote_average)

data2["MEAN_VALUES"]=["high"if i>threshold else "low" for i in data2.vote_average ]

data2.loc[:10,["MEAN_VALUES","vote_average"]]

#print(threshold) #6.09

data2.head()
print(data2["genres"].value_counts(dropna=False))

#kaç farklı Tür olduğunu bulmak için kullanılan bir yöntem
data2.describe()
data2.boxplot(column='vote_average')

plt.show()
data2.boxplot(column='runtime')

plt.show()
data_new=data2.head(5)

data_new
melted=pd.melt(frame=data_new,id_vars='original_title',value_vars={'vote_average','popularity'})

melted
melted.pivot(index="original_title",columns='variable',values='value')
data_new1=data2.head()

data_new2=data2.tail()



conc_data_row=pd.concat([data_new1,data_new2],axis=0,ignore_index=True)

conc_data_row
data_new1=data2["vote_average"].head()

data_new2=data2["vote_count"].head()

conct_h=pd.concat([data_new1,data_new2],axis=1)

conct_h
data2.dtypes
data2['id']=data2["id"].astype("float64")

data2['revenue']=data2['revenue'].astype("object")
data2.dtypes
data2.info()
data2.tail(10)
data2["homepage"].value_counts(dropna=False)
assert data2.columns[1]=='genres'

#dogru ise cıktı vermez
data2.columns
country=['Turkey','Netherlands']

deger=['9','20']

list_training=['country','deger']

list_deger=[country,deger]

zipped=list(zip(list_training,list_deger))

data_dict=dict(zipped)

df=pd.DataFrame(data_dict)

df
df["capital"]=["istanbul","Valkenburg"]

df
df['sıfır_deger']=0

df
yenidata2=data2.loc[:,["vote_average","budget","vote_count"]]

yenidata2.plot()

plt.show()
yenidata2.plot(subplots=True)

plt.show()
yenidata2.plot(kind="scatter",x="vote_average",y="budget")

plt.show()
yenidata2['vote_average']=yenidata2["vote_average"].astype("int64")
yenidata2.dtypes
yenidata2.plot(kind="hist",y="vote_average",bins= 50,range=(0,35),normed=True)
#burası hata veriyor çalışmadı.



#fig, axes=plt.subplots(nrows=2,ncols=1)

#yenidata2.plot(kind="hist",y="vote_average",bins=50,range=(0,35),normed=True,ax=axes[0]

#yenidata2.plot(kind="hist",y="vote_average",bins=50,range=(0,35),normed=True,ax=axes[1],cumulative=True)

#plt.savefig("graph.png")

#plt
data2.head()
time_list=["1992-03-08","1992-04-12"]

print(type(time_list[1]))

datatime_object=pd.to_datetime(time_list)

print(type(datetime_object))
import warnings

warnings.filterwarnings("ignore")

data3=data2.head()

date_list=["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object=pd.to_datetime(date_list)

data3["date"]=datetime_object

data3=data3.set_index("date")

data3
print(data3.loc["1993-03-16"])

#tarihe göre bir filmi çağırdım

print(data3.loc["1992-03-10":"1993-03-16"])

#arasındaki filmleri çağırdım.
data3.resample("A").mean()

#yıllara göre tüm değerlerin ortalamasını al (A=YıIa göre)
data3.resample("M").mean()
data3.resample("M").first().interpolate("linear")

#tarihler arasında veriable değerlerini ortalama arasında doldurur.
data2['genres'][1]
data2.genres[1]
data2.loc[1,"genres"]
data2[["genres","budget"]]
print(type(data2["vote_average"])) #seriess

print(type(data2[["vote_average"]])) #date frame
data2.loc[1:10,"genres":"id"]



#10 tane genres le id arasındaki değerleri yazdır
data2.loc[10:1:-1,"genres":"id"]

#tersten genres ve id arasını yazdırma

data2.loc[1:10,"budget":]

#budgetten sonra olan featureları yazdırma
deneme=data2.vote_average>9.5

data2[deneme]
first_filter=data2.vote_average>9.5

second_filter=data2.vote_count>1

data2[first_filter&second_filter]
#vote_counta göre filtrele ama bana vote_averagalarını göster
data2.vote_average[data2.vote_count<2]
def div(n):

    return 2*n

data2.vote_average.apply(div)
data2.dtypes
data2["yeni"]=data2.vote_count+data2.vote_average

data2.head()
print(data2.index.name)

data2.index.name="yaser"

data2.head()
data2.head()

data3=data2.copy()

data3.index= range(100,4903,1)

data3.head()
data2=data2.set_index(["vote_average","vote_count"])

data2.head()

#"vote_average and vote_count" artık index olarak kullanmaya başladık.
dic={"treatman":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,20,30,50],"age":[15,20,32,40]}

df=pd.DataFrame(dic)

df
df.pivot(index="treatman",columns="gender",values="response")
df1=df.set_index(["treatman","gender"])

df1
df1.unstack(level=0)
df2=df1.swaplevel(0,1)

df2
df
pd.melt(df,id_vars="treatman",value_vars=["age","response"])
df
df.groupby("treatman").mean()  #gruplayıp ortalama deger alma
df.groupby("treatman").age.mean()
df.groupby("treatman")["age","response"].min()
df.info()