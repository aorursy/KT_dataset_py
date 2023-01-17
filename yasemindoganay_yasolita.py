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
# iki veri kümesini de programa tanıttık
data = pd.read_csv('../input/athlete_events.csv')
data1= pd.read_csv('../input/noc_regions.csv')
#verimizin içeriği hakkında bilgi aldık. Veri kümesinde kaç adet veri bulunduğunu, kaç adet sütun 
#bulunduğunu , verilerin tiplerini(tamsayı,ondalıklı sayı,yazı vb), özellikler içinde hangi tipten 
#veri olduğunu
data.info()
# Özellikler(sütunlar) arasındaki bağıntıyı gösterir. eğer değerler 1'e yakın çıkarsa doğru orantılı
# -1'e yakın çıkarsa ters orantılı ve 0 çıkarsa bağıntı yok demektir.

data.corr()
# Verinin sahip olduğu sütun adlarını bize gösterir.
data.columns 


data1.columns
data1.info()

#Bağıntı haritası
#
f,ax = plt.subplots(figsize=(10,10))  # figürümüzün boyutunu belirledik.
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax) #corr metoduyla yaptığımız işi
#görselleştirmemizi sağlar, fmt='.1f' = tabloda 0 dan sonra bi tane değer yazdırılmasını istiyor.
plt.show()
data.head(20) # verimizin ilk 20 satırını yazdırır. Bu komutun defoult değeri 5 tir.
data1.head()
#Age ve Year özelliklerinin grafiğini tek grafikte çizdiriyoruz.
# alpha dediğimiz şey grafiğin saydamlığı.

data.Age.plot(kind = 'line', color = 'm',label = 'Age',linewidth=1.5,alpha = 0.5,grid = True,linestyle = ':')
data.Year.plot(color = 'r',label = 'Year',linewidth=1.5, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')   #grafiğin çizgi adlarını sağ üste yazdırır  
plt.xlabel('x ekseni')              
plt.ylabel('y ekseni')
plt.title('age-year graphic')   #grafiğin başlığını belirledik.
plt.show()
#Age ve Year özellikleri arasında nasıl bi bağıntı olduğunu anlamak için scatter plot kullanıyoruz 
#(doğru orantı ters orantı vb.)
data.plot(kind='scatter', x='Age', y='Year',alpha = 0.5,color = 'pink')
plt.xlabel('Age')              
plt.ylabel('Year')
plt.title('age-year scatter graphic')  
plt.show()
# Age özelliğinin grafiğini histogram(istatistiksel sonuçlar elde etmek için kullanılır)
data.Age.plot(kind = 'hist',bins = 100,figsize = (10,10))
plt.show()

#plt.clf() : Bu kod ile çizdiğimiz grafiği silebiliriz.
# PANDAS kütüphanesi iki çeşit veri tipinden oluşuyor. 'Series' ve 'DataFrame'
series = data['Age']        # Age verimizi PANDAS kütüphanesinin metodu olan series adlı değişkene atadık.
                            # veriler vektör şeklinde yazılır
print(type(series))         # Verimizin türünü seri yaptık.
data_frame = data[['Age']]  # Age verimizi PANDAS kütüphanesinin metodu olan data_frame adlı değişkene atadık.
print(type(data_frame))     # Verimizin türünü dataframe yaptık.
x = data['Age'] > 50    # Age özelliği 50 den büyük olan verileri x e atadık ve yazdırdık. Yalnız 
# bu kodu yazarsak age i 50 den büyük olanları true olmayanları false döndürür.
data[x]   # bu kodu da eklersek bize sadece age i 50 den büyük olanları yazdırır.
# np.logical_and : Numpy kütüphanesine bağlı bir metot 've' bağlacını kullanabilmemiz için ihtiyacımız var.
data[np.logical_and(data['Age']>50, data['Weight']>120,)] #Age i 50den büyük ve Weight i 
#120den büyük olanları yazdırdık
#Bir önceki kod verine '&' bağlacını kullandığımızda da aynı işlemi yapmış oluruz.
data[(data['Height']<200) & (data['Year']>2006)]
#Ödevim hazır
#►2. ödev
import builtins
dir(builtins)
data.info()
data.columns
ortalama_boy= sum(data.Height)/ len(data.Height)
print("ortalama boy =" , ortalama_boy)
data["boy_oranı"]=["uzun" if i > ortalama_boy else "kısa" for i in data.Height]
data.loc[:10,["boy_oranı","Height"]]
data.describe()

print(data['Age'].value_counts(dropna =False))
data.boxplot(column='Weight',by = 'Height')
data.info()
data_new = data.head()    
data_new
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Sex','Age'])
melted 
melted.pivot(index = 'Name', columns = 'variable',values='value')
data2 = data.head()
data3 = data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row
data4 = data['Weight'].head()
data5 = data['Height'].head()
conc_data_col = pd.concat([data4,data5],axis =1) 
conc_data_col
data.dtypes
data['Age'] = data['Age'].astype('category')
data['Height'] = data['Height'].astype('float')
data.info()
data.dtypes
data["Year"].value_counts(dropna =False)
data["Age"].value_counts(dropna =False)
data1=data   
data1["Age"].dropna(inplace = True)
assert 1==1
assert 1==2
assert  data['Age'].notnull().all()
data["Year"].fillna('empty',inplace = True)
assert data['Year'].notnull().all()
assert data ["Height"].fillna('empty',inplace = True)
assert data.columns[1] == 'Name'
assert data.columns[2] == 'naber'
assert data.Year.dtypes == np.int64
# sözlük ile dataframe oluşturma
ulke = ["Türkiye","Mısır"]
nufus = ["45","78"]
list_label = ["ulke","nufus"]
list_col = [ulke,nufus]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
df["sehir"] = ["Eskişehir","Kahire"]
df
df["gelir"] = 0 
df
data1 = data.loc[:,["Name","Height","Weight"]]
data1.plot()
plt.show()
data.columns
data1 = data.loc[:, ["Weight","Year"]]
data1.plot
data1.plot(subplots = True)
plt.show()
time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
# görmeyi istemediğimiz program mesajlarını görünmez yaptık.
import warnings
warnings.filterwarnings("ignore")

data2 = data.head()    # verinin ilk 5 değerini aldık.
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list) #datelist in tipini datetime a çevirdik
data2["date"] = datetime_object   #datetime_objectlerden oluşan date özelliği yarattık
data2= data2.set_index("date")    # data2 mizin indexleri artık date adlı değişken oldu.
data2 
print(data2.loc["1993-03-16"])  # veride "1993-03-16" indexine sahip satırı yazdırdık.
print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()   #veriyi yıllara göre ayırıp aynı olan yılların
                             #özelliklerinin ortalamasını aldık.  A, bu kütüphanede yıl demek.
data2.resample("M").mean()  # aynı işlemi aylar için yaptık
data2.resample("M").mean().interpolate("linear")  #değeri NaN olanları en küçük ve en büyük
                                                  #sayı arasında lineer olarak doldur.
data.head()
data = pd.read_csv('../input/athlete_events.csv')
data = data.set_index("ID")     #verimizin indexini ID ye atadık artık index imiz 1den başlıyor
data.head()

data["Team"][1]  #Team sütununu birinci satırını seçtik.
data.Team[1]    #yukarıdaki kodla aynı görevi yapıyor.
data.loc[1,["Team"]]   #1. satır Team adlı sütunu aldık
data[["Year","Age"]]
data = pd.read_csv('../input/athlete_events.csv')
data.head()
data1 = data.set_index(["Team","Sport"]) 
data1.head(100)


















