# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

from sklearn.preprocessing import LabelEncoder

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"

chipo =pd.read_csv(url, sep = "\t") #tablarla ayrıldıgı için sep= tab seçicez

chipo.head()
#ilk 10 data verimize bakalım

chipo.head(10)
#Veri kümesindeki gözlem sayısı nedir?

chipo.shape
chipo.info()
#Veri kümesindeki sütun sayısı nedir? 

len(chipo.columns)
#Tüm sütunların adını yazdırın

chipo.columns
#Veri seti nasıl indexlenir?

chipo.index
#En çok sipariş edilen ürün hangisidir?

chipo.head()

chipo.groupby(by="item_name").sum().sort_values(by="quantity",ascending=False).head(1)
#En çok sipariş edilen ürün için kaç ürün sipariş edildi?

chipo.groupby(by="item_name").sum().sort_values(by="quantity",ascending=False).head(1)
#choice_description sütununda en çok sipariş edilen öğe neydi?

chipo.groupby(by="choice_description").sum().sort_values(by="quantity",ascending=False).head(1)
#Toplam kaç ürün sipariş edildi?

chipo.quantity.sum()
#Ürün fiyat tipini kontrol edin 

chipo["item_price"].dtype
#Bir lambda fonks oluşturun ve ürün fiyatının türünü değiştirin 

#type(float(chipo["item_price"][1][1:-1]))

fonksiyon = lambda x:float(x[1:-1])

chipo["item_price"]=chipo["item_price"].apply(fonksiyon)
chipo["item_price"].dtype
chipo.head()
#Veri kümesindeki dönemin geliri ne kadardı?

(chipo["quantity"]*chipo["item_price"]).sum()
#Dönem içinde kaç order(sipariş) verildi? 

chipo["order_id"].value_counts().count()
#Sipariş başına ortalama gelir miktarı nedir?

chipo["gelir"] = chipo["quantity"]*chipo["item_price"]

chipo.groupby(by="order_id").sum().mean()["gelir"]
#Kaç farklı ürün satılıyor?

chipo.item_name.value_counts().count()
#users degişkene atıp index column'unu "user_id" olarak ayarlayın

url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user"

users = pd.read_csv(url,sep="|")
#ilk 25 veriye bakın?

users.head(25)
#Son 10 veriye bakın?

users.tail(10)
#Veri kümesindeki gözlem sayısı nedir? 

users.shape
users.info()
#Veri kümesindeki sütun sayısı nedir?

len(users.columns)
#Tüm sütunların adını yazdırın.

users.columns
users.index
#Her bir sütunun veri türü nedir?

users.dtypes
#Yalnızca meslek sütununu yazdırın

users["occupation"].head()
#Bu veri kümesinde kaç farklı meslek var? 

users["occupation"].value_counts().count()
##Bu veri kümesinde kaç farklı meslek var? 

users.occupation.nunique()
#En sık karşılaşılan meslek nedir? 

users["occupation"].value_counts(ascending=False).head(1)
#DataFrame'i özetleyin

users.describe()
#Tüm sütunları özetleyin

users.describe(include="all")
#Yalnızca meslek sütununu özetleyin 

users.occupation.describe()
users.head()
#Kullanıcıların ortalama yaşı nedir? 

users.age.mean()
#En düşük yaş nedir? 

users.age.value_counts().tail()
chipo.head()
# kaç ürün maliyeti 10.00 $' den fazla'?

chipo[chipo.item_price >10].nunique()
#her öğenin fiyatı nedir? 

cols = ["item_name","item_price"]

chipo[cols].head()
#Öğenin adına göre sıralayın

chipo.sort_values(by="item_name").head()
#Sipariş edilen en pahalı ürünün miktarı neydi?

chipo.sort_values(by="item_price",ascending=False).head()
#Bir Veggle Salad Bowl kaç kez sipariş edildi?

len(chipo[chipo["item_name"]=="Vaggle Salad Bowl"])
url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv"

euro12=pd.read_csv(url)
euro12.head()
#Yalnızca Goal sütununu seçin

euro12.Goals
#Euro2012'ye kaç takım katıldı?

euro12.Team.count()
#Veri kümesindeki sütun sayısı nedir? 

len(euro12.columns)
#Görünümü yalnızca sütunları Takım, Sarı Kartlar ve Kırmızı Kartları ve disiplin denilen bir dataframe ata
cols = ["Team","Yellow Cards","Red Cards"]

disipline = euro12[cols]

disipline
#Kırmızı ve sarı kartlara göre sırala

disipline.sort_values(by=["Red Cards","Yellow Cards"],ascending=False)
#Takım başına verilen ortalama Sarı Kartı hesaplayın

disipline["Yellow Cards"].mean()
euro12.head()
#6'dan fazla gol atan takımları yazdır

euro12[euro12["Goals"] > 6]
#G  ile başlayan takımları seçin

euro12[euro12["Team"].str.startswith('G')]
#İlk 7 sütunu seçin

euro12.iloc[:,:7]
# Son 3 dışındaki tüm sütunları seçin.

euro12.iloc[:,:-3]
#Yalnızca İngiltere, İtalya ve Rusya'dan Shooting Accuracy göster

teams = ["England","Italy","Russia"]

euro12[euro12["Team"].isin(teams)]["Shooting Accuracy"]
#drink adı verilen bir değişkene atayın

url= "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv"

drink=pd.read_csv(url)
drink.head()
# Hangi kıta ortalama olarak daha fazla bira içer?

drink.groupby(by="continent")["beer_servings"].mean()
#Her kıta için şarap tüketimine ilişkin istatistikleri yazdırın. 

drink.groupby(by="continent").wine_servings.describe()
#Her sütun için kıta başına ortalama alkol tüketimini yazdırın 

drink.groupby(by="continent").mean()
#Alkol tüketimi için ortalama, minimum ve maksimum değerleri yazdırın. 

drink.groupby(by="continent").spirit_servings.agg(["mean","min","max"])
#Veri kümesini bu adresten içe aktarın . 

#users adında bir değişkene atayın.

#https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user

url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user"

users=pd.read_csv(url,sep="|")
users.head()
#Meslek başına ortalama yaşın ne olduğunu göster

users.groupby(by="occupation").age.mean()
# Meslek başına Erkek oranını keşfedin ve bunu en yüksekten en düşüğe doğru sıralayın

male=users[users["gender"]== "M"].groupby(by="occupation").count()

female=users[users["gender"]== "F"].groupby(by="occupation").count()
(male["gender"] / (male["gender"] + female["gender"])).sort_values(ascending=False)
#Her meslek için minimum ve maksimum yaşları hesaplayın

users.groupby(by="occupation").agg(["min","max"]).age
#Her meslek ve cinsiyet kombinasyonu için ortalama yaşı hesaplayın 

users.groupby(by=["occupation","gender"]).mean().age.to_frame()
# https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv

# Veriyi df adlı degişkene ata

url="https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv"

df = pd.read_csv(url)
df.head()
#Bu alıştırmanın amacı için, veri çerçevesini 'scholl' "guardian" sütununa kadar dilimleyin

df2=df.loc[:,"school":"guardian"]

df2.head()

#Dizeleri büyük harfe çevirecek bir lambda işlevi oluşturun (büyük harfle başla)

capitalize = lambda x: x.capitalize()

df2["Fjob"] = df2["Fjob"].apply(capitalize)

df2["Mjob"] = df2["Mjob"].apply(capitalize)
#Veri kümesinin son öğelerini yazdırın.

df2.tail()
#Alkol kullanabilirligini True False ile göster fonks yaz

def icebilir(x):

    if x>13:

        return True

    else:

        return False

    

    
df2["icebilir"]=df2["age"].apply(icebilir)
df2.head()
#Her birini cars1 ve cars2 adlı bir değişkene atayın

cars1=pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars1.csv")

cars2=pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars2.csv")
cars2.head()
#Hata, ilk veri kümemizde adlandırılmamış boş sütunlar var düzelt 

cars1.head()

cars1 = cars1.loc[:,"mpg":"car"]

cars1.head()
cars2.head()
#Her bir veri kümesindeki gözlem sayısı nedir

cars1.shape
cars2.shape
cars = pd.concat([cars1,cars2])
cars
#. Hata! Sahip adı verilen bir sütun eksik. Rastgele bir sayı oluşturun 15 binden 73 bine seri oluştur

import numpy as np
owners = np.random.randint(low=15000,high=73001,size=398)
owners
cars["owners"]= owners
cars