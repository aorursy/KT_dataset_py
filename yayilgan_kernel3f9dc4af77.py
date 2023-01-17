import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt
tablet=pd.read_csv('tablet.csv')

df=tablet.copy()
df.head() #ilk 5 gözlem
df.tail() #son 5 gözlem
df.sample(5) # rastgele 5 satır 
df.shape # boyut
df.info() #Özniteliklerin veri türleri, 

#içerdikleri kayıt sayıları ve bellek kullanımı hakkında bilgi edinme
#Korelasyon Gösterim

import seaborn as sns

corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
df.describe().T #ortalama, medyan, standart sapma  değerleri
# veride kaçtane NAN değeri var?

df.isnull().sum().sum()
# veride hangi sütunda kaçtane NAN değeri var?

df.isnull().sum()
#eksik değerlerin % oranlarını aşağıdaki fonksiyon ile bulalım

def eksik_deger_tablosu(df): 

    eksik_deger = df.isnull().sum()

    eksik_deger_yuzde = 100 * df.isnull().sum()/len(df)

    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)

    eksik_deger_tablo_son = eksik_deger_tablo.rename(

    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})

    return eksik_deger_tablo_son

  

eksik_deger_tablosu(df)

# ortalama ile doldurulması daha mantıklı zaten sayıca az olduklarından dolayı verinin çoğunluğunu değiştirme gibi bir büyüklük söz konusu değil 

df['RAM'] = df['RAM'].fillna(df['RAM'].mean())

df['OnKameraMP'] = df['OnKameraMP'].fillna(df['OnKameraMP'].mean())

# tekrar eksik değerler bir göz atalım ve Dolu olduğunu göreceğiz

df.isnull().sum()
#kategorik değişkenlere bakmak için tekrara çağıralım 

df.head()
#  aşağıdaki öznitelikler için de kategorik değerden sayısal değere dönüşüm

from sklearn import preprocessing

df['Bluetooth']= label_encoder.fit_transform(df['Bluetooth'])

df['CiftHat']= label_encoder.fit_transform(df['CiftHat'])

df['4G']= label_encoder.fit_transform(df['4G'])

df['3G']= label_encoder.fit_transform(df['3G'])

df['Dokunmatik']= label_encoder.fit_transform(df['Dokunmatik'])

df['WiFi']= label_encoder.fit_transform(df['WiFi'])

df['FiyatAraligi']= label_encoder.fit_transform(df['FiyatAraligi'])

df['Renk']= label_encoder.fit_transform(df['Renk'])

df