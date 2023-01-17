# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/unsupervised.csv",index_col =0 )
df.head()
df.isnull().sum

#Eksik gözlem var mı kontrol edelim
df.info()
df.describe().T


df.hist(figsize=(10,10));

#Dagılımları grafikle göstermek istiyoruz , figsize (10, 10) dedik yani oluşturlan grafigin boyutunu belirtiyoruz
kmeans = KMeans(n_clusters=4)

#4 Clasdan olşacak küme nesnesi oluşturduk
kmeans
k_fit=kmeans.fit(df)
k_fit.n_clusters
dir(k_fit)
k_fit.cluster_centers_

#4 farklı kümenin merkezinde yer alan gözlem birimlerimiz
k_fit.labels_

#gözlem birimlerinin hangi classlara ait oldugu bilgileri
#Bunun için 2 adet degişken seçicez çünkü 2 degişken üzerinde kümelenmeyi göstericez 
k_means = KMeans(n_clusters=2).fit(df)
kumeler=k_means.labels_
kumeler
#iloc bagımsız şekilde satır ve sutunlardan seçim yapma

#loc bagımlı şekilde isimlerden seçim yapma  

plt.scatter(df.iloc[:,0],df.iloc[:,1],c=kumeler,s=50 , cmap="viridis")
#Veri setini 2 kümeye ayırdık

#merkezlerin bu kümelere işaretliyelim
merkezler = k_means.cluster_centers_
merkezler
plt.scatter(merkezler[:,0],merkezler[:,1], c="black", s=200,alpha=0.5)
plt.scatter(df.iloc[:,0],df.iloc[:,1],c=kumeler,s=50 , cmap="viridis")

plt.scatter(merkezler[:,0],merkezler[:,1], c="black", s=200,alpha=0.5)
#50 adet eyaleti 2 kümelere ayırdık neye göre ayrıldı 

#2 degişken göz önünde bulunduruldugunda gözlem birimlerinin bu merkezlere olan uzaklara göre belirledik
df


ssd = []

K=range(1,30)

for k in K:

    kmeans=KMeans(n_clusters=k).fit(df)

    ssd.append(kmeans.inertia_)
plt.plot(K,ssd,"bx-")

plt.xlabel("Farklı K Değerlerine Karşılık Uzaklık Artık Toplamları")

plt.title("Optimum Küme Sayisi İçin Elbow Yöntemi")
from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans()

visu = KElbowVisualizer(kmeans , k = (2,20))

visu.fit(df)

visu.poof()
kmeans_Final = KMeans(n_clusters=4).fit(df)
kumeler=kmeans_Final.labels_
kumeler
pd.DataFrame({"Eyaletler": df.index , "Kümeler": kumeler})
df["Küme_No"]= kumeler
df
from scipy.cluster.hierarchy import linkage
hc_complete = linkage(df,"complete")

hc_average = linkage(df,"average")

#bu nesnleri dendogram oluşturmak için kulalnıcaz dendogram isehiyerarşik yapıyı ifade eden görselin adı
from scipy.cluster.hierarchy import dendrogram
plt.figure(figsize= (15,10))

plt.title("Hiyerarşik Kümeleme Dendogramı")

plt.xlabel("Gözlem Birimleri")

plt.ylabel("Hesaplanan Uzaklıklar")

dendrogram(hc_complete,leaf_font_size=10);



# Görsel oluşturuyoruz.figure= çıkacak olan görselin boyutunu ayarladık 

#dendrogram leaf_font_size=oluşacak olan grafigin x eksenindeki boyutunu ifade eder

# ; koyarsanız show ile aynı yapabilirsiniz
plt.figure(figsize= (15,10))

plt.title("Hiyerarşik Kümeleme Dendogramı")

plt.xlabel("Gözlem Birimleri")

plt.ylabel("Hesaplanan Uzaklıklar")

dendrogram(hc_complete,truncate_mode="lastp",p=4,show_contracted=True,leaf_font_size=10);

#truncate = lastp en son p adet göster p =4 dedik 

#show= kümeleme yapınca kaçar adet eleman oldugu getirmesini istiyoruz
plt.figure(figsize= (10,5))

plt.title("Hiyerarşik Kümeleme Dendogramı")

plt.xlabel("Gözlem Birimleri")

plt.ylabel("Hesaplanan Uzaklıklar")

dendrogram(hc_average,leaf_font_size=10);

df1 = pd.read_csv("../input/hidders/hittesr13.csv")

df1.dropna(inplace=True) #veri setinde eksik bilgileri sil

df1 = df1._get_numeric_data() #sayısal degişkenleri seçiyorum

df1.head()
from sklearn.preprocessing import StandardScaler
df1 = StandardScaler().fit_transform(df1)
df1[0:5,0:5]
from sklearn.decomposition import PCA
pca = PCA(n_components =2)

pca_fit=pca.fit_transform(df1)
bilesen_df1=pd.DataFrame(data=pca_fit , columns =["Birinci_Bileşen","İkinci_Bileşen"])
bilesen_df1

#Elimizdeki taşımış oldugu varyansı 2 adet deigşken ile temsil ettik kayıpları göze alarak gerçekleştirdik
pca.explained_variance_ratio_

#Birinci bileşence içinde buldugu degişkenligin %45 ve ikinci bileşence %24 açıkalabilmiş
pca.components_[1]

#OPTİMUM BİLEŞEN SAYISI

pca =PCA().fit(df1)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel("Bileşen Sayisi")

plt.ylabel("Kümülatif Varyans Oranı")

# biz burada eger bileşen sayısı belirtmezsek degişken sayısı kadar bileşen oluşur 

#eger degişken sayısı kadar oluşan bileşenlerin toplamlarını alırsak veri setinin ne kadar açıklanabildigini görürüz

#FİNAL

pca_final = PCA(n_components=3)

pca_fit1=pca_final.fit_transform(df1)
pca_final.explained_variance_ratio_
45+24+10
#Yaklaşık %80 ini temsil etmiş olduk