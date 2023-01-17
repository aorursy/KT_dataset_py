# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Data Okuma

df = pd.read_csv("../input/heart.csv")

#İlk 5 Satırı Listeleyelim

df.head()
#Kaç adet hasta var Kaç adet Hasta Yok

df.target.value_counts()
#Görselleştirelim

sns.countplot(x="target",data=df,palette="bwr")

plt.show()
hastaolmayanlar=len(df[df.target==0])

hastaolanlar=len(df[df.target==1])

print("Kalp Hastalığı Olmayan Hastaların Yüzdesi: {:.2f}% ".format((hastaolmayanlar /len(df.target))*100))

print("Kalp Hastalığı Olan Hastaların Yüzdesi: {:.2f}%".format((hastaolanlar/len(df.target==1))*100))
#Cinsiyetleri Görselleştirelim

sns.countplot(x="sex",data=df,palette="mako_r")

plt.xlabel("Cinsiyet(0=Kadın,1=Erkek)")

plt.show()
kadinsayisi=len(df[df.sex==0])

erkeksayisi=len(df[df.sex==1])

print("Kadin Hasta Yüzdesi: {:.2f}%".format((kadinsayisi/len(df.sex))*100))

print("Erkek Hasta Yüzdesi: {:.2f}%".format((erkeksayisi/len(df.sex))*100))
#Hastalıgı diger sutundaki verilerin ortalamasına bakalım

df.groupby('target').mean()
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))

plt.title('Kalp Hastalığı Sıklığı')

plt.xlabel('Yaş')

plt.ylabel('Hastalık Sıklıgı')

plt.savefig('hastaliksikligi.png')

plt.show()
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])

plt.title('Cinsiyete Göre Hastalık Sıklığı')

plt.xlabel('Cinsiyet (0 = Kadın, 1 = Erkek)')

plt.xticks(rotation=0)

plt.legend(["Hastalık Yok', 'Hastalık Var'"])

plt.ylabel('Hastalık Sıklıgı')

plt.show()
a = pd.get_dummies(df['cp'], prefix = "cp")

b = pd.get_dummies(df['thal'], prefix = "thal")

c = pd.get_dummies(df['slope'], prefix = "slope")
frames = [df,a,b,c]

df=pd.concat(frames,axis=1)

df.head()
#Ne yaptık? Şöyle mesela ilk cp sutunumuz var bu sutun gögüs  agrı tipi 0-1-2-3 diye numalandırılmış biz bunu 

#Her gögüs ayrı tipini sutunlara ekleyip numeric degerlere dönderdik



#CP THAL VE SLOPE SUTUNLARINI DF DATA VERİMİZNEN ÇIKARDIK
y = df.target.values

x_data=df.drop(['target'],axis=1)

#y ye target 'in degerlerini x ise target olmayan datayı atadık
x = (x_data-np.min(x_data) / (np.max(x_data)-np.min(x_data))).values

x
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#text size 0.2 dememizin sebebi 

#verilerimizin % 80'i ögrencek ve% 20'si test verisi olacak.
#Transpoz alalım

x_train=x_train.T

y_train=y_train.T

x_test=x_test.T

y_test=y_test.T
from sklearn.linear_model import LinearRegression

    

    
dogruluk = {}



lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

acc=lr.score(x_test.T,y_test.T)*100

dogruluk['Lojistik Regression']=acc

print("Dopruluk Oranı {:.2f}%".format(acc))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2)

knn.fit(x_train.T,y_train.T)

knn_pred = knn.predict(x_test.T)

print("KNN Dopruluk Oranı: {:.2f}%".format(knn.score(x_test.T,y_test.T)*100))

scoreList = []

for i in range(1,20):

    knn2=KNeighborsClassifier(n_neighbors = i)

    knn2.fit(x_train.T,y_train.T)

    scoreList.append(knn2.score(x_test.T,y_test.T))

    

        

plt.plot(range(1,20), scoreList)

plt.xticks(np.arange(1,20,1))

plt.xlabel("n_neigbors")

plt.ylabel("Score Değeri")

plt.show()

acc = max(scoreList)*100

dogruluk['KNN'] = acc

print("Max KNN Doğruluk Oranı {:.2f}%".format(acc))        

from sklearn.svm import SVC
svm = SVC(random_state = 1)

svm.fit(x_train.T,y_train.T)



acc=svm.score(x_test.T,y_test.T)*100

dogruluk['SVM']=acc

print('SVM Algoritması Dogruluk Oranı: {:.2f}%'.format(acc) )
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train.T, y_train.T)



acc = nb.score(x_test.T,y_test.T)*100

dogruluk['Naive Bayes'] = acc

print("Naive Bayes Algoritması Dogruluk Oranı: {:.2f}%".format(acc))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train.T, y_train.T)



acc = dtc.score(x_test.T, y_test.T)*100

dogruluk['Decision Tree'] = acc

print("Decision Tree Algoritması Dogruluk Oranı {:.2f}%".format(acc))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(x_train.T, y_train.T)



acc = rf.score(x_test.T,y_test.T)*100

dogruluk['Random Forest'] = acc

print("Random Forest Algoritması Dogruluk Oranı : {:.2f}%".format(acc))
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]



sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Dogruluk Oranı %")

plt.xlabel("Algoritma")

sns.barplot(x=list(dogruluk.keys()), y=list(dogruluk.values()), palette=colors)

plt.show()