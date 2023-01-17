# KÜtüphaneleri import ediyoruz.

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


#Veri setimizi okutuyoruz.
data = pd.read_csv('../input/wisc_bc_data.csv')
data = data.drop('id',axis=1)
#Describe metodu ile matematiksel değerlerimizi inceliyoruz
data.describe()
data.info()
# Datalarımızı türlerine bakıyoruz.
# Kanserin iyi yada kötü huylu olmasını burada diagnosis değeri belirliyor. Bu değer  'B'(Benign) iyi huylu ve 'M'(Malignant) kötü huylu olmak üzere iki değer alıyor.
data.head()
#Datamızın ilk on değerini inceliyoruz. Arema_mean,compactness_mean,radius_se,area_se featureları çok uç noktalarda değerler alabilmektedir.
data.tail()
data.shape
# Datamız 569 satırdan oluşan değerlerden oluşurken , 31 tanede featuredan oluşmaktadır.
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools

M = data.radius_mean[data.diagnosis == 'M']
B = data.radius_mean[data.diagnosis == 'B']

trace1 = go.Histogram(
    x=M,
    opacity=0.75,
    name = "Radius mean değerleri",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
trace2 = go.Histogram(
    x=B,
    opacity=0.75,
    name = "Kanser Durumu",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2]
layout = go.Layout(barmode='overlay',
                   title=' students-staff ratio in 2011 and 2012',
                   xaxis=dict(title='students-staff ratio'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
data.corr()

plt.figure(figsize=(20,20))
sns.heatmap(data[data.columns[0:]].corr(),annot=True)
#Burada korelasyon grafiğimizi heatmap ile göstermeye çalıştık.
#Bir çok parametremiz diğer parametreler ile büyük korelasyonlara sahip buda verimizi eğitip modelimizden güzel sonuçlar almamızı sağlayacağını düşünüyorum.

fig, axes = plt.subplots(2,2, figsize = (16,8), sharex=False, sharey=False)
sns.boxplot(y='radius_mean',data=data, ax=axes[0,0])
sns.boxplot(y='area_worst',data=data, ax=axes[0,1])
sns.boxplot(y='radius_worst',data=data, ax=axes[1,0])
sns.boxplot(y='concave points_worst',data=data, ax=axes[1,1])
plt.tight_layout()

#Datamızın ilk 4 featurenın  uç değerlerini inceliyoruz.
#Radius mean  ve area worst çok uç değerlere sahipken  concave points worst ortalama değerlere sahiptir.

#Object türündeki verimizi int tipine çevirdik
data.diagnosis = [1 if each == "B" else 0 for each in data.diagnosis] 
data.info()

#Diagnosis featuremızı verimizden ayırıyoruz
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)
#Normalizasyon yapıyoruz
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#Verilerimizi test ve train olmak üzere ayırıyoruz
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.15,random_state=5)
#Logistic Regression modelimize verimizi uyguluyoruz
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print("Test Doğruluğu(Accuracy) %{}".format(lr.score(x_test,y_test)*100))

y_pred = lr.predict(x_test)
y_true = y_test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)

#Seaborn kullanarak confusion matriximizi heatmap aracıyla görselleştiriyoruz.
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
predictions = lr.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

# %97.67 lik büyük bir doğruluk oranı aldık
# Confusion Matrix'e baktığımız zaman iyi huylu tümörlerde %100 başarı saklarken kötü huylu tümörlerde başarı oranımızın iyiye göre bir tık kötü olduğunu görüyoruz
#KNN modeline verimizi uyguluyoruz

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(3)
knn.fit(x_train,y_train)
print("Test Doğruluğu(Accuracy) %{}".format(knn.score(x_test,y_test)*100))

y_pred = knn.predict(x_test)
y_true = y_test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)

#Seaborn kullanarak confusion matriximizi heatmap aracıyla görselleştiriyoruz.
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
predictions = knn.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

# %96.51 lik büyük bir doğruluk oranı aldık
# Confusion Matrix'e baktığımız zaman iyi huylu tümörlerde %100 başarı saklarken kötü huylu tümörlerde başarı oranımızın LR modelimize göre bir tık kötü çalıştığını söyleyebiliriz.
# SVC modelimize verimizi uyguluyoruz
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
print("Test Doğruluğu(Accuracy) %{}".format(svc.score(x_test,y_test)*100))


y_pred = svc.predict(x_test)
y_true = y_test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)

#Seaborn kullanarak confusion matriximizi heatmap aracıyla görselleştiriyoruz.

import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
predictions = svc.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

# %96.51 lik büyük bir doğruluk oranı aldık
# Confusion Matrix'e baktığımız zaman KNN algoritması ile aynı sonuçları almış olduğunu görüyoruz.
# 3 Model ile test ettiğimiz veri setimizde en iyi sonucu Logistic Regression algoritmamız  ile aldık.
# Aldığımız sonuç çok iyi bir sonuç olup veri setimizi büyütüp modelimizi daha iyi eğitebilir ve daha yüksek sonuçlar alabiliriz.
# İyi huylu tümörlerin tespit edilmesinde çok iyiyken kötü huylu tümörleri tespit ederken model daha  da iyileştirilebilir.