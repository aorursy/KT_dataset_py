# Anomali Kayıtların Çıkarılarak Yapılan Çalışma
import numpy as np # numpy kütüphanesinin yüklenmesi 
import pandas as pd # pandas kütüphanesinin yüklenmesi 
import seaborn as sb # seaborn kütüphanesinin yüklenmesi 
import matplotlib.pyplot as plt # matplotlib kütüphanesinin yüklenmesi 
import pylab as pl # pylab kütüphanesinin yüklenmesi

# Verilerin Sisteme Yüklenmesi ve Gösterimi
data = pd.read_csv("../input/data_final.csv") # Verinin import edilmesi 
data.head(5) # Veride ilk 5 satırın gösterimi
# Verinin Öğrenme ve Test Etme Diye 2'ye Ayrılması 
train = data[:1560] # İlk 1560 kayıt öğrenme için kullanılacak.
test = data[1560:] # Sonraki 720 kayıt test için kullanılacak.
# Anomali Kayıtların KMeans Kümeleme Yaklaşımıyla Testipi (Öğrenme Verisi)
from sklearn.cluster import KMeans 
kmeans = KMeans(n_clusters=8, random_state=42,n_jobs=-1).fit(train) # Küme sayısı 8 olarak girilmiştir.
X_train_clusters=kmeans.predict(train) 
X_train_clusters_centers=kmeans.cluster_centers_ # Küme merkezlerinin belirlenmesi
dist = [np.linalg.norm(x-y) for x,y in zip(train.values,X_train_clusters_centers[X_train_clusters])] 
km_y_pred=np.array(dist)
km_y_pred[dist>=np.percentile(dist,90)]=1 # Küme merkezlerinden uzak olan kayıtlara 1 atanması (Anomali) (Bu kayıtlar sonradan çıkarılacak.)
km_y_pred[dist<np.percentile(dist,90)]=0 # Küme merkezlerine yakın olan kayıtlara 0 atanması (Normal) (Bu kayıtlar sınıflandırmada kullanılacak.)
train['Anomali_Etiket'] = km_y_pred # 0 ve 1 atanan etiketli sütunun Anomali_Etiket ismi ile train datasına eklenmesi
Ogrenme_Anomali_Kayıtlar = train.Anomali_Etiket > 0 # Anomali kayıtların filtrelenmesi
Ogrenme_Anomali_Kayıtlar = train[Ogrenme_Anomali_Kayıtlar]
print("Ögrenme Verisindeki Anomali Kayıt Sayısı")
len(Ogrenme_Anomali_Kayıtlar)
Ogrenme_Normal_Kayıtlar = train.Anomali_Etiket < 1 # Normal kayıtların filtrelenmesi
Ogrenme_Normal_Kayıtlar = train[Ogrenme_Normal_Kayıtlar]
print("Ögrenme Verisindeki Normal Olan Kayıt Sayısı")
len(Ogrenme_Normal_Kayıtlar)
train = Ogrenme_Normal_Kayıtlar.drop("Anomali_Etiket",axis = 1) # Anomali_Etiket adlı sütununun sınıflandırmada kullanılacak veriden çıkarılması
train_y = train.AL_SAT # Öğrenme verisi için hedef değişkenin tayini
train_x = train.drop("AL_SAT",axis = 1) # Öğrenme verisi için sadece açıklayıcı değişkenlerin olduğu bir tablonun oluşturulması
train_x.head(3) 
# Anomali Kayıtların KMeans Kümeleme Yaklaşımıyla Testipi (Test Verisi)
kmeans = KMeans(n_clusters=8, random_state=42,n_jobs=-1).fit(test)
X_test_clusters=kmeans.predict(test)
X_test_clusters_centers=kmeans.cluster_centers_
dist = [np.linalg.norm(x-y) for x,y in zip(test.values,X_test_clusters_centers[X_test_clusters])]
km_y_pred=np.array(dist)
km_y_pred[dist>=np.percentile(dist,90)]=1
km_y_pred[dist<np.percentile(dist,90)]=0
test['Anomali_Etiket'] = km_y_pred # 0 ve 1 atanan etiketli sütunun Anomali_Etiket ismi ile train datasına eklenmesi
Test_Anomali_Kayıtlar = test.Anomali_Etiket > 0 # Anomali kayıtların filtrelenmesi
Test_Anomali_Kayıtlar = test[Test_Anomali_Kayıtlar]
print("Test Verisindeki Anomali Kayıt Sayısı")
len(Test_Anomali_Kayıtlar)
Test_Normal_Kayıtlar = test.Anomali_Etiket < 1 # Normal kayıtların filtrelenmesi
Test_Normal_Kayıtlar = test[Test_Normal_Kayıtlar]
print("Test Verisindeki Normal Olan Kayıt Sayısı")
len(Test_Normal_Kayıtlar)
test = Test_Normal_Kayıtlar.drop("Anomali_Etiket",axis = 1) # Anomali_Etiket adlı sütununun sınıflandırmada kullanılacak veriden çıkarılması
test_y = test.AL_SAT # Test verisi için hedef değişkenin tayini
test_x = test.drop("AL_SAT",axis = 1) # Test verisi için sadece açıklayıcı değişkenlerin olduğu bir tablonun oluşturulması
test_x.head(3) 
# Rastgele Orman Modeli
from sklearn.ensemble import RandomForestClassifier
karar_agaci = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=2, max_features='auto').fit(train_x,train_y) # Rastgele orman algoritması ile öğrenme sürecine başlanması 
predict_y1 = karar_agaci.predict(test_x) # Test datası üzerinde modelin (pattern) test edilmesi
# Rastgele Orman Modelinin Doğruluk Oranının Tespiti
from sklearn import metrics
print("Rastgele Orman Modeli Doğruluk Oranı :",metrics.accuracy_score(test_y, predict_y1)) # Test verisi üzerinde tahmin gücünün tespit edilmesi 
# Rastgele Orman Modeli Metrikleri
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y, predict_y1)) # Karmaşıklık matrisinin elde edilmesi  
print(classification_report(test_y, predict_y1)) # Recall , precision vb. değerlerin elde edilmesi
# Lojistik Regresyon Modeli
from sklearn.linear_model import LogisticRegression  # sklearn kütüphanesinin (Logistic Regression) yüklenmesi 
logreg = LogisticRegression(random_state=0,solver = 'lbfgs', multi_class='auto').fit(train_x,train_y) # Lojistik regresyon ile öğrenme sürecine başlanması 
predict_y2 = logreg.predict(test_x) # Test datası üzerinde modelin (pattern) test edilmesi 
# Lojistik Regresyon Modelinin Doğruluk Oranının Tespiti
from sklearn import metrics
print("Lojistk Regresyon Doğruluk Oranı :",metrics.accuracy_score(test_y, predict_y2)) # Test verisi üzerinde tahmin gücünün tespit edilmesi
#  Lojistik Regresyon Modeli Metrikleri
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y, predict_y2)) # Karmaşıklık matrisinin elde edilmesi  
print(classification_report(test_y, predict_y2)) # Recall , precision vb. değerlerin elde edilmesi 
# En Yakın K Komşu Algoritması Modeli
from sklearn.neighbors import KNeighborsClassifier # sklearn kütüphanesinin (KNN) yüklenmesi 
KNN = KNeighborsClassifier(n_neighbors=3).fit(train_x,train_y) # KNN algoritması ile öğrenme sürecine başlanması 
predict_y3 = KNN.predict(test_x) # Test datası üzerinde modelin (pattern) test edilmesi 
# KNN Algoritması Modelinin Doğruluk Oranının Tespiti
from sklearn import metrics
print("KNN Algoritması Doğruluk Oranı :",metrics.accuracy_score(test_y, predict_y3)) # Test verisi üzerinde tahmin gücünün tespit edilmesi
#  KNN Algoritması Modelinin Metrikleri
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y, predict_y3))  # Karmaşıklık matrisinin elde edilmesi  
print(classification_report(test_y, predict_y3)) # Recall , precision vb. değerlerin elde edilmesi 
# SVM Algoritması Modeli
from sklearn.svm import SVC  # sklearn kütüphanesinin (SVC) yüklenmesi 
SVM = SVC(C=0.1, gamma = 0.5 , kernel='linear').fit(train_x,train_y) # SVM algoritması ile öğrenme sürecine başlanması 
predict_y4 = SVM.predict(test_x) # Test datası üzerinde modelin (pattern) test edilmesi 
# SVM Algoritması Modelinin Doğruluk Oranının Tespiti
from sklearn import metrics
print("SVM Algoritması Doğruluk Oranı :",metrics.accuracy_score(test_y, predict_y4)) # Test verisi üzerinde tahmin gücünün tespit edilmesi
# SVM Algoritması Modelinin Metrikleri
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y, predict_y4)) # Karmaşıklık matrisinin elde edilmesi  
print(classification_report(test_y, predict_y4)) # Recall , precision vb. değerlerin elde edilmesi 
# Model başarımı açısından en iyi olanın Rastgele Orman Modeli olduğu görülmüştür. Rastgele Orman Modeli Doğruluk Oranı : 0.6188271604938271
# Rastgele Orman Modeli test verisinde SAT (-1) etiketli 283 kaydın 117 adedini doğru bilmiştir.
# Rastgele Orman Modeli test verisinde AL (+1) etiketli 365 kaydın 284 adedini doğru bilmiştir.
# Anomali kayıtların veriden çıkarılması tahmin gücünü artırmıştır.