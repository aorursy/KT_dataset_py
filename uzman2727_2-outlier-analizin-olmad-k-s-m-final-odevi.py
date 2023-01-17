# Anomali Kayıtları Çıkarmadan Yapılan Çalışma
import numpy as np # numpy kütüphanesinin yüklenmesi 
import pandas as pd # pandas kütüphanesinin yüklenmesi 
import seaborn as sb # seaborn kütüphanesinin yüklenmesi 
import matplotlib.pyplot as plt # matplotlib kütüphanesinin yüklenmesi 
import pylab as pl # pylab kütüphanesinin yüklenmesi 

# Verilerin Sisteme Yüklenmesi ve Gösterimi
data = pd.read_csv("../input/Data_BH_Final.csv") # Verinin import edilmesi 
data.head(5) # Veride ilk 5 satırın gösterimi 
ax = sb.boxplot(x="AL_SAT", y="Talep1", data=data)
ax = sb.swarmplot(x="AL_SAT", y="Talep1", data=data, color=".25") # SAT etiketli kayıtların ortalaması (Talep1) AL etiketli kayıtların ortalamasından (Talep1) büyüktür. 
ax = sb.boxplot(x="AL_SAT", y="Fiyat1", data=data)
ax = sb.swarmplot(x="AL_SAT", y="Fiyat1", data=data, color=".25") # AL etiketli kayıtların ortalaması (Fiyat1) SAT etiketli kayıtların ortalamasından (Fiyat1) büyüktür.
ax = sb.boxplot(x="AL_SAT", y="DengeFiyat1", data=data)
ax = sb.swarmplot(x="AL_SAT", y="DengeFiyat1", data=data, color=".25") # AL etiketli kayıtların ortalaması (DengeFiyat1) SAT etiketli kayıtların ortalamasından (DengeFiyat1) büyüktür.
ax = sb.boxplot(x="AL_SAT", y="Pozisyon", data=data)
ax = sb.swarmplot(x="AL_SAT", y="Pozisyon", data=data, color=".25")
# Verinin Öğrenme ve Test Verisi Şeklinde Ayrılması
train = data[:1560] # İlk 1560 kayıt öğrenme için kullanılacak.
test = data[1560:] # Sonraki 720 kayıt test için kullanılacak.
train_y = train.AL_SAT # Öğrenme için hedef değişkenin belirtilmesi 
test_y = test.AL_SAT # Test için hedef değişkenin belirtilmesi 
train_x = train.drop("AL_SAT", axis = 1) # Öğrenme verisinde sadece açıklayıcı değişkenlerin bulundurulması
test_x = test.drop("AL_SAT", axis = 1) # Test verisinde sadece açıklayıcı değişkenlerin bulundurulması
# Karar Ağacı Modeli
from sklearn.tree import DecisionTreeClassifier # sklearn kütüphanesinin  (Decision Tree Classifier) yüklenmesi 
karar_agaci = DecisionTreeClassifier().fit(train_x,train_y) # Karar ağacı ile öğrenme sürecine başlanması 
predict_y1 = karar_agaci.predict(test_x) # Test datası üzerinde modelin (pattern) test edilmesi 
# Karar Ağacı Modelinin Doğruluk Oranının Tespiti
from sklearn import metrics
print("Karar Ağacı Doğruluk Oranı :",metrics.accuracy_score(test_y, predict_y1)) # Test verisi üzerinde tahmin gücünün tespit edilmesi  
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y, predict_y1)) # Karmaşıklık matrisinin elde edilmesi  
print(classification_report(test_y, predict_y1)) # Recall , precision vb. değerlerin elde edilmesi  
# Lojistik Regresyon Modeli
from sklearn.linear_model import LogisticRegression  # sklearn kütüphanesinin (Logistic Regression) yüklenmesi 
logreg = LogisticRegression().fit(train_x,train_y) # Lojistik regresyon ile öğrenme sürecine başlanması 
predict_y2 = logreg.predict(test_x) # Test datası üzerinde modelin (pattern) test edilmesi 
# Lojistik Regresyon Modelinin Doğruluk Oranının Tespiti
from sklearn import metrics
print("Lojistk Regresyon Doğruluk Oranı :",metrics.accuracy_score(test_y, predict_y2)) # Test verisi üzerinde tahmin gücünün tespit edilmesi
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y, predict_y2)) # Karmaşıklık matrisinin elde edilmesi  
print(classification_report(test_y, predict_y2)) # Recall , precision vb. değerlerin elde edilmesi 
# En Yakın K Komşu Algoritması Modeli
from sklearn.neighbors import KNeighborsClassifier # sklearn kütüphanesinin (KNN) yüklenmesi 
KNN = KNeighborsClassifier().fit(train_x,train_y) # KNN algoritması ile öğrenme sürecine başlanması 
predict_y3 = KNN.predict(test_x) # Test datası üzerinde modelin (pattern) test edilmesi 
# KNN Algoritması Modelinin Doğruluk Oranının Tespiti
from sklearn import metrics
print("KNN Algoritması Doğruluk Oranı :",metrics.accuracy_score(test_y, predict_y3)) # Test verisi üzerinde tahmin gücünün tespit edilmesi
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y, predict_y3))  # Karmaşıklık matrisinin elde edilmesi  
print(classification_report(test_y, predict_y3)) # Recall , precision vb. değerlerin elde edilmesi 
# SVM Algoritması Modeli
from sklearn.svm import SVC  # sklearn kütüphanesinin (SVC) yüklenmesi 
SVM = SVC(C=1.0, gamma = 0.5 , kernel='rbf',degree=2).fit(train_x,train_y) # SVM algoritması ile öğrenme sürecine başlanması 
predict_y4 = SVM.predict(test_x) # Test datası üzerinde modelin (pattern) test edilmesi 
# SVM Algoritması Modelinin Doğruluk Oranının Tespiti
from sklearn import metrics
print("SVM Algoritması Doğruluk Oranı :",metrics.accuracy_score(test_y, predict_y4)) # Test verisi üzerinde tahmin gücünün tespit edilmesi
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y, predict_y4)) # Karmaşıklık matrisinin elde edilmesi  
print(classification_report(test_y, predict_y4)) # Recall , precision vb. değerlerin elde edilmesi 
# Model başarımı açısından en iyi olanın Lojistik Regresyon modeli olduğu görülmüştür.
# Lojistik Regresyon modeli test verisinde SAT (-1) etiketli 319 kaydın 233 adedini doğru bilmiştir.
# Lojistik Regresyon modeli test verisinde AL (+1) etiketli 401 kaydın 185 adedini doğru bilmiştir.
# SVM'de parametrelerin (C ve gamma)  değiştirilmesi SVM'in model başarımını etkileyebilir.Method için 'rbf' yerine 'linear' olmasıda model başarımını değiştirecektir.
