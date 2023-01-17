# Anomali Kayıtların Olduğu Veri İle Yapılan Çalışma
# Kapsamlı (Düzeltmeler Yapılmış.) 
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
train_y = train.AL_SAT # Öğrenme için hedef değişkenin belirtilmesi 
test_y = test.AL_SAT # Test için hedef değişkenin belirtilmesi 
train_x = train.drop("AL_SAT", axis = 1) # Öğrenme verisinde sadece açıklayıcı değişkenlerin bulundurulması
test_x = test.drop("AL_SAT", axis = 1) # Test verisinde sadece açıklayıcı değişkenlerin bulundurulması
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
# Model başarımı açısından en iyi olanın SVM algoritması modeli olduğu görülmüştür. SVM Algoritması Doğruluk Oranı : 0.5875
# SVM algoritması modeli test verisinde SAT (-1) etiketli 319 kaydın 208 adedini doğru bilmiştir.
# SVM algoritması modeli test verisinde AL (+1) etiketli 401 kaydın 215 adedini doğru bilmiştir.