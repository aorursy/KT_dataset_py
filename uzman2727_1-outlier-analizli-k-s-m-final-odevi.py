# Anomali Kayıtların Çıkarılarak Yapılan Çalışma
import numpy as np # numpy kütüphanesinin yüklenmesi 
import pandas as pd # pandas kütüphanesinin yüklenmesi 
import seaborn as sb # seaborn kütüphanesinin yüklenmesi 
import matplotlib.pyplot as plt # matplotlib kütüphanesinin yüklenmesi 
import pylab as pl # pylab kütüphanesinin yüklenmesi
# Verilerin Sisteme Yüklenmesi ve Gösterimi
data = pd.read_csv("../input/data_final.csv") # Verinin import edilmesi 
data.head(5) # Veride ilk 5 satırın gösterimi
data['Pozisyon'].plot.hist(bins = 50,title = "Pozisyon") # Pozisyon Değişkeninin Histogramı. Anomali Bir Durum Yok.Düzgün Bir Dağılım Var.
data['Talep1'].plot.hist(bins = 50,title = "Talep1") # Talep1 Değişkeninin Histogramı . İncelenmeli.
data['Talep2'].plot.hist(bins = 50,title = "Talep2") # Talep2 Değişkeninin Histogramı. İncelenmeli.
data['Talep3'].plot.hist(bins = 50,title = "Talep3") # Talep3 Değişkeninin Histogramı. İncelenmeli.
data['U1'].plot.hist(bins = 50,title = "U1") # U1 Değişkeninin Histogramı. İncelenmeli.
data['U2'].plot.hist(bins = 50,title = "U2") # U2 Değişkeninin Histogramı. İncelenmeli.
data['U3'].plot.hist(bins = 50,title = "U3") # U3 Değişkeninin Histogramı. İncelenmeli.
data['Fiyat1'].plot.hist(bins = 50,title = "Fiyat1") # Fiyat1 Değişkeninin Histogramı. İncelenmeli.
data['Fiyat2'].plot.hist(bins = 50,title = "Fiyat2") # Fiyat2 Değişkeninin Histogramı. İncelenmeli.
data['Fiyat3'].plot.hist(bins = 50,title = "Fiyat3") # Fiyat3 Değişkeninin Histogramı. İncelenmeli.
data['DengeFiyat1'].plot.hist(bins = 50,title = "DengeFiyat1") # DengeFiyat1 Değişkeninin Histogramı. İncelenmeli.
data['DengeFiyat2'].plot.hist(bins = 50,title = "DengeFiyat2") # DengeFiyat2 Değişkeninin Histogramı. İncelenmeli.
data['DengeFiyat3'].plot.hist(bins = 50,title = "DengeFiyat3") # DengeFiyat3 Değişkeninin Histogramı. İncelenmeli.
data['AL_SAT'].plot.hist(bins = 50,title = "AL_SAT") # AL_SAT (hedef) Değişkeninin Histogramı. Anomali Bir Durum Yok.Düzgün Bir Dağılım Var.
# Verinin Öğrenme ve Test Etme Diye 2'ye Ayrılması 
train = data[:1560] # İlk 1560 kayıt öğrenme için kullanılacak.
test = data[1560:] # Sonraki 720 kayıt test için kullanılacak.
# Anomali Kayıtlardan Arındırmak İçin Zscore Yaklaşımı Kullanılacaktır. Anomaly detection için IQR,DBSCAN,Isolation Forest vb. kullanılabilirdi.
from scipy.stats import zscore # scipy kütüphanesinin yüklenmesi 
train["Talep1_Zscore"] = zscore(train["Talep1"]) # Talep1_Zscore değişkeninin türetilmesi ve her bir kayıt için zscore değerinin hesaplanması
train = train.loc[np.logical_and(train.Talep1_Zscore < 2,train.Talep1_Zscore > -2),:] # Cutoff Tayini. Uç Değerlerden Arındırma.
train = train.drop("Talep1_Zscore", axis = 1) # Arındırmadan sonra türetilen Talep1_Zscore değişkeninin silinmesi 
# Her bir aşamada bütün değişkenlerdeki anomali kayıtlar gidene dek işlem tekrarlanır.Zscore için %1 önem seviyesinde data incelendiğinde çok fazla anomali kayıt bulunmaktaydı. 
#  %5 önem seviyesinde anomali kayıt bulunmamaktaydı.   Zscore için %4 önem seviyesi baz alınmıştır.   
train["Talep2_Zscore"] = zscore(train["Talep2"]) # Talep2_Zscore değişkeninin türetilmesi ve her bir kayıt için zscore değerinin hesaplanması
train = train.loc[np.logical_and(train.Talep2_Zscore < 2,train.Talep2_Zscore > -2),:] # Cutoff Tayini. Uç Değerlerden Arındırma.
train = train.drop("Talep2_Zscore", axis = 1) # Arındırmadan sonra türetilen Talep2_Zscore değişkeninin silinmesi 
train["Talep3_Zscore"] = zscore(train["Talep3"]) # Talep3_Zscore değişkeninin türetilmesi ve her bir kayıt için zscore değerinin hesaplanması
train = train.loc[np.logical_and(train.Talep3_Zscore < 2,train.Talep3_Zscore > -2),:] # Cutoff Tayini. Uç Değerlerden Arındırma.
train = train.drop("Talep3_Zscore", axis = 1) # Arındırmadan sonra türetilen Talep3_Zscore değişkeninin silinmesi 
train["U1_Zscore"] = zscore(train["U1"]) # U1_Zscore değişkeninin türetilmesi ve her bir kayıt için zscore değerinin hesaplanması
train = train.loc[np.logical_and(train.U1_Zscore < 2,train.U1_Zscore > -2),:] # Cutoff Tayini. Uç Değerlerden Arındırma.
train = train.drop("U1_Zscore", axis = 1) # Arındırmadan sonra türetilen U1_Zscore değişkeninin silinmesi 
train["U2_Zscore"] = zscore(train["U2"]) # U2_Zscore değişkeninin türetilmesi ve her bir kayıt için zscore değerinin hesaplanması
train = train.loc[np.logical_and(train.U2_Zscore < 2,train.U2_Zscore > -2),:] # Cutoff Tayini. Uç Değerlerden Arındırma.
train = train.drop("U2_Zscore", axis = 1) # Arındırmadan sonra türetilen U2_Zscore değişkeninin silinmesi 
train["U3_Zscore"] = zscore(train["U3"]) # U3_Zscore değişkeninin türetilmesi ve her bir kayıt için zscore değerinin hesaplanması
train = train.loc[np.logical_and(train.U3_Zscore < 2,train.U3_Zscore > -2),:] # Cutoff Tayini. Uç Değerlerden Arındırma.
train = train.drop("U3_Zscore", axis = 1) # Arındırmadan sonra türetilen U3_Zscore değişkeninin silinmesi 
train["Fiyat1_Zscore"] = zscore(train["Fiyat1"]) # Fiyat1_Zscore değişkeninin türetilmesi ve her bir kayıt için zscore değerinin hesaplanması
train = train.loc[np.logical_and(train.Fiyat1_Zscore < 2,train.Fiyat1_Zscore > -2),:] # Cutoff Tayini. Uç Değerlerden Arındırma.
train = train.drop("Fiyat1_Zscore", axis = 1) # Arındırmadan sonra türetilen Fiyat1_Zscore değişkeninin silinmesi 
train["Fiyat2_Zscore"] = zscore(train["Fiyat2"]) # Fiyat2_Zscore değişkeninin türetilmesi ve her bir kayıt için zscore değerinin hesaplanması
train = train.loc[np.logical_and(train.Fiyat2_Zscore < 2,train.Fiyat2_Zscore > -2),:] # Cutoff Tayini. Uç Değerlerden Arındırma.
train = train.drop("Fiyat2_Zscore", axis = 1) # Arındırmadan sonra türetilen Fiyat2_Zscore değişkeninin silinmesi 
train["Fiyat3_Zscore"] = zscore(train["Fiyat3"]) # Fiyat3_Zscore değişkeninin türetilmesi ve her bir kayıt için zscore değerinin hesaplanması
train = train.loc[np.logical_and(train.Fiyat3_Zscore < 2,train.Fiyat3_Zscore > -2),:] # Cutoff Tayini. Uç Değerlerden Arındırma.
train = train.drop("Fiyat3_Zscore", axis = 1) # Arındırmadan sonra türetilen Fiyat3_Zscore değişkeninin silinmesi 
train["DengeFiyat1_Zscore"] = zscore(train["DengeFiyat1"]) # DengeFiyat1_Zscore değişkeninin türetilmesi ve her bir kayıt için zscore değerinin hesaplanması
train = train.loc[np.logical_and(train.DengeFiyat1_Zscore < 2,train.DengeFiyat1_Zscore > -2),:] # Cutoff Tayini. Uç Değerlerden Arındırma.
train = train.drop("DengeFiyat1_Zscore", axis = 1) # Arındırmadan sonra türetilen DengeFiyat1_Zscore değişkeninin silinmesi
train["DengeFiyat2_Zscore"] = zscore(train["DengeFiyat2"]) # DengeFiyat2_Zscore değişkeninin türetilmesi ve her bir kayıt için zscore değerinin hesaplanması
train = train.loc[np.logical_and(train.DengeFiyat2_Zscore < 2,train.DengeFiyat2_Zscore > -2),:] # Cutoff Tayini. Uç Değerlerden Arındırma.
train = train.drop("DengeFiyat2_Zscore", axis = 1) # Arındırmadan sonra türetilen DengeFiyat2_Zscore değişkeninin silinmesi
train["DengeFiyat3_Zscore"] = zscore(train["DengeFiyat3"]) # DengeFiyat3_Zscore değişkeninin türetilmesi ve her bir kayıt için zscore değerinin hesaplanması
train = train.loc[np.logical_and(train.DengeFiyat3_Zscore < 2,train.DengeFiyat3_Zscore > -2),:] # Cutoff Tayini. Uç Değerlerden Arındırma.
train = train.drop("DengeFiyat3_Zscore", axis = 1) # Arındırmadan sonra türetilen DengeFiyat3_Zscore değişkeninin silinmesi

len(train) #  Öğrenme Datasında Anomali Kayıtları Çıkardıktan Sonra Kalan Kayıt Sayısı
train_y = train.AL_SAT # Arındırma sonrası öğrenme verisinde hedef değişkenin belirtilmesi 
train_x = train.drop("AL_SAT", axis = 1) # Arındırma sonrası elde edilen öğrenme verisinde sadece açıklayıcı değişkenlerin bulundurulması
# Yukarıda öğrenme verisi için yapılan arındırma işlemlerinin aynısı test verisi üzerinde de yapılacaktır.
test["Talep1_Zscore"] = zscore(test["Talep1"])
test = test.loc[np.logical_and(test.Talep1_Zscore < 2,test.Talep1_Zscore > -2),:]
test = test.drop("Talep1_Zscore", axis = 1)
test["Talep2_Zscore"] = zscore(test["Talep2"])
test = test.loc[np.logical_and(test.Talep2_Zscore < 2,test.Talep2_Zscore > -2),:]
test = test.drop("Talep2_Zscore", axis = 1)
test["Talep3_Zscore"] = zscore(test["Talep3"])
test = test.loc[np.logical_and(test.Talep3_Zscore < 2,test.Talep3_Zscore > -2),:]
test = test.drop("Talep3_Zscore", axis = 1)
test["U1_Zscore"] = zscore(test["U1"])
test = test.loc[np.logical_and(test.U1_Zscore < 2,test.U1_Zscore > -2),:]
test = test.drop("U1_Zscore", axis = 1)
test["U2_Zscore"] = zscore(test["U2"])
test = test.loc[np.logical_and(test.U2_Zscore < 2,test.U2_Zscore > -2),:]
test = test.drop("U2_Zscore", axis = 1)
test["U3_Zscore"] = zscore(test["U3"])
test = test.loc[np.logical_and(test.U3_Zscore < 2,test.U3_Zscore > -2),:]
test = test.drop("U3_Zscore", axis = 1)
test["Fiyat1_Zscore"] = zscore(test["Fiyat1"])
test = test.loc[np.logical_and(test.Fiyat1_Zscore < 2,test.Fiyat1_Zscore > -2),:]
test = test.drop("Fiyat1_Zscore", axis = 1)
test["Fiyat2_Zscore"] = zscore(test["Fiyat2"])
test = test.loc[np.logical_and(test.Fiyat2_Zscore < 2,test.Fiyat2_Zscore > -2),:]
test = test.drop("Fiyat2_Zscore", axis = 1)
test["Fiyat3_Zscore"] = zscore(test["Fiyat3"])
test = test.loc[np.logical_and(test.Fiyat3_Zscore < 2,test.Fiyat3_Zscore > -2),:]
test = test.drop("Fiyat3_Zscore", axis = 1)
test["DengeFiyat1_Zscore"] = zscore(test["DengeFiyat1"])
test = test.loc[np.logical_and(test.DengeFiyat1_Zscore < 2,test.DengeFiyat1_Zscore > -2),:]
test = test.drop("DengeFiyat1_Zscore", axis = 1)
test["DengeFiyat2_Zscore"] = zscore(test["DengeFiyat2"])
test = test.loc[np.logical_and(test.DengeFiyat2_Zscore < 2,test.DengeFiyat2_Zscore > -2),:]
test = test.drop("DengeFiyat2_Zscore", axis = 1)
test["DengeFiyat3_Zscore"] = zscore(test["DengeFiyat3"])
test = test.loc[np.logical_and(test.DengeFiyat3_Zscore < 2,test.DengeFiyat3_Zscore > -2),:]
test = test.drop("DengeFiyat3_Zscore", axis = 1)
len(test) #  Test Etme Datasında Anomali Kayıtları Çıkardıktan Sonra Kalan Kayıtların Sayısı
test_y = test.AL_SAT # Arındırma sonrası test etme verisinde hedef değişkenin belirtilmesi 
test_x = test.drop("AL_SAT", axis = 1) # Arındırma sonrası elde edilen test etme verisinde sadece açıklayıcı değişkenlerin bulundurulması
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
# Modeller içerinde  model başarımı en iyi olan modelin (accuracy açısından) SVM modeli olduğu görülmüştür.
# Ancak modelin karmaşıklık matrisine bakıldığında SVM modelininde başarımının iyi olmadığı görülmektedir.
