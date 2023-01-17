# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data_csv = "../input/logistic_regression_predictions.csv"
data = pd.read_csv(data_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data.head() # ilk 5 satır
data.tail() # son 5 satır
data.shape # satır ve sütun sayısı
data.info() # bellek kullanımı ve veri türleri
data.describe() # basit istatistikler
data.isnull().any() # null veri varmı
num_bins=10
data.hist(bins=num_bins, figsize=(20,15)) # veri görselleştirme - histogram grafiği
plt.matshow(data.corr()) # Korelasyon grafik
# seaborn ısı haritası
import seaborn as sns
corr = data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
data.corr() # Koralasyon istatistik
# Veri bölme işlemi (train & test split)
X = data.iloc[:, 0:1] 
Y = data.iloc[:, 1]
# bölünen veriler ile train & test işlemi
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)  
# train işleminden çıkan veriler ile 1. Model oluşturma
from sklearn.linear_model import LinearRegression  
model = LinearRegression()  
model.fit(X_train, y_train) 
print("Kesim noktası:", model.intercept_)  
# Modeli test etme işlemi
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state = 100)

model = model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

result = metrics.accuracy_score(Y_test, Y_pred)*100
#Accuracy değeri 
print("Kaç kişi öldü : ACC: %%%.2f" %result)
print("Kaç kişi kurtuldu: ACC: %%%.2f" %(100-result))
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
#Naive Bayes modelinin oluşturulması 2. Model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
#NB modelinin K-katlamalı çapraz doğrulama ile ACC değerinin hesaplanması
from sklearn.model_selection import cross_val_score
scoring = 'accuracy'
kfold = KFold(n_splits=10, random_state=seed)
cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
cv_results
msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
msg
# 2. Modelin testi
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score

result2 = accuracy_score(Y_pred,Y_test)
print("Kaç kişi öldü : ACC: ",result2)
print("Kaç kişi kurtludu: ACC: ",0.99-result2)
