# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

bank_file = "../input/Churn_Modelling.csv"
bank_data = pd.read_csv(bank_file)

import os
print(os.listdir("../input"))

# Sıralama işlemi
bank_data = bank_data.sort_values(by=['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited'])

# Any results you write to the current directory are saved as output.
bank_data.head() #ilk 5 satır
bank_data.tail() # son 5 satır
bank_data.shape # satır ve sütun sayısı
bank_data.info() # bellek kullanımı ve türleri
bank_data.isnull().any() # null değer varmı
bank_data.describe() # basit bir istatistik
# Histogram Gösterimi
num_bins=10
bank_data.hist(bins=num_bins, figsize=(20,15))
plt.matshow(bank_data.corr()) # Korelasyon grafik
bank_data.corr() # Tablo halinde koralasyon
# seaborn ısı haritası
import seaborn as sns
corr = bank_data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
# Veri ön işlemeyi burada erken yaptık, ilerki adımlarda kolaylık olsun
bank_data.drop('RowNumber', axis=1, inplace = True)
bank_data.drop('CustomerId', axis=1, inplace = True)
bank_data.corr() # tekrar korelasyon istatistikleri kontrol ediyoruz
# Plot çizini oluşturma - 2 yüksek koralasyon değeri olan öznitelik seçtik
bank_data.plot(x='CreditScore', y='Balance', style='o')  
plt.title('CreditScore - Balance')  
plt.xlabel('CreditScore')  
plt.ylabel('Balance')  
plt.show() 

# Dataset üzerinde bulunan verileri model çıkarmak için dönüştürme işlemi
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder() 

bank_data['Surname'] = lb.fit_transform(bank_data['Surname'])
bank_data['Geography'] = lb.fit_transform(bank_data['Geography'])
bank_data['Gender'] = lb.fit_transform(bank_data['Gender'])


bank_data.sample(10) # 10 adet örnek gösterimi
bank_data['Age'].max()
bank_data['CreditScore'].max ()
#Eksik Değer kontrolü
bank_data.isnull().any() # null Değer varmı

bank_data.isnull().sum() # Null olan öznitelikler
# Yaşın aykırı uç verilerin tespiti
import seaborn as sns
sns.boxplot(x=bank_data['Age'])

# Kredi puanının aykırı uç verilerin tespiti
sns.boxplot(x=bank_data['CreditScore'])
# Mevcut özniteliklerden yeni bir öznitelik oluşturma
# Bu arada burada Kredi Puanına göre ev alma durumunu oluşturmuş olduk

def KrediPuani(puan):
    return (puan>550)

bank_data['EvAlma'] = bank_data['CreditScore'].apply(KrediPuani)
bank_data['EvAlma'] = lb.fit_transform(bank_data['EvAlma']) # EvAlma sütununu convert ettik 
bank_data.sample(10) # sonuçlar
# Balance ve CreditScore sütununu aldık 
from sklearn import preprocessing
# Normalize total_bedrooms column
x_array = np.array(bank_data['Balance'])
y_array = np.array(bank_data['CreditScore'])
normalized_X = preprocessing.normalize([x_array])
normalized_Y = preprocessing.normalize([y_array])
bank_data.sample(15) # sonuçları inceleme
# Verileri eğitim ve test verisi olarak bölüyoruz.
X = bank_data.iloc[:,0:12] 
Y = bank_data.iloc[:,11]
#Eğitim ve doğrulama veri kümelerinin ayrıştırılması
from sklearn.model_selection import train_test_split  
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
#Naive Bayes modelinin oluşturulması 1.model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

model = GaussianNB()
#NB modelinin K-katlamalı çapraz doğrulama ile ACC değerinin hesaplanması
from sklearn.model_selection import KFold
scoring = 'accuracy'
kfold = KFold(n_splits=10, random_state=seed)
cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
cv_results # sonuç
msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
msg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(Y_test, Y_pred))
print("Karmaşıklık Matrisi \n",confusion_matrix(Y_test, Y_pred))
# Accuracy score

print("Banka müşterisinin Exited(Bankadan Ayrılmama/Memnuniyet) olma olasılığı (ACC): %%%.2f" % (accuracy_score(Y_pred,Y_test)*100))