# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
data_file = "../input/data.csv"
data = pd.read_csv(data_file)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data.shape
data.info()
data.isnull().any()
data.drop('Unnamed: 32', axis=1, inplace =True)

data.sample(10)
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder() 

data['diagnosis'] = lb.fit_transform(data['diagnosis'])
data.sample(10)
data.shape
X = data.iloc[:,0:31] 
Y = data.iloc[:,1]
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