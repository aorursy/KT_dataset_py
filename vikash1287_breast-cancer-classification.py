



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/data.csv")
#y//dependen variable

#x//independent variable

y=dataset["diagnosis"]

dataset.shape
y.head()
x=dataset.iloc[:,2:-1].values
x
plt.figure(figsize=(20,10))

sns.heatmap(dataset.corr(),annot=True)
#diagnosis

from sklearn.preprocessing import LabelEncoder
y_label_encoder = LabelEncoder()

y=y_label_encoder.fit_transform(y)
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
classifier = SVC(kernel="linear",random_state=5)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)
print(classification_report(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier(n_estimators=10,criterion = "entropy",random_state = 0)
classifier1.fit(x_train,y_train)
y_pred1 = classifier1.predict(x_test)
cm1 = confusion_matrix(y_test,y_pred1)

sns.heatmap(cm1,annot=True)
print(classification_report(y_test,y_pred1))
from sklearn.naive_bayes import GaussianNB
classifier2 = GaussianNB()

classifier2.fit(x_train,y_train)
y_pred2 = classifier2.predict(x_test)
cm2 = confusion_matrix(y_test,y_pred2)

sns.heatmap(cm2,annot=True)
print(classification_report(y_test,y_pred2))