import pandas as pd

import numpy as np
data = pd.read_csv("../input/winequality-red.csv")



data.head()
data.isnull().sum()
data.corr()
list(data)
x = data[['fixed acidity',

 'volatile acidity',

 'citric acid',

 'chlorides',

 'total sulfur dioxide',

 'density',

 'sulphates',

 'alcohol']]



y = data['quality']
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

clf.fit(x_train,y_train)
a = clf.predict(x_test)



from sklearn.metrics import accuracy_score,confusion_matrix



print(accuracy_score(y_test,a))

print(confusion_matrix(y_test,a))
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

clf.fit(x_train,y_train)

a = clf.predict(x_test)
print(accuracy_score(y_test,a))

print(confusion_matrix(y_test,a))