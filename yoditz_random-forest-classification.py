import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data=pd.read_csv('../input/mushrooms.csv')
data.head()
data=pd.get_dummies(data, drop_first=True)
from sklearn.cross_validation import train_test_split
X=data.drop('class_p',axis=1)

y=data['class_p']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.ensemble import RandomForestClassifier
rd= RandomForestClassifier(n_estimators=100)
rd.fit(X_train,y_train)
prediction=rd.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,prediction))

print(classification_report(y_test,prediction))