# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


#packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#dataset

dataset = pd.read_csv('../input/mushrooms.csv')

#explore dataset

dataset.head(10)
dataset.describe()
dataset.info()
#dataset.loc[dataset["EDIBLE"].isnull()]
#assign X and y
X = dataset.iloc[:, 1:23].values
y = dataset.iloc[:, [0]].values
#X = pd.DataFrame(X)
#y = pd.DataFrame(y)
# Encoding categorical data for X dummy variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
for i in range (0,22):
    labelencoder_X = LabelEncoder()
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
#encoding for y
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y_0 = LabelEncoder()
y[:, 0] = labelencoder_y_0.fit_transform(y[:, 0])
#split data -try the model with random state 42 and also it is chwcked with random state 0
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
y_train = y_train.astype(int)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 3, criterion = 'entropy', max_depth=7, random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_test = y_test.astype(int)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 100)
print(accuracies)
print(accuracies.mean())
print(accuracies.std())