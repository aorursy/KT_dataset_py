import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score



dataset = pd.read_csv('../input/breast_cancer.csv')

X = dataset.iloc[:, 1:-1].values

y = dataset.iloc[:, -1].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



classifier = LogisticRegression()

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)



conf_mat = confusion_matrix(y_test, y_pred)

acc_scr =  accuracy_score(y_test, y_pred)



print(conf_mat)

print(acc_scr)



accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))