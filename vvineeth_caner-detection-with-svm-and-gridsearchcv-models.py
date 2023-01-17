import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.datasets import load_breast_cancer



cancer = load_breast_cancer()

cancer.keys()
df_feat = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])

df_feat.head()
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC



X = df_feat

y = cancer['target']



#splitting the dataset

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state=100)



#traing the model

model = SVC()



model.fit(train_X, train_y)



prediction = model.predict(test_X)
# Gettung Accuracy

from sklearn.metrics import confusion_matrix, classification_report 



print(confusion_matrix(test_y, prediction))

print('\n')

print(classification_report(test_y, prediction))
# training and predicting with GridSearchCV model



from sklearn.model_selection import GridSearchCV



parameters = {'C':[0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001]}



grid = GridSearchCV(SVC(), param_grid = parameters)



grid.fit(train_X, train_y)



predict = grid.predict(test_X)
# Accuracy



print(confusion_matrix(test_y, predict))

print('\n')

print(classification_report(test_y, predict))