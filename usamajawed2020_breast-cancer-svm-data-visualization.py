import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV



cancer = load_breast_cancer()

cancer.keys()



#Data Visualization

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))



sns.pairplot(df_cancer, hue = 'target' ,vars=['mean radius','mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])

sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data= df_cancer)



plt.figure(figsize = (20, 10))

sns.heatmap(df_cancer.corr(), annot= True)



X = df_cancer.drop(['target'], axis = 1)

y = df_cancer['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)



#Normalizing the DataSet



#Training Set

min_train = X_train.min()

range_train = (X_train - min_train).max()

X_train_scaled = (X_train - min_train) / range_train



#Test Set

min_test = X_test.min()

range_test = (X_test - min_train).max()

X_test_scaled = (X_test - min_test) / range_test



#Fitting into Model

svc_model = SVC()

svc_model.fit(X_train_scaled, y_train)

y_pred = svc_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (20, 10))

sns.heatmap(cm, annot = True)

print("Non-Tuned Model Report")

print(classification_report(y_test, y_pred))



#Tuning the SVC Model

param_grid = { 'C' : [0.1, 1, 10, 100], 'gamma' : [1, 0.1, 0.01, 0.001], 'kernel' : ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4 )

grid.fit(X_train_scaled, y_train)

print(grid.best_params_)



#Predicting with Tuned Model

grid_prediction = grid.predict(X_test_scaled)

cm = confusion_matrix(y_test, grid_prediction)

plt.figure(figsize = (20, 10))

sns.heatmap(cm, annot=True)

print("Tuned Model Report")

print(classification_report(y_test, grid_prediction))