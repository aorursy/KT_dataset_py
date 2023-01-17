import warnings

warnings.filterwarnings('ignore')

#lib

%matplotlib inline

import matplotlib as plt

import matplotlib.pyplot as plt

import numpy as np

from pandas import read_csv as read

import seaborn as sns
import pandas as pd
#open

path='../input/Dataset_spine.csv'

data=read(path,delimiter=",")

pd.set_option('display.max_columns',100)

data.head()
data.keys()
data.shape
#copy in work dataset

dataset=data.iloc[:,0:12]
dataset.head()
target=data['Class_att']
target.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(dataset,target,random_state=0)
# 1. neighbors

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=6)#1

knn.fit(X_train,y_train)



# 1. testing

#np.mean(y_pred == y_test))) #alternative

#knn.score(X_test, y_test)

print("In learn dataset: {:.3f}".format(knn.score(X_train, y_train)))

print("In test dataset: {:.3f}".format(knn.score(X_test, y_test)))
#Test of neighbors

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(

cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []

test_accuracy = []

#  n_neighbors between 1 and 10

neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:

    # model

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)

    clf.fit(X_train, y_train)

    # write learn

    training_accuracy.append(clf.score(X_train, y_train))

    # write test

    test_accuracy.append(clf.score(X_test, y_test))

  

plt.plot(neighbors_settings, training_accuracy, label="learn dataset")

plt.plot(neighbors_settings, test_accuracy, label="test dataset")

plt.ylabel("Result")

plt.xlabel("count neighbors")

plt.legend()
from sklearn.ensemble import RandomForestClassifier

X_train,X_test,y_train,y_test=train_test_split(dataset,target,random_state=0)

forest = RandomForestClassifier(n_estimators=1000, random_state=0)

forest.fit(X_train, y_train)

print("In learn dataset: {:.3f}".format(forest.score(X_train, y_train)))

print("In test dataset: {:.3f}".format(forest.score(X_test, y_test)))
from sklearn.neural_network import MLPClassifier

X_train,X_test,y_train,y_test=train_test_split(dataset,target,random_state=0)

mlp = MLPClassifier(max_iter=1000,solver='lbfgs', random_state=0,hidden_layer_sizes=[10, 10])

mlp.fit(X_train, y_train)

print("In learn dataset: {:.3f}".format(mlp.score(X_train, y_train)))

print("In test dataset: {:.3f}".format(mlp.score(X_test, y_test)))
#try scaler for neural

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# fit 

scaler.fit(X_train)

# transform train

X_train_scaled = scaler.transform(X_train)

# transform test

X_test_scaled = scaler.transform(X_test)
from sklearn.neural_network import MLPClassifier

X_train,X_test,y_train,y_test=train_test_split(dataset,target,random_state=0)

mlp = MLPClassifier(max_iter=160,solver='lbfgs',alpha=0.1,random_state=0,hidden_layer_sizes=[45])

mlp.fit(X_train_scaled, y_train)

print("In learn dataset: {:.3f}".format(mlp.score(X_train_scaled, y_train)))

print("In test dataset: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
#statistic

data['Col5'].plot(kind='kde')

plt.ylabel('Плотность')

plt.title('Плотность распределения')

plt.grid(True)
#hist

sns.distplot(data['Col5'])

plt.ylabel('Плотность')

plt.title('Плотность распределения')

plt.grid(True)
sns.jointplot(x="Col1", y="Col4", data=data )
data.corr()
sns.pairplot(dataset)
#попытка найти корреляцию в двух сигналах

pd.DataFrame({'s1':dataset.Col1/max(dataset.Col1), 's2':dataset.Col4/max(dataset.Col4)}).plot()

pd.rolling_corr(dataset.Col1/max(dataset.Col1),dataset.Col4/max(dataset.Col4),window=5).plot(style='.')