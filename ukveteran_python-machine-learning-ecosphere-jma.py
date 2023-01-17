import numpy as np
arr = np.array([1,3,4,5,6])
arr
arr.shape
arr.dtype
arr = np.array([1,'st','er',3])
arr.dtype
arr = np.array([[1,2,3],[2,4,6],[8,8,8]])
arr.shape
arr
arr = np.zeros((2,4))
arr
arr = np.ones((2,4))
arr
arr = np.identity(3)
arr
arr = np.random.randn(3,4)
arr
from io import BytesIO
b = BytesIO(b"2,23,33\n32,42,63.4\n35,77,12")
arr = np.genfromtxt(b, delimiter=",")
arr
arr[1]
arr = np.arange(12).reshape(2,2,3)
arr
arr[0]
arr = np.arange(10)
arr[5:]
arr[5:8]
arr[:-5]
arr = np.arange(12).reshape(2,2,3)
arr
arr[1:2]
arr = np.arange(27).reshape(3,3,3)
arr
arr[:,:,2]
arr[...,2]
arr = np.arange(9).reshape(3,3)
arr
arr[[0,1,2],[1,0,0]]
cities = np.array(["delhi","banglaore","mumbai","chennai","bhopal"])
city_data = np.random.randn(5,3)
city_data
city_data[cities =="delhi"]
city_data[city_data >0]
city_data[city_data >0] = 0
city_data
arr = np.arange(15).reshape(3,5)
arr
arr + 5
arr * 2
arr1 = np.arange(15).reshape(5,3)
arr2 = np.arange(5).reshape(5,1)
arr2 + arr1
arr1
arr2
arr1 = np.random.randn(5,3)
arr1
np.modf(arr1)
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
B = np.array([[9,8,7],[6,5,4],[1,2,3]])
A.dot(B)
A = np.arange(15).reshape(3,5)
A.T
np.linalg.svd(A)
a = np.array([[7,5,-3], [3,-5,2],[5,3,-7]])
b = np.array([16,-8,0])
x = np.linalg.solve(a, b)
x
np.allclose(np.dot(a, x), b)
import pandas as pd
d =  [{'city':'Delhi',"data":1000},
      {'city':'Banglaore',"data":2000},
      {'city':'Mumbai',"data":1000}]
pd.DataFrame(d)
df = pd.DataFrame(d)
df = pd.DataFrame(np.random.randn(8, 3),
columns=['A', 'B', 'C'])
nparray = df.values
type(nparray)
from numpy import nan
df.iloc[4,2] = nan
df
df.fillna(0)
df1 = pd.DataFrame({'col1': ['col10', 'col11', 'col12', 'col13'],
                    'col2': ['col20', 'col21', 'col22', 'col23'],
                    'col3': ['col30', 'col31', 'col32', 'col33'],
                    'col4': ['col40', 'col41', 'col42', 'col43']},
                   index=[0, 1, 2, 3])
df1
df4 = pd.DataFrame({'col2': ['col22', 'col23', 'col26', 'col27'],
                    'Col4': ['Col42', 'Col43', 'Col46', 'Col47'],
                    'col6': ['col62', 'col63', 'col66', 'col67']},
                   index=[2, 3, 6, 7])

pd.concat([df1,df4], axis=1)
from sklearn import datasets
diabetes = datasets.load_diabetes()
X = diabetes.data[:10]
y = diabetes.target
X[:5]
y[:10]
feature_names=['age', 'sex', 'bmi', 'bp',
               's1', 's2', 's3', 's4', 's5', 's6']
from sklearn import datasets
from sklearn.linear_model import Lasso

from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

diabetes = datasets.load_diabetes()
X_train = diabetes.data[:310]
y_train = diabetes.target[:310]

X_test = diabetes.data[310:]
y_test = diabetes.target[310:]

lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

scores = list()
scores_std = list()

estimator = GridSearchCV(lasso,
                         param_grid = dict(alpha=alphas))

estimator.fit(X_train, y_train)
estimator.best_score_
estimator.best_estimator_
estimator.predict(X_test)
import numpy
import theano.tensor as T
from theano import function
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)
f(8, 2)
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X_train = cancer.data[:340]
y_train = cancer.target[:340]

X_test = cancer.data[340:]
y_test = cancer.target[340:]

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(15, input_dim=30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          epochs=20,
          batch_size=50)
predictions = model.predict_classes(X_test)
from sklearn import metrics

print('Accuracy:', metrics.accuracy_score(y_true=y_test, y_pred=predictions))
print(metrics.classification_report(y_true=y_test, y_pred=predictions))
model = Sequential()
model.add(Dense(15, input_dim=30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=20,
          batch_size=50)
predictions = model.predict_classes(X_test)
print('Accuracy:', metrics.accuracy_score(y_true=y_test, y_pred=predictions))
print(metrics.classification_report(y_true=y_test, y_pred=predictions))