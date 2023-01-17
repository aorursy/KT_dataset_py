import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sklearn import datasets as dt

iris= dt.load_iris()
data = pd.read_csv("/kaggle/input/hepatitis/hepatitis.csv")

data.head()
data.describe()
data.shape
data.info()
data.isnull().sum()
import matplotlib.pyplot as plt

idade = data['age']

plt.figure(figsize=(8, 6))

plt.hist(idade, bins=range(2, 100,10))

print (idade)

plt.title('distribuição de idades')

plt.xlabel('idade')

plt.grid()

plt.show()
iris.keys()

X=iris.data

y= iris.target
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X= ss.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)
from sklearn.linear_model import Perceptron

pp =Perceptron()

modelo=pp.fit(X_train,y_train)

y_pred = modelo.predict(X_test)
score =modelo.score (X_test,y_test)

score