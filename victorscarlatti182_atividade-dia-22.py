import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

hepatitis = pd.read_csv("../input/hepatitis/hepatitis.csv")
hepatitis.describe
hepatitis.shape
hepatitis.info()
hepatitis.isnull().sum()
import matplotlib.pyplot as plt

plt.hist(hepatitis['age'], bins=8, rwidth=0.9)

plt.xlim(0,100)

plt.ylim(0,100)

plt.xlabel('Idade')

plt.ylabel('quantidade')

plt.show()
hepatitis.keys()

X = hepatitis.loc[:,'age':'histology']

y = hepatitis.loc[:,['class']]
from sklearn.preprocessing import StandardScaler

SS = StandardScaler()

X = SS.fit_transform(X)
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
from sklearn.linear_model import Perceptron

pp = Perceptron()

modelo = pp.fit(X_train, y_train)

y_predict = modelo.predict(X_test)
score = modelo.score(X_test, y_test)

score