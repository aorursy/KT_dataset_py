import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/hepatitis/hepatitis.csv")

df.head()
df.describe()

df.shape
df.info()
df.isnull()
df.sum()
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

plt.hist(df['age'], bins=20, rwidth=0.9)

plt.title('Idade')

plt.show()
df.columns
x = df.iloc[: , 1:]

x.head()
y = df['class']

y.head()
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

x = ss.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2)
from sklearn.linear_model import Perceptron

pp = Perceptron()

modelo = pp.fit(x_train, y_train)

y_predict = modelo.predict(x_test)
score = modelo.score(x, y)

score