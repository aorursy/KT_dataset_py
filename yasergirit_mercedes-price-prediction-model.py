# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sbn



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/used-car-dataset-ford-and-mercedes/merc.csv")

df.head()
df.describe()
df.isnull().sum()
sbn.distplot(df["price"])
sbn.countplot(df["year"])
df.corr()
df.corr()["price"].sort_values()
sbn.scatterplot(x = "mileage", y = "price", data = df)
df.sort_values("price", ascending = False).head(20)
df.sort_values("price", ascending = True).head(20)
len(df)
len(df) * 0.01
df_99 = df.sort_values("price", ascending = False).iloc[131:]
df_99.describe()
sbn.distplot(df_99["price"])
df.describe()
df.groupby("year").mean()["price"]
df_99.groupby("year").mean()["price"]
df[df.year != 1970 ].groupby("year").mean()["price"] # drop the 1970's cars
df = df[df.year != 1970]
df.groupby("year").mean()["price"]
df.head()
df.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["model"] = le.fit_transform(df["model"])
df["transmission"] = le.fit_transform(df["transmission"])
df["fuelType"] = le.fit_transform(df["fuelType"])
df.head()
y = df["price"].values

x = df.drop("price", axis = 1).values

y
x
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 17)
len(x_train)
len(x_test)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
x_train.shape
model = Sequential()
model.add(Dense(12,activation="relu"))

model.add(Dense(12,activation="relu"))

model.add(Dense(12,activation="relu"))

model.add(Dense(12,activation="relu"))



model.add(Dense(1))



model.compile(optimizer = "adam", loss = "mse")
model.fit(x = x_train, y = y_train, validation_data = (x_test, y_test), batch_size = 250, epochs = 300)
dataLoss = pd.DataFrame(model.history.history)
dataLoss.head()
dataLoss.plot()
from sklearn.metrics import mean_squared_error, mean_absolute_error
predList = model.predict(x_test)
predList
mean_absolute_error(y_test, predList) 
plt.scatter(y_test, predList)

plt.plot(y_test, y_test, "g-*")