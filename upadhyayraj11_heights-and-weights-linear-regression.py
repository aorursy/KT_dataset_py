import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

%matplotlib inline



print(os.listdir("../input"))
df = pd.read_csv("../input/data.csv")

df.head(5)
plt.title("Us data")

plt.xlabel("Height")

plt.ylabel("Weight")

plt.scatter(df.Height,df.Weight,color='blue')
np.where(np.isnan(df['Height']))
np.where(np.isnan(df['Weight']))
from sklearn.model_selection import train_test_split

x = df[['Height']]

y = df['Weight']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train,y_train)
df.head(1)
print(model.predict([[1.47]]))
model.score(x_test,y_test)