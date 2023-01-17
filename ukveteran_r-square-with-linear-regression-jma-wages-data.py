import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("../input/wages-data/Griliches.csv")
df.head()
df=df.drop(["rns","rns80","mrt","mrt80","smsa","smsa80","iq","med"],axis=1)

df.head()
df.corr()
df.describe().T
linear_reg=LinearRegression()

x=df.age.values.reshape(-1,1)

y=df.tenure.values.reshape(-1,1)



linear_reg.fit(x,y)
y_head=linear_reg.predict(x)
plt.plot(x,y_head,color="r")

plt.show()
from sklearn.metrics import r2_score

print("r_square score:", r2_score(y,y_head))