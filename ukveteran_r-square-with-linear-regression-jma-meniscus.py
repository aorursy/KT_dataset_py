import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("../input/meniscus-repair-methods/Meniscus.csv")

df
df.corr()
df.describe().T
linear_reg=LinearRegression()

x=df.Displacement.values.reshape(-1,1)

y=df.Stiffness.values.reshape(-1,1)



linear_reg.fit(x,y)
y_head=linear_reg.predict(x)
plt.plot(x,y_head,color="r")

plt.show()
from sklearn.metrics import r2_score

print("r_square score:", r2_score(y,y_head))