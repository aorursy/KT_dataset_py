import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("../input/meniscus-repair-methods/Meniscus.csv")

df.head()
x=df.Displacement.values.reshape(-1,1)

y=df.Stiffness.values.reshape(-1,1)
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100,random_state=42)

rf.fit(x,y)
rf.predict([[7.5]])
x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=rf.predict(x_)
plt.figure(figsize=(18,6))

plt.scatter(x,y,color="r")

plt.plot(x_,y_head,color="g")

plt.xlabel("Displacement")

plt.ylabel("Stiffness")

plt.show()