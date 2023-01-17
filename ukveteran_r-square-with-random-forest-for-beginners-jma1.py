import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("../input/cost-function-for-electricity-producers/Electricity.csv")
df.head()
x=df.cost.values.reshape(-1,1)

y=df.q.values.reshape(-1,1)
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100,random_state=42)

rf.fit(x,y)
rf.predict([[7.5]])
y_head=rf.predict(x)
from sklearn.metrics import r2_score

print("r_score:",r2_score(y,y_head))