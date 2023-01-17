# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

import seaborn as sns



from sklearn.preprocessing import scale

import sklearn.linear_model as skl_lm

from sklearn.metrics import mean_squared_error, r2_score





%matplotlib inline

plt.style.use('seaborn-white')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from sklearn.model_selection import train_test_split

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/train.csv")

df.head()
df=df.drop(["songtitle", "artistname","songID","artistID"], axis = 1) 

df.head()
d=df.Top10

d.head()
df=df.drop(columns="Top10")

df.describe()
X_train, X_test, y_train, y_test = train_test_split(df,d, test_size=0.2)
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X_train,y_train)
df_test=pd.read_csv("../input/test.csv")

df_test.head()


df_test=df_test.drop(["songtitle", "artistname","songID","artistID"], axis = 1) 

df_test.head()
model.score(X_test,y_test)
d=pd.DataFrame({})



d["songID"]=pd.read_csv("../input/test.csv").songID

d5=model.predict_proba(df_test)

d["Top10"]=d5.T[1]

d.to_csv("final.csv", index=False)