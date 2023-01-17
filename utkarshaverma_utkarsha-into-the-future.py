import numpy as np

import pandas as pd

train_data=pd.read_csv("/kaggle/input/into-the-future/train.csv")

test_data=pd.read_csv("/kaggle/input/into-the-future/test.csv")
train_data.head()
test_data.head()
train_data.describe()
from sklearn.linear_model import LinearRegression

f1=train_data['feature_1']

f1_df=pd.DataFrame(f1)

f1_df.head()
f2=train_data['feature_2']

f2_df=pd.DataFrame(f2)
f2_df.head()
LR=LinearRegression()

LR.fit(f1_df,f2_df)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(f1_df,f2_df,test_size=0.2,random_state=0)
y_test_a=LR.predict(X_test)

y_train_a=LR.predict(X_train)
from sklearn.metrics import r2_score
r2_score(y_test,y_test_a)
r2_score(y_train,y_train_a)
x = test_data['feature_1']

x = x.values.reshape(-1,1)

y = LR.predict(x)





print(y)

print(exit)

data = pd.read_csv("../input/into-the-future/test.csv")

data['feature_2'] = y



data.to_csv('data2.csv')

data.head()