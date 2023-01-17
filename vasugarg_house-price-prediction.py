import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

import numpy as np
data=pd.read_csv("../input/kc-house-data/kc_house_data.csv")

data=data.drop(["id","date"],axis=1)

data["sqft_above"].fillna(1788.39,inplace=True)

data.describe()
Y=data["price"].values

Y=np.log(Y)

features=data.columns

X1=list(set(features)-set(["price"]))

X=data[X1].values

ss=StandardScaler()

X=ss.fit_transform(X)
Xt,Xts,Yt,Yts=train_test_split(X,Y,test_size=0.4,random_state=0)

lr=LinearRegression()

model=lr.fit(Xt,Yt)

predict=lr.predict(Xts)

print("train Accuracy",lr.score(Xt,Yt))

print("test Accuracy",lr.score(Xts,Yts))
regr = RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=80 ,min_samples_leaf=1

                             ,min_samples_split=2,random_state=0)

model=regr.fit(Xt,Yt)

predict=regr.predict(Xts)

print("train Accuracy : ",regr.score(Xt,Yt))

print("test Accuracy : ",regr.score(Xts,Yts))