import pandas as pd

import numpy as np

from sklearn.datasets import load_boston
boston=load_boston()
boston.keys()
print(boston.DESCR)
from sklearn.preprocessing import StandardScaler

ss=StandardScaler().fit(boston.data)

data=pd.DataFrame(ss.transform(boston.data))
data
data=pd.DataFrame(data)
data.columns=boston.feature_names
data['target']=boston.target
data.head()
data.corr()
import seaborn as sns

import matplotlib.pyplot as plt

sns.set()
fig_dims = (20,15)

fig, ax = plt.subplots(figsize=fig_dims)



cmap=sns.diverging_palette(h_neg=15,h_pos=240,as_cmap=True)

sns.heatmap(data.corr(),center=0,cmap=cmap,linewidths=1,annot=True,fmt='.2f',ax=ax);
plt.plot(data.TAX)

plt.plot(data.RAD)
data.drop(['TAX','CHAS','DIS'],axis=1,inplace=True)
data.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test =train_test_split(data.drop('target',axis=1),data['target'],test_size=0.25,random_state=12)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_train,y_train)
lr.score(X_test,y_test)
from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor(n_estimators=100,max_depth=2,max_features=0.2,subsample=0.6,random_state=4)

gbr.fit(X_train,y_train)
gbr.score(X_train,y_train)
gbr.score(X_test,y_test)
import pickle

pic_file=open('model.pkl','wb')

pickle.dump(gbr,pic_file)

pic_file.close()
import pickle

pic_file=open('data_prep.pkl','wb')

pickle.dump(ss,pic_file)

pic_file.close()
pic_in=open('model.pkl','rb')

model=pickle.load(pic_in)
model.predict(X_train.head(1))