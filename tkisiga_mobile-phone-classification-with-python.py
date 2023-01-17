import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df_train=pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')

df_train.head()
df_train.columns
df_train.info
df_train['price_range'].describe()
sns.distplot(df_train['price_range'])
df_train['price_range'].hist()
corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(18, 14))

sns.heatmap(corrmat, square=True,annot=True);
var='battery_power'

data = pd.concat([df_train['price_range'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price_range', ylim=(0,4));



var='blue'

data = pd.concat([df_train['price_range'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price_range', ylim=(0,4));

var='clock_speed'

data = pd.concat([df_train['price_range'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price_range', ylim=(0,4));

var='ram'

data = pd.concat([df_train['price_range'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price_range', ylim=(0,4));

df_train['px_area']= df_train['px_height']* df_train['px_width']
df_train['phone_area']= df_train['sc_h']* df_train['sc_w']
train_data=df_train.drop(['px_height','px_width','sc_h','sc_w'],axis=1)

train_data
df_test=pd.read_csv('/kaggle/input/mobile-price-classification/test.csv')
df_test['px_area']= df_test['px_height']*df_test['px_width']

df_test['phone_area']= df_train['sc_h']* df_train['sc_w']

df_test
test_data=df_test.drop(['px_height','px_width','sc_h','sc_w','id'],axis=1)

test_data
#modelling

y=train_data['price_range']


X=train_data.drop('price_range',axis=1)

X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
#===LOGISTIC REGRESSION=====

from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
logmodel.score(X_train,y_train)
logmodel.score(X_test,y_test)
#=====LINEAR REGRESSION=======

from sklearn.linear_model import LinearRegression

lm=LinearRegression()
lm.fit(X_train,y_train)
lm.score(X_train,y_train)
lm.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train,y_train)
knn.score(X_train,y_train)
knn.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=1000)

rfc.fit(X_train,y_train)
rfc.score(X_train,y_train)
rfc.score(X_test,y_test)
predictions=lm.predict(test_data)

predictions
RF_pred=rfc.predict(test_data)

RF_pred
test_data['price_range']=RF_pred
test_data