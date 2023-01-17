import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import style

import seaborn as sns
train = pd.read_csv("../input/genpact/train.csv")

test=pd.read_csv("../input/genpact/test.csv")
train.head()
train.shape
center_info=pd.read_csv("../input/genpact/fulfilment_center_info.csv")
center_info.head()
center_info.shape
meal_info=pd.read_csv("../input/genpact/meal_info.csv")
meal_info.head()
meal_info.shape
train.isnull().sum()
#No null values in train dataset

center_info.isnull().sum()
meal_info.isnull().sum()
#No null values in any dataframe

#Now we will merge our datasets

train_new=pd.merge(train, center_info, on='center_id',how='outer')



train_new=pd.merge(train_new, meal_info,on='meal_id',how='outer')
train_new = train_new.drop(['center_id', 'meal_id'], axis=1)
train_new.head()
train_new.shape
cols = train_new.columns.tolist()

cols = cols[:2] + cols[9:] + cols[7:9] + cols[2:7]

train_new= train_new[cols]
from sklearn.preprocessing import LabelEncoder
lb1 = LabelEncoder()

lb2 = LabelEncoder()

lb3 = LabelEncoder()



train_new['center_type'] = lb1.fit_transform(train_new['center_type'])



train_new['category'] = lb2.fit_transform(train_new['category'])



train_new['cuisine'] = lb3.fit_transform(train_new['cuisine'])
train_new.dtypes
train_new.head()
plt.style.use('fivethirtyeight')

plt.figure(figsize=(12,7))

sns.distplot(train_new.num_orders, bins = 25)

plt.xlabel("num_orders")

plt.ylabel("Number of Buyers")

plt.title("num_orders Distribution")
#We need to reduce these num order values as they are ranging to a very high extent.

import math

def reciprocal(x):

    y = 1/x

    return y



def log(x):

    y = math.log(x, 10)

    return y
hehe = reciprocal(train_new.num_orders)
haha['numorders'] = log(train_new.num_orders)
plt.style.use('fivethirtyeight')

plt.figure(figsize=(12,7))

sns.distplot(hehe, bins = 25)

plt.xlabel("num_orders")

plt.ylabel("Number of Buyers")

plt.title("num_orders Distribution")
correlation = df.corr(method='pearson')

columns = correlation.nlargest(8, 'num_orders').index

columns
train2 = train_new.drop(['id'], axis=1)

correlation = train2.corr(method='pearson')

columns = correlation.nlargest(8, 'num_orders').index

columns
correlation_map = np.corrcoef(train2[columns].values.T)

sns.set(font_scale=1.0)

heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns.values, xticklabels=columns.values)



plt.show()
features = columns.drop(['num_orders'])

train3 = train_new[features]

X = train3.values

y = train_new['num_orders'].values



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
from sklearn.linear_model import Lasso

L = Lasso() 

L.fit(X_train, y_train) 

y_pred = L.predict(X_val) 

y_pred[y_pred<0] = 0 

from sklearn import metrics 

print('RMSLE:', np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))
testfinal = pd.merge(test, meal_info, on="meal_id", how="outer")

testfinal = pd.merge(testfinal, center_info, on="center_id", how="outer")

testfinal = testfinal.drop(['meal_id', 'center_id'], axis=1)



tcols = testfinal.columns.tolist()

print(tcols)
tcols = tcols[:2] + tcols[8:] + tcols[6:8] + tcols[2:6]

testfinal = testfinal[tcols]



lb1 = LabelEncoder()

testfinal['center_type'] = lb1.fit_transform(testfinal['center_type'])



lb2 = LabelEncoder()

testfinal['category'] = lb1.fit_transform(testfinal['category'])



lb3 = LabelEncoder()

testfinal['cuisine'] = lb1.fit_transform(testfinal['cuisine'])



testfinal.head()
X_test = testfinal[features].values
features
pred = L.predict(X_test)

pred[pred<0] = 0

submit = pd.DataFrame({

    'id' : testfinal['id'],

    'num_orders' : pred

})
pred.shape
submit.to_csv("submission.csv", index=False)