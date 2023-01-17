#Import Library

import numpy as np #linear algebra

import pandas as pd #data processing

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
#Path file

path_train = '../input/seleksidukungaib/train.csv'

path_test = '../input/seleksidukungaib/test.csv'



#read file csv

train_data = pd.read_csv(path_train)

test_data = pd.read_csv(path_test)
#check data

train_data.head()
#Summary of the train_data

train_data.describe()
train_data.shape
train_data['isChurned'].value_counts()
#Check for NaN

train_data.isna().sum()
#drop Na data from train_data

train_data = train_data.dropna(axis=0)
#cek Na data

train_data.isna().sum()
sns.countplot(x='premium', hue='isChurned',data = train_data)
sns.countplot(x='isVerifiedEmail', hue='isChurned',data = train_data)
sns.countplot(x='isVerifiedPhone', hue='isChurned',data = train_data)
sns.countplot(x='blocked', hue='isChurned',data = train_data)
numerical_features = ['num_topup_trx', 'num_recharge_trx']

fig, ax = plt.subplots(1, 2, figsize=(28, 8))

train_data[train_data.isChurned == 0][numerical_features].hist(bins=20, color="blue", alpha=0.5, ax=ax)

train_data[train_data.isChurned == 1][numerical_features].hist(bins=20, color="orange", alpha=0.5, ax=ax)
data = pd.concat([train_data,test_data],ignore_index=True)
#Label encod true/false and date to int

date = ['date_collected','date']

bin = ['premium','super','pinEnabled']

col = date + bin

le = LabelEncoder()

for i in col: 

    data[i] = le.fit_transform(list(data[i].values))
data.head()
train_data = data[~data.isChurned.isnull()]

test_data = data[data.isChurned.isnull()]
features = ['date_collected','num_topup_trx', 'num_recharge_trx','isActive','isVerifiedEmail','blocked','premium','super','userLevel','pinEnabled']

X = train_data[features]

y = train_data['isChurned']
#use logisticregression

model = LogisticRegression()

model.fit(X,y)
#Prediction

pred = model.predict(test_data[features])
submission = pd.DataFrame({'idx':test_data['idx'],'isChurned':pred.astype(int)})

submission.to_csv('submission.csv',index=False)
#cek submission

submission.head()