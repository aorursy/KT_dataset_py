# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
train_data = pd.read_csv("../input/train.csv")

train_data.head(5)
test_data = pd.read_csv("../input/test.csv")

test_data.head(5)
train_data.describe()
test_data.describe()
train_data.info()
test_data.info()
train_data = train_data.astype({"Survived":'category', "Pclass":'category', "Sex":'category', "SibSp":'category', "Parch":'category', "Embarked":'category'})



test_data = test_data.astype({"Pclass":'category', "Sex":'category', "SibSp":'category', "Parch":'category', "Embarked":'category'})
train_data.isna().sum()
test_data.isna().sum()
test_data['Fare'] = test_data['Fare'].fillna(0)
#train_data.Age.mean()

#test_data.Age.mean()

train_data.Embarked.mode()
train_data['Age'] = train_data['Age'].fillna(29.69)
test_data['Age'] = test_data['Age'].fillna(30.27)
train_data['Embarked'] = train_data['Embarked'].fillna("S")
train_data.isnull().sum()
#train_data = train_data.dropna()

#test_data = test_data.dropna()
y_train = train_data["Survived"]

#X_train = train_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]

X_train = train_data[["Pclass", "Sex", "SibSp", "Fare", "Age", "Parch", "Embarked"]]
#y_test = test_data["Survived"]

#X_test = test_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]

X_test = test_data[["Pclass", "Sex", "SibSp", "Fare", "Age", "Parch", "Embarked"]]
test_data.shape
X_train_d = pd.get_dummies(X_train, columns=["Pclass", "Sex", "SibSp", "Parch", "Embarked"])
X_test_d = pd.get_dummies(X_test, columns=["Pclass", "Sex", "SibSp", "Parch", "Embarked"])
X_train_d.head()
X_test_d.head()
X_test_d = X_test_d.drop(['Parch_9'], axis=1)
print(X_train_d.shape)

print(X_test_d.shape)
scaler = StandardScaler().fit(X_train_d)

X_train_scale = scaler.transform(X_train_d)

X_train_scale = pd.DataFrame(X_train_scale)

X_train_scale.head()
X_test_d.head()
scaler1 = StandardScaler().fit(X_test_d)

X_test_scale = scaler1.transform(X_test_d)

X_test_scale = pd.DataFrame(X_test_scale)

X_test_scale.head()
sns.heatmap(X_train_d.corr())
y_train.head()
model = LogisticRegression()

model.fit(X_train_d, y_train)
model_s = LogisticRegression()

model_s.fit(X_train_scale, y_train)
model.coef_
model_s.coef_
print(X_test_d.shape)

print(y_pred_df.shape)
y_pred = model.predict(X_test_d)

y_pred_df = pd.DataFrame(y_pred, columns=["Survived"])

print(y_pred_df.head())
y_pred_scale = model.predict(X_test_scale)

y_pred_df_scale = pd.DataFrame(y_pred_scale, columns=["Survived"])

print(y_pred_df_scale.head())
print(y_pred_df.shape)

print(y_pred_df_scale.shape)
pred_df = pd.concat([test_data, y_pred_df], axis=1, sort=False)



pred_df_scale = pd.concat([test_data, y_pred_df_scale], axis=1, sort=False)
print(pred_df.shape)

print(pred_df_scale.shape)
pred_df.head()
pred_df_scale.head()
predictions = pred_df[["PassengerId", "Survived"]]

predictions.head()
predictions_s = pred_df_scale[["PassengerId", "Survived"]]

predictions_s.head()
predictions_s.to_csv("predictions_s_4.csv")