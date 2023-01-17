# python 라이브러리 호출

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pandas import Series, DataFrame

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from pandas_summary import DataFrameSummary

import time

from sklearn.svm import LinearSVC

from sklearn.preprocessing import StandardScaler
# 데이터 불러오기

path = '../input/train.csv'

path2 = '../input/test.csv'

path3 = '../input/gender_submission.csv'

train = pd.read_csv(path)

test = pd.read_csv(path2)

submission = pd.read_csv(path3)
train[:5]
test[:5]
DataFrameSummary(train).columns_stats
DataFrameSummary(test).columns_stats
categorical_features = ['Pclass',

                        'Sex',

                        'SibSp',

                        'Parch',

                        'Embarked']



for feature in categorical_features:

        df = pd.crosstab(train['{}'.format(feature)], train.Survived)

        df.plot(kind='bar', title = 'Survive by {}'.format(feature), rot=0)
fig, axes = plt.subplots(2, 1, figsize=(12, 12))



train.isna().sum().sort_values(ascending=True).plot(ax=axes[0], kind='barh',color='blue', fontsize=10)

axes[0].set_title('Missing values in train dataset')



test.isna().sum().sort_values(ascending=True).plot(ax=axes[1], kind='barh',color='red', fontsize=10)

axes[1].set_title('Missing values in test dataset')
# 결측치 제거

train.Age.dropna(inplace=True)

test.Age.dropna(inplace=True)
# 연속형 변수의 분포 시각화

fig, axes = plt.subplots(2,1, figsize=(10,5))



train.Fare.plot(kind='kde', ax = axes[0], title= 'Fare')

train.Age.plot(kind='kde', ax = axes[1], title= 'Age')
# 데이터 초기화

train = pd.read_csv(path)

test = pd.read_csv(path2)

submission = pd.read_csv(path3)
# Name's title

# Age와 Name을 이용해서 Age의 NaN값을 대체하기 위한 정규표현식 이용하기.

train['Initial'] = train.Name.str.extract('([A-Za-z]+)\.')

test['Initial'] = test.Name.str.extract('([A-Za-z]+)\.')
# Initial에 따른 Age의 평균값으로 NaN값 대체

train['Age'] = train.groupby('Initial')['Age'].apply(lambda x: x.fillna(x.mean()))



# 학습 데이터의 평균값

Age_mean = train.groupby('Initial')['Age'].mean()



# test에도 동일하게 적용하지만, 학습 데이터의 평균값을 대입시켜 줄 것.

for name in set(train['Initial'].values):

    form_ = '{}'.format(name)

    test.loc[test['Initial'] == form_, 'Age'] = test.loc[test['Initial'] == form_, 'Age'].fillna(Age_mean[form_])
test[test.Age.isnull()]
# 학습 데이터 Fare의 평균값

Fare_mean = train.Fare.mean()



# NaN값 대체

train[train.Fare.isnull()] = Fare_mean

test[test.Fare.isnull()] = Fare_mean
# Sex, Embarked 

train["Sex"]=np.where(train["Sex"]=="male",0,1)

train["Embarked"]=np.where(train["Embarked"]=="S",0,

                                  np.where(train["Embarked"]=="C",1,

                                           np.where(train["Embarked"]=="Q",2,3)

                                          )

                                 )

test["Sex"]=np.where(test["Sex"]=="male",0,1)

test["Embarked"]=np.where(test["Embarked"]=="S",0,

                                  np.where(test["Embarked"]=="C",1,

                                           np.where(test["Embarked"]=="Q",2,3)

                                          )

                                 )
# Fare

train['Fare'] = train['Fare'].apply(lambda i: np.log(i) + 1 if i > 0 else 0)

test['Fare'] = test['Fare'].apply(lambda i: np.log(i) + 1 if i > 0 else 0)



train.Fare.plot(kind='kde', title= 'log Fare')
# scaler

scaler = StandardScaler()

# fitting

scaler.fit(train[['Fare', 'Age']])



# transforming

train[['Fare', 'Age']] = scaler.transform(train[['Fare', 'Age']])

test[['Fare', 'Age']] = scaler.transform(test[['Fare', 'Age']])
# SibSp + Parch + 1 = Family

train['Family'] = train['SibSp'] + train['Parch'] + 1

test['Family'] = test['SibSp'] + test['Parch'] + 1
train[['Family', 'Survived']].groupby(by='Family').agg('mean').plot(kind= 'bar', rot=0)

plt.title('Survive by Family')
used_features =[

    "Pclass",

    "Sex",

    "Embarked",

    "Age",

    "Fare",

    "Family"]
# Split dataset in training and test datasets

X_train, X_val, y_train, y_val = train_test_split(train[used_features], 

                                                    train.Survived, test_size=0.4, stratify = train['Survived'])
svc = LinearSVC(max_iter=3000)

svc.fit(X_train, y_train)

print("{}".format(svc))

print("train set 정확도: {:.2f}".format(svc.score(X_train, y_train)))

print("val set 정확도: {:.2f}\n".format(svc.score(X_val, y_val)))
# parameter

params = {'loss': 'squared_hinge',

          'penalty': 'l2',

          'C': 1.0,

          'max_iter':10000

          }



svc.set_params(**params)

print("{}\n".format(svc))

print("train set 정확도: {:.2f}".format(svc.score(X_train, y_train)))

print("val set 정확도: {:.2f}\n".format(svc.score(X_val, y_val)))
svc.set_params(**params)

svc.fit(X_train, y_train)

y_pred = svc.predict(test[used_features])
# submission의 Survived column을 예측한 값으로 대체하기

submission.Survived = y_pred

submission.Survived = submission.Survived.astype(int)
# DataFrame -> csv 파일 저장

submission.to_csv('gender_submission.csv', index=False)