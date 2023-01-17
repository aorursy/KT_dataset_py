# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# 使用するライブラリをインポート

import matplotlib.pyplot as plt

import seaborn as sns



# データの読み込み

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
# データ総数を確認する

print(train.shape)

print(test.shape)
# 学習データの確認

train.head(10)
# テストデータの確認

test.head(10)
# 学習データの詳細の確認

train.describe()
# 学習データの欠損数の確認

train.isnull().sum()
# テストデータの欠損数の確認

test.isnull().sum()
# 生存フラグの度数分布と割合を確認する

print(train['Survived'].value_counts())



count = train.shape[0]

die, live = train['Survived'].value_counts()

print('死者', die / count * 100, end='%\n')

print('生存者', live / count * 100, end='%')
# Pclassごとの度数分布と生存率を確認する

print(train['Pclass'].value_counts())

train[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Pclass')
# Embarkedごとの度数分布と生存率を確認する

print(train['Embarked'].value_counts())

train[['Embarked', 'Survived']].groupby(['Embarked']).mean()
# Fareごとの度数分布を確認する

print(train['Fare'].value_counts())
# 金額をグループ分けして、新しく作ったカラム「FareBand」に挿入する (pd.qcutを使うとデータをソートして4分割してくれる, 分割数は第２引数で指定可能)

# (最小値, 最大値) の形でデータが挿入される

train['FareBand'] = pd.qcut(train['Fare'], 4)

train.head()
# FareBandの生存率を確認する (度数は均等にされる)

train[['FareBand', 'Survived']].groupby(['FareBand']).mean().sort_values(by='FareBand')
print(train['Ticket'].value_counts())
# チケット番号から文字列を削除して、100のグループに分割した

import re

ticketNumbers = []

for ticket in train['Ticket']:

    ticket = re.sub(r'([a-z]|[A-Z]|\s|[\.!-/:-@¥[-`{-~])', '', ticket)

    ticket = 0 if ticket == '' else ticket

    print(ticket)

    ticketNumbers.append(float(ticket))

    

train['TicketNumber'] = pd.qcut(ticketNumbers, 100)
# TicketNumberの生存率でグルーピングする

xs = train[['TicketNumber', 'Survived']].groupby(['TicketNumber']).mean().sort_values(by='TicketNumber')

index = xs['Survived'].index

values = xs['Survived'].values





# indexはタプルもどきの型になっているので、整数のindexに変換する

xs = []

for i in range(len(index)):

    xs.append(i)



# 可視化してみる

import matplotlib.pyplot as plt

plt.plot(xs, values)

plt.show()
# 性別の度数分布と生存率を確認

print(train['Sex'].value_counts())

train[['Sex', 'Survived']].groupby(['Sex']).mean()
# 分割しすぎるとデータ数に偏りがあるので、あまり多すぎないようにする

train['AgeBand'] = pd.qcut(train['Age'], 10)

xs = train[['AgeBand', 'Survived']].groupby(['AgeBand']).mean().sort_values(by='AgeBand')

print(xs)

index = xs['Survived'].index

values = xs['Survived'].values





# indexはタプルもどきの型になっているので、整数のindexに変換する

xs = []

for i in range(len(index)):

    xs.append(i)



# 可視化してみる

import matplotlib.pyplot as plt

plt.plot(xs, values)

plt.show()
print(train['SibSp'].value_counts())

train[['SibSp', 'Survived']].groupby(['SibSp']).mean()
print(train['Parch'].value_counts())

train[['Parch', 'Survived']].groupby(['Parch']).mean()
train['FamilySize'] = train['Parch'] + train['SibSp'] + 1

print(train['FamilySize'].value_counts())

train[['FamilySize', 'Survived']].groupby(['FamilySize']).mean()
train_survive0 = train[train['Survived'] == 0]

train_survive1 = train[train['Survived'] == 1]



# サブプロットを生成

figure, (Left, Right) = plt.subplots(ncols=2, figsize=(10, 4))



# 年齢を10分割する

Left.hist(train_survive0['Age'], bins=10)

Right.hist(train_survive1['Age'], bins=10)

figure.show()
# 学習データの欠損数の確認

print("学習データ\n", train.isnull().sum(), "\n")

print("テストデータ\n", test.isnull().sum())
# 学習データとテストデータの両方を埋める (種類で分けられてるのは最頻値)

for dataset in [train, test]:

    # fillna() で NaN の値を埋めることが出来る

    # mode()[0] で最頻値を取得出来る

    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])

    

# データが大きく変わっていないか確認する

train[['Embarked', 'Survived']].groupby(['Embarked']).mean()
for dataset in [train, test]:

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)



train.head()
# 金額は数値データなので、平均値か中央地で埋める (平均値は異常値に引っ張られるので中央地を利用)

test['Fare'] = test['Fare'].fillna(test['Fare'].median())
for dataset in [train, test]:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)
# 平均値か中央地で埋める (今回は平均値でやってみる)

for dataset in [train, test]:

    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())

train.head()
for dataset in [train, test]:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

    dataset['Age'].astype(np.int)
for dataset in [train, test]:

    dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1}).astype(int)
print("学習データ\n", train.isnull().sum(), "\n")

print("テストデータ\n", test.isnull().sum())
test['FamilySize'] = test['Parch'] + test['SibSp'] + 1
# 乗客IDと名前とチケットは使わない

# Cabinは欠損が多すぎるので使わない

# ShibSpとParchはFamilySizeに統合したので使わない

# AgeBandとFareBandは見やすくするために作ったので使わない

train = train.drop(['PassengerId', 'Name', 'Ticket', 'TicketNumber', 'Cabin', 'SibSp', 'Parch', 'AgeBand', 'FareBand'], axis=1)



# PassengerIdは削除しない

# AgeBandとFareBandは存在しない

test = test.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
# dropは指定したカラム以外のカラムを取得可能

X_train = train.drop(['Survived'], axis=1)

Y_train = train['Survived']

X_test = test.drop(['PassengerId'], axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# 今回はロジスティック回帰を使用

from sklearn.linear_model import LogisticRegression



# インスタンス化

logreg = LogisticRegression()



# fit() でデータを組み込む (学習データの説明変数と目的変数)

# この段階ですでに学習が完了している (logreg が学習済みモデルになる)

logreg.fit(X_train, Y_train)



# predict() は fit() で作成した学習モデルで答えを予測する

# Y_pred は predict の略で、予測した結果が変える (つまり、生きてるか死んでるかが変える)

Y_pred = logreg.predict(X_test)



print(Y_pred)
# スコアの確認を行う (正答率)

# この割合が高すぎると過学習している可能性があり、汎化誤差が大きくなることがある

# 60%とかの場合は単純に学習出来ていない, 100%の場合は過学習

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
# テストデータからPassengerIdを取り出してArrayに変換

passenger_id = np.array(test['PassengerId']).astype(int)

# DataFrame() は第一引数をデータフレーム化して、第二引数をindexとする (columnsはカラムのタイトル)

solution = pd.DataFrame(Y_pred, passenger_id, columns=["Survived"])

# csvファイルにエクスポートする

solution.to_csv("Logistic.csv", index_label=["PassengerId"])