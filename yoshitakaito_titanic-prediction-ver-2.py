import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import re

import itertools



from matplotlib import pyplot as plt

import seaborn as sns

sns.set()

sns.set_style('whitegrid')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# トレインデータ取り出し

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

train_df.head(3)
# 敬称から年齢推測する

# Mr., Mister, Mistress, Mr., Mrs., Dr.- 大人として14~最高年齢の平均値で埋める

# Master, Miss - 男の子女の子とみなして0~13歳の子ども達の平均年齢を埋める (Misterが13歳までらしいのでとりあえず13歳で区切る)



# 大人の平均年齢

adult_age = round(train_df[train_df['Age'] >= 14]['Age'].mean(), 1)

# 子供の平均年齢

child_age = round(train_df[train_df['Age'] <= 13]['Age'].mean(), 1)



# 欠損値穴埋め

# 大人

train_df.loc[train_df['Age'].isnull() & train_df['Name'].str.findall(r'.*(Mr\.|Mister|Mistress|Mr\.|Mrs\.|Dr\.).*'), 'Age'] = adult_age

# 子供

train_df.loc[train_df['Age'].isnull() & train_df['Name'].str.findall(r'.*(Master|Miss).*'), 'Age'] = child_age



# 確認

train_df[train_df['Age'].isnull()]
# Pclass-Embarkedグラフより、class1はEmbarked"S"が６割占め、

# Sed-EmbarkedグラフよりfemaleでEmbark"S"が７割近く占めているので、二人共"S"で補う

train_df.loc[[61, 829], 'Embarked'] = "S"



# 確認

train_df.loc[[61, 829], :]
# 対象変数取り出し

train_data = train_df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare','SibSp', 'Parch', 'Embarked']]
# one-hotエンコーディング作成関数

def one_hot_enc(df, dummy_list):

    for col in dummy_list:

        dummies = pd.get_dummies(df[col], prefix=col)

        df = df.drop(col, axis=1)

        df = pd.concat([df, dummies], axis=1)

    return df



# one-hotエンコーディング対象列

dummy_list = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']



# one-hotエンコーディング適用

train_data = one_hot_enc(train_data, dummy_list)



# 確認

train_data.head(3)
from sklearn.preprocessing import MinMaxScaler # min:0 ~ max:1
# minmaxインスタンス

scaler = MinMaxScaler()



# scalerインスタンスのフィッティング

scaler = scaler.fit(train_data.Age.values.reshape(-1,1)) # fit()の引数は2次元アレイなので２次元に変換してわたす



# minmax変換

tr_age = scaler.transform(train_data.Age.values.reshape(-1,1)).reshape(891,)



# 確認

print("Age's min: ", tr_age.min())

print("Age's max: ", tr_age.max())



# train_dataに反映

train_data.Age = tr_age
# minmaxインスタンス

scaler2 = MinMaxScaler()



# scalerインスタンスのフィッティング

scaler2 = scaler2.fit(train_data.Fare.values.reshape(-1,1)) # fit()の引数は2次元アレイなので２次元に変換してわたす



# minmax変換

tr_fare = scaler2.transform(train_data.Fare.values.reshape(-1,1)).reshape(891,)



# 確認

print("Age's min: ", tr_fare.min())

print("Age's max: ", tr_fare.max())



# train_dataに反映

train_data.Fare = tr_fare
train_data.head(3)
import pandas as pd

import numpy as np



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
# 独立変数データ

data_x = train_data[train_data.columns[train_data.columns != 'Survived']]

# 従属変数データ

data_y = train_data[['Survived']]

# 訓練データとテストデータに分ける(testに３割は多いか、少ないか？)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression
%%time



# パラメータ設定（とりあえず以下の設定で一番いい結果を選ぶ

param_grid = {'C': [0.001, 0.01, 1, 10,100], 'random_state': [0], 'max_iter':[5000]}



# grid search で良さげなパラメータ見つける

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)

grid_search.fit(x_train.values, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test.values, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)
from sklearn.svm import SVC
%%time



# パラメータ設定（とりあえず以下の設定で一番いい結果を選ぶ)

param_grid = { 'kernel': ['rbf'],

                           'C' : [0.001, 0.01, 1, 10, 100,1000],

                           'gamma':[0.001, 0.01, 1, 10, 100,1000]}



# grid search で良さげなパラメータ見つける

grid_search = GridSearchCV(SVC(), param_grid, cv=5)

grid_search.fit(x_train.values, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test.values, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)
from sklearn.ensemble import  GradientBoostingClassifier
%%time



# パラメータセッティング１

# learning_rate と n_estimators のよさげな値を見つける

param_grid = { 'learning_rate': [0.05, 0.1, 0.2],

                           'n_estimators' : [50, 100, 200, 300, 400, 500]}



grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)

grid_search.fit(x_train.values, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test.values, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)
%%time



# パラメータセッティング２

# learnig_rate:0.1, n_estimators:50に固定して、min_samples_splitとmax_depthを決める

# set n_estimators

param_grid = { 'learning_rate': [0.1],

                'n_estimators' : [50],

                'min_samples_split': [x for x in range(2,21,4)],

                'max_depth': [2, 4, 6, 8, 10]}



grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)

grid_search.fit(x_train.values, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test.values, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)
# testデータ取り出し

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
# 5の欠損値処理のコードをそのまま適用する



# 敬称から年齢推測する

# Mr., Mister, Mistress, Mr., Mrs., Dr.- 大人として14~最高年齢の平均値で埋める

# Master, Miss - 男の子女の子とみなして0~13歳の子ども達の平均年齢を埋める (Misterが13歳までらしいのでとりあえず13歳で区切る)



# 大人の平均年齢

adult_age = round(test_df[test_df['Age'] >= 14]['Age'].mean(), 1)

# 子供の平均年齢

child_age = round(test_df[test_df['Age'] <= 13]['Age'].mean(), 1)



# 欠損値穴埋め

# 大人

test_df.loc[test_df['Age'].isnull() & test_df['Name'].str.findall(r'.*(Mr\.|Mister|Mistress|Mr\.|Mrs\.|Dr\.|Ms\.).*'), 'Age'] = adult_age

# 子供

test_df.loc[test_df['Age'].isnull() & test_df['Name'].str.findall(r'.*(Master|Miss).*'), 'Age'] = child_age



# 確認

test_df[test_df['Age'].isnull()]
# Pclass３の平均Fareで穴埋め

test_df.loc[152,'Fare'] = test_df.groupby('Pclass')['Fare'].mean()[3]
# test_dataのParch 9 のデータを Parch 6 に置き換える。

test_df.loc[test_df[test_df['Parch'] == 9].index, 'Parch'] = 6



# 対象変数取り出し

test_data = test_df[['Pclass', 'Sex', 'Age', 'Fare','SibSp', 'Parch', 'Embarked']]



# one-hotエンコーディング作成関数

def one_hot_enc(df, dummy_list):

    for col in dummy_list:

        dummies = pd.get_dummies(df[col], prefix=col)

        df = df.drop(col, axis=1)

        df = pd.concat([df, dummies], axis=1)

    return df



# one-hotエンコーディング対象列

dummy_list = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']



# one-hotエンコーディング適用

test_data = one_hot_enc(test_data, dummy_list)



# 確認

test_data.head(3)
# minmaxインスタンス

scaler = MinMaxScaler()



# scalerインスタンスのフィッティング（フィッティングはtrain_dataで行う）

scaler = scaler.fit(train_df.Age.values.reshape(-1,1)) # fit()の引数は2次元アレイなので２次元に変換してわたす



# minmax変換

tr_age_test = scaler.transform(test_data.Age.values.reshape(-1,1)).reshape(418,)



# test_dataに反映

test_data.Age = tr_age_test
# minmaxインスタンス

scaler2 = MinMaxScaler()



# scalerインスタンスのフィッティング

scaler2 = scaler2.fit(train_df.Fare.values.reshape(-1,1)) # fit()の引数は2次元アレイなので２次元に変換してわたす



# minmax変換

tr_fare_test = scaler2.transform(test_data.Fare.values.reshape(-1,1)).reshape(418,)



# test_dataに反映

test_data.Fare = tr_fare_test

test_data.head(3)
# train_dataでモデルを構築する

# Gradient Boosting: learning_rate=0.1, n_estimators=50



# 独立変数データ

data_x = train_data[train_data.columns[train_data.columns != 'Survived']]

# 従属変数データ

data_y = train_data[['Survived']]



# モデル構築

GB_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=50)

GB_model.fit(data_x.values, np.ravel(data_y))



# 予測

predict_result = GB_model.predict(test_data.values)
# 提出用データ作成と保存

predict_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predict_result})

predict_df.to_csv('/kaggle/working/submission_0325.csv', index=False)
# trainデータ用意



# トレインデータ取り出し

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')



# 欠損値穴埋め

# Age

adult_age = round(train_df[train_df['Age'] >= 14]['Age'].mean(), 1) # 大人の平均年齢

child_age = round(train_df[train_df['Age'] <= 13]['Age'].mean(), 1) # 子供の平均年齢

train_df.loc[train_df['Age'].isnull() & train_df['Name'].str.findall(r'.*(Mr\.|Mister|Mistress|Mr\.|Mrs\.|Dr\.).*'), 'Age'] = adult_age

train_df.loc[train_df['Age'].isnull() & train_df['Name'].str.findall(r'.*(Master|Miss).*'), 'Age'] = child_age

# Embarked

train_df.loc[[61, 829], 'Embarked'] = "S"



# one-hotエンコーディング

# 対象変数取り出し

train_data = train_df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare','SibSp', 'Parch', 'Embarked']]

# one-hotエンコーディング作成関数

def one_hot_enc(df, dummy_list):

    for col in dummy_list:

        dummies = pd.get_dummies(df[col], prefix=col)

        df = df.drop(col, axis=1)

        df = pd.concat([df, dummies], axis=1)

    return df

# one-hotエンコーディング対象列

dummy_list = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

# one-hotエンコーディング適用

train_data = one_hot_enc(train_data, dummy_list)
import sklearn.preprocessing as preproc
# 対象変数

data = train_data[['Fare', 'Age']]



# 交互作用特徴量作成

added_train = preproc.PolynomialFeatures(include_bias=False).fit_transform(data)



# 結果確認

added_train[0]
# テストデータと訓練データの用意



# 独立変数データ

add1_data = train_data.drop(['Survived', 'Age', 'Fare'], axis=1)

# 交互作用特徴量を元データに連結

add_train_data = np.hstack([add1_data.values, added_train])

# 従属変数データ

data_y = train_data[['Survived']].values



# 訓練データとテストデータに分ける(testに３割は多いか、少ないか？)

x_train, x_test, y_train, y_test = train_test_split(add_train_data, data_y, test_size=0.3, random_state=1)
%%time

# Logistic regression



# パラメータ設定（とりあえず以下の設定で一番いい結果を選ぶ

param_grid = {'C': [0.001, 0.01, 1, 10,100], 'random_state': [0], 'max_iter':[5000]}



# grid search で良さげなパラメータ見つける

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)

grid_search.fit(x_train, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)
%%time

# SVM ( rbf kernel)



# パラメータ設定（とりあえず以下の設定で一番いい結果を選ぶ)

param_grid = { 'kernel': ['rbf'],

                           'C' : [0.001, 0.01, 1, 10, 100,1000],

                           'gamma':[0.001, 0.01, 1, 10, 100,1000]}



# grid search で良さげなパラメータ見つける

grid_search = GridSearchCV(SVC(), param_grid, cv=5)

grid_search.fit(x_train, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)
%%time

# Gradient Boosting

# パラメータセッティング１

# learning_rate と n_estimators のよさげな値を見つける

param_grid = { 'learning_rate': [0.05, 0.1, 0.2],

                           'n_estimators' : [50, 100, 200, 300, 400, 500]}



grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)

grid_search.fit(x_train, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)
%%time



# パラメータセッティング２

# learnig_rate:0.05, n_estimators:50に固定して、min_samples_splitとmax_depthを決める

# set n_estimators

param_grid = { 'learning_rate': [0.05],

                'n_estimators' : [50],

                'min_samples_split': [x for x in range(2,21,4)],

                'max_depth': [2, 4, 6, 8, 10]}



grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)

grid_search.fit(x_train, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)
# スケールを揃えるため、min-max変換行う



scaler3 = MinMaxScaler() # インスタンス

scaler3 = scaler3.fit(added_train) # 交互作用特徴量をmin-max(0-1)変換

tr_data = scaler3.transform(added_train)
# テストデータと訓練データの用意



# 独立変数データ

add1_data = train_data.drop(['Survived', 'Age', 'Fare'], axis=1)

# 交互作用特徴量を元データに連結

add_train_data = np.hstack([add1_data.values, tr_data])

# 従属変数データ

data_y = train_data[['Survived']].values



# 訓練データとテストデータに分ける(testに３割は多いか、少ないか？)

x_train, x_test, y_train, y_test = train_test_split(add_train_data, data_y, test_size=0.3, random_state=1)
%%time

# Logistic regression



# パラメータ設定（とりあえず以下の設定で一番いい結果を選ぶ

param_grid = {'C': [0.001, 0.01, 1, 10,100], 'random_state': [0], 'max_iter':[5000]}



# grid search で良さげなパラメータ見つける

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)

grid_search.fit(x_train, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)
%%time

# SVM ( rbf kernel)



# パラメータ設定（とりあえず以下の設定で一番いい結果を選ぶ)

param_grid = { 'kernel': ['rbf'],

                           'C' : [0.001, 0.01, 1, 10, 100,1000],

                           'gamma':[0.001, 0.01, 1, 10, 100,1000]}



# grid search で良さげなパラメータ見つける

grid_search = GridSearchCV(SVC(), param_grid, cv=5)

grid_search.fit(x_train, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)
%%time

# Gradient Boosting

# パラメータセッティング１

# learning_rate と n_estimators のよさげな値を見つける

param_grid = { 'learning_rate': [0.05, 0.1, 0.2],

                           'n_estimators' : [50, 100, 200, 300, 400, 500]}



grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)

grid_search.fit(x_train, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)
# testデータ取り出し

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')



# Ageの欠損値処理

adult_age = round(test_df[test_df['Age'] >= 14]['Age'].mean(), 1) # 大人の平均年齢

child_age = round(test_df[test_df['Age'] <= 13]['Age'].mean(), 1) # 子供の平均年齢

test_df.loc[test_df['Age'].isnull() & test_df['Name'].str.findall(r'.*(Mr\.|Mister|Mistress|Mr\.|Mrs\.|Dr\.|Ms\.).*'), 'Age'] = adult_age

test_df.loc[test_df['Age'].isnull() & test_df['Name'].str.findall(r'.*(Master|Miss).*'), 'Age'] = child_age

# Fareの欠損値処理

test_df.loc[152,'Fare'] = test_df.groupby('Pclass')['Fare'].mean()[3]

# test_dataのParch 9 のデータを Parch 6 に置き換える。

test_df.loc[test_df[test_df['Parch'] == 9].index, 'Parch'] = 6



# one-hotエンコーディング

test_data = test_df[['Pclass', 'Sex', 'Age', 'Fare','SibSp', 'Parch', 'Embarked']] # 対象変数

def one_hot_enc(df, dummy_list): # one-hotエンコーディング作成関数

    for col in dummy_list:

        dummies = pd.get_dummies(df[col], prefix=col)

        df = df.drop(col, axis=1)

        df = pd.concat([df, dummies], axis=1)

    return df

dummy_list = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'] # one-hotエンコーディング対象列

test_data = one_hot_enc(test_data, dummy_list) # one-hotエンコーディング適用



# 交互作用特徴量作成

data = test_data[['Fare', 'Age']] # 対象変数

added_test = preproc.PolynomialFeatures(include_bias=False).fit_transform(data) # 交互作用特徴量作成



# 交互作用特徴量を元データに連結

add_test_data = np.hstack([test_data.drop(['Age', 'Fare'], axis=1).values, added_test])
# 訓練データの用意

add1_data = train_data.drop(['Survived', 'Age', 'Fare'], axis=1) # 独立変数データ

add_train_data = np.hstack([add1_data.values, added_train]) # 交互作用特徴量を元データに連結

data_y = train_data[['Survived']] # 従属変数データ



# モデリング

GB_model = GradientBoostingClassifier(learning_rate=0.05, n_estimators=50)

GB_model.fit(add_train_data, np.ravel(data_y))



# 予測

predict_result = GB_model.predict(add_test_data)



# 提出用データ作成と保存

predict_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predict_result})

predict_df.to_csv('/kaggle/working/submission_0325_2.csv', index=False)