import numpy as np

import pandas as pd



from sklearn.preprocessing import MinMaxScaler # min:0 ~ max:1

from sklearn.preprocessing import StandardScaler # mean:0 variation:1



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import  GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier



import re

import itertools



from matplotlib import pyplot as plt

import seaborn as sns

sns.set()

sns.set_style('whitegrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# トレインデータ取り出し

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
# 各変数ごとの生存率を計算

cross_list = [] # 各変数ごとのcrosstab集計入れるリスト

cat = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

for i in range(len(cat)):

    cross = pd.crosstab(train_df[cat[i]], train_df['Survived'], margins=True)

    cross['survival_rate'] = cross[1] / cross['All']

    cross['survival_rate'] = cross['survival_rate'].round(2)

    cross_list.append(cross)

    

# 可視化

fig = plt.figure(figsize=(10,10))

plt.subplots_adjust(wspace=0.4, hspace=0.6)

for t in range(len(cat)):

    df = cross_list[t]

    ax = fig.add_subplot(3, 2, t+1)

    index = [str(x) for x in df.index]

    ax.bar(index, df['survival_rate'], width=0.3)

    ax.set_ylim(0, 1)

    ax.set_ylabel('survival rate')

    ax.set_title(df.index.name)

plt.show()
# Cabinのデータ確認

Cabin_data = train_df[train_df['Cabin'].notnull()]

Cabin_data['Cabin'][:5]
# 先頭のアルファベット取り出し

Cabin_data['Cabin_head'] = Cabin_data['Cabin'].apply(lambda x: x[0])



# Cabinの頭文字毎の生存率確認

count_df = Cabin_data.groupby(['Cabin_head'])['Survived'].value_counts().unstack().fillna(0)

count_df['survival_rate']= count_df[1] / (count_df[0]+count_df[1]) # 生存率計算



# 可視化

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.bar(count_df.index, count_df['survival_rate'])

ax.set_ylim(0,1)

ax.set_ylabel('survival rate')

ax.set_title('survival rate of Cabin')

plt.show()
# $512以上を$263に置き換え

train_df.replace({'Fare': {512.3292: 263.0000}}, inplace=True)
# 敬称から年齢推測する

# Mr., Mister, Mistress, Mr., Mrs.- 大人として14~最高年齢の平均値で埋める

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
# ver1. 3-2-2 categorical & categoricalの グラフより推測

# Pclass-Embarkedグラフより、class1はEmbarked"S"が６割占め、

# Sed-EmbarkedグラフよりfemaleでEmbark"S"が７割近く占めているので、二人共"S"で補う

train_df.loc[[61, 829], 'Embarked'] = "S"
# Cabinと、Pclass, Sex, SibSp, Parch, Embarkedの関係をグラフ化



# Cabinのカテゴリごとのcategorical変数値を集計

col_list = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

cross_list = []

for col in col_list:

    cross_list.append(pd.crosstab(Cabin_data['Cabin_head'], Cabin_data[col], normalize='index'))



# 可視化

fig = plt.figure(figsize=(20,20))

plt.subplots_adjust(wspace=0.4, hspace=0.6)

for t in range(len(cross_list)):

    ax = fig.add_subplot(3, 2, t+1)

    cross_list[t].plot.bar(stacked=True, ax=ax)

plt.show()
# boxplotで可視化して確認



col_list = ['Age', 'Fare']

fig = plt.figure(figsize=(10,5))

plt.subplots_adjust(wspace=0.4)

Cabin_dict = Cabin_data.groupby('Cabin_head').indices

x = 1 # グラフ表示位置用

for col in col_list:

    # boxplot用集計値作成

    element_list = []

    label_list = []

    for key, index in Cabin_dict.items():

        label_list.append(key)

        df = Cabin_data.iloc[index, Cabin_data.columns.get_loc(col)]

        element_list.append(list(df[df.notnull()]))

    # 可視化

    ax = fig.add_subplot(1,2,x)

    ax.boxplot(element_list, labels=label_list)

    ax.set_title('Cabin & '+ col)

    ax.set_ylabel(col)

    ax.set_xlabel('Cabin_head')

    x += 1

plt.show()
# null値を'N'で置き換え

train_df.fillna({'Cabin': 'N'}, inplace=True)
# 「3-1 目的変数(Survived)と各変数の関係チェック」で求めた変数毎の生存率を利用する。



# 変数毎にカテゴリの生存率(survival_rate)を求めて変数化する

cross_list = [] # 各変数ごとのcrosstab集計入れるリスト

cat = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

for i in range(len(cat)):

    cross = pd.crosstab(train_df[cat[i]], train_df['Survived'], margins=True)

    cross['survival_rate'] = cross[1] / cross['All']

    cross_list.append(cross)



# 変数化

surv_col = ['surv_Pclass', 'surv_Sex', 'surv_SibSp', 'surv_Parch', 'surv_Embarked'] # 変換後の値入れる列名リスト

for i in range(len(cross_list)):

    df = cross_list[i]

    cat_dic = df['survival_rate'].to_dict()



    train_df[surv_col[i]] = train_df[cat[i]].map(lambda x: cat_dic[x])



# Cabinの変数化

train_df['Cabin_initial'] = train_df['Cabin'].apply(lambda x: x[0]) # Cabin毎の頭文字取り出し

count_df.loc['N'] = 0.0 # 頭文字毎の生存率が入ったcount_dfに'N'データ追加

cabin_dic = count_df['survival_rate'].to_dict() # 生存率変換用辞書

# 頭文字毎に生存率を変数化

train_df['surv_Cabin'] = train_df['Cabin_initial'].map(lambda x: cabin_dic[x]).astype('float64')





train_df[['Survived', 'surv_Pclass', 'surv_Sex', 'surv_SibSp', 'surv_Parch', 'surv_Embarked','surv_Cabin']].head(5)
# one-hotエンコーディング作成関数

def one_hot_enc(df, dummy_list):

    for col in dummy_list:

        dummies = pd.get_dummies(df[col], prefix=col)

        df = pd.concat([df, dummies], axis=1)

    return df



# one-hotエンコーディング対象列

# cabinのone-hotエンコーディングを足したら、スコアが下がったので、cabinはone-hotから外す

dummy_list = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']



# one-hotエンコーディング適用

train_df = one_hot_enc(train_df, dummy_list)



# 確認

train_df.head(3)
# 年齢をビン分けして、生存率を計算

bins = [0, 4, 9, 14, 19, 29, 39, 49, 59, 69, 80]

su_age_df = train_df.loc[:,['Survived', 'Age']]

su_age_df['bins'] = pd.cut(su_age_df['Age'], bins)

su_age_cross = pd.crosstab(su_age_df['bins'], su_age_df['Survived'])

su_age_cross['All'] = su_age_cross.sum(axis=1)

su_age_cross['survival_rate'] = su_age_cross[1] / su_age_cross['All']



# ビン毎の生存率(live_rate)を、年齢ごとに当て込む。



# 年齢をビンわけして'age_binsへ

bins = [0, 4, 9, 14, 19, 29, 39, 49, 59, 69, 80]

train_df['age_bins'] = pd.cut(train_df['Age'], bins)



# ビン毎に該当生存値を"surv_Age"列に追加

age_dic = su_age_cross['survival_rate'].to_dict()

train_df['surv_Age'] = train_df['age_bins'].map(lambda x: age_dic[x]).astype('float64')



train_df.head(3)
# Fareを４つの四分位範囲のビンに分類して、生存率を計算

su_fa_df = train_df.loc[:, ['Survived', 'Fare']]

su_fa_df['bins'] = pd.qcut(su_fa_df['Fare'], 4)

su_fa_cross = pd.crosstab(su_fa_df['bins'], su_fa_df['Survived'])

su_fa_cross['All'] = su_fa_cross.sum(axis=1)

su_fa_cross['survival_rate'] = su_fa_cross[1] / su_fa_cross['All']



# ビンごとの生存率(survival_rate)を、Fare毎に当て込む。

# 乗船料を四分位範囲で４つにビンわけして'Fare_bins'へ

train_df['Fare_bins'] = pd.qcut(train_df['Fare'], 4)



# ビンごとに該当生存値を'repl_Fare'列に追加

fare_dic = su_fa_cross['survival_rate'].to_dict()

train_df['surv_Fare'] = train_df['Fare_bins'].map(lambda x: fare_dic[x]).astype('float64')



train_df[['Survived', 'Fare', 'Fare_bins', 'surv_Fare']].head(3)
# minmaxインスタンス

scaler = MinMaxScaler()



# scalerインスタンスのフィッティング

scaler = scaler.fit(train_df.Age.values.reshape(-1,1)) # fit()の引数は2次元アレイなので２次元に変換してわたす



# minmax変換

tr_age = scaler.transform(train_df.Age.values.reshape(-1,1)).reshape(891,)



# 確認

print("Age's min: ", tr_age.min())

print("Age's max: ", tr_age.max())



# train_dfに反映

train_df['min_max_Age'] = tr_age
# minmaxインスタンス

scaler2 = MinMaxScaler()



# scalerインスタンスのフィッティング

scaler2 = scaler2.fit(train_df.Fare.values.reshape(-1,1)) # fit()の引数は2次元アレイなので２次元に変換してわたす



# minmax変換

tr_fare = scaler2.transform(train_df.Fare.values.reshape(-1,1)).reshape(891,)



# 確認

print("Fare's min: ", tr_fare.min())

print("Fare's max: ", tr_fare.max())



# train_dfに反映

train_df['min_max_Fare'] = tr_fare
# インスタンス作成

scaler3 = StandardScaler()



# フィッティング

sclaler3 = scaler3.fit(train_df.Age.values.reshape(-1,1))



# 標準化

tr_train = scaler3.transform(train_df.Age.values.reshape(-1,1)).reshape(891,)



train_df['norm_Age'] = tr_train
# インスタンス作成

scaler4 = StandardScaler()



# フィッティング

sclaler4 = scaler4.fit(train_df.Fare.values.reshape(-1,1))



# 標準化

tr_train = scaler4.transform(train_df.Fare.values.reshape(-1,1)).reshape(891,)



train_df['norm_Fare'] = tr_train
# 敬称取り出し関数の作成

def get_title(name):

    title = ['Mister', 'Mistress', 'Master', 'Mrs\.',

             'Miss', 'Mr\.', 'Prof\.', 'Professor', 'Doctor'] # 敬称一覧

    for ti in title:

        if re.search(ti, name):

            return ti

    # 敬称ヒットしない時はotherを返す

    return 'other'

# 敬称取り出し

na_su_df = train_df.loc[: , ['Survived', 'Name']]

na_su_df['title'] = na_su_df['Name'].apply(get_title) # 敬称取り出し関数呼び出し

# 敬称ごとの生存率算出

na_su_cross = pd.crosstab(na_su_df['title'], na_su_df['Survived'], margins=True)

na_su_cross['survival_rate'] = na_su_cross[1] / na_su_cross['All']



# 敬称->生存値 変換辞書作成

title_dic = na_su_cross['survival_rate'].to_dict()

# 生存率設定

train_df['surv_Name'] = train_df['Name'].apply(get_title).map(lambda x: title_dic[x])



train_df[['Survived', 'surv_Name']].head(3)
train_df.columns
# 独立変数データ

# min-max

data_x = train_df.drop(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',

                        'Cabin_initial', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked',

                        'age_bins','Fare_bins', 'norm_Age', 'norm_Fare'], axis=1)

# nomlize

# data_x = train_df.drop(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',

#                         'Cabin_initial', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked',

#                         'age_bins','Fare_bins', 'min_max_Age', 'min_max_Fare'], axis=1)



# 従属変数データ

data_y = train_df[['Survived']]

# 訓練データとテストデータに分ける

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=1)
len(data_x.columns)
%%time



# パラメータ設定（とりあえず以下の設定で一番いい結果を選ぶ

param_grid = {'C': [0.001, 0.01, 1, 10,100,1000], 'random_state': [0], 'max_iter':[5000, 7000, 10000]}



# grid search で良さげなパラメータ見つける

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)

grid_search.fit(x_train.values, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test.values, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)
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

# learnig_rate:0.05, n_estimators:300に固定して、min_samples_splitとmax_depthを決める

# set n_estimators

param_grid = { 'learning_rate': [0.05],

                'n_estimators' : [300],

                'min_samples_split': [x for x in range(2,21,4)],

                'max_depth': [2, 4, 6, 8, 10]}



grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)

grid_search.fit(x_train.values, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test.values, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)
%%time



# layerサイズは、特徴量の数の３層と、neuralnetoのblogで出ていた数を比較

param_grid = {'hidden_layer_sizes': [(33,33,33,),(150,100,50)],

              'max_iter': [1000,1500,2000],

              'random_state': [0]}



grid_search = GridSearchCV(MLPClassifier(), param_grid, cv=5)

grid_search.fit(x_train.values, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test.values, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)

# testデータ取り出し

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
# 5の欠損値処理のコードをそのまま適用する



# 大人の平均年齢

adult_age = round(test_df[test_df['Age'] >= 14]['Age'].mean(), 1)

# 子供の平均年齢

child_age = round(test_df[test_df['Age'] <= 13]['Age'].mean(), 1)



# 欠損値穴埋め

# 大人

test_df.loc[test_df['Age'].isnull() & test_df['Name'].str.findall(r'.*(Mr\.|Mister|Mistress|Mr\.|Mrs\.|Ms\.).*'), 'Age'] = adult_age

# 子供

test_df.loc[test_df['Age'].isnull() & test_df['Name'].str.findall(r'.*(Master|Miss).*'), 'Age'] = child_age
# Fareが欠損値のデータ内容確認

test_df[test_df['Fare'].isnull()]
# Pclassが３なので、Pclass３の平均Fareで穴埋め

test_df.loc[152,'Fare'] = test_df.groupby('Pclass')['Fare'].mean()[3]

test_df.loc[152, 'Fare']
# Fareの外れ値対応として、263ドルより大きい値を263ドルに置き換える

test_df.loc[test_df[test_df['Fare'] > 263].index, 'Fare'] = 263
# 5の欠損値処理のコードをそのまま適用する



# null値を'N'で置き換え

test_df.fillna({'Cabin': 'N'}, inplace=True)

# Cabin毎の頭文字取り出し

test_df['Cabin_initial'] = test_df['Cabin'].apply(lambda x: x[0])

# 生存率変換用辞書

cabin_dic = count_df['survival_rate'].to_dict()

# 頭文字毎に生存率を変数化

test_df['surv_Cabin'] = test_df['Cabin_initial'].map(lambda x: cabin_dic[x]).astype('float64')



test_df['surv_Cabin'][:4]
# test_data と train_data のSibSpとParchのカテゴリが同じか確認



# SibSp

print("train_data's SibSp: ", train_df.groupby('SibSp').mean().index)

print("test_data's SibSp: ", test_df.groupby('SibSp').mean().index)

# ↓ SibSpのカテゴリは同じなので、6.変数変換のコードがそのまま使える
# Parch

print("train_data's Parch: ",train_df.groupby('Parch').mean().index)

print("test_data's Parch: ",test_df.groupby('Parch').mean().index)
# test_dataのParchには、9がある

# test_dateのParchの"9"のデータ内容確認

test_df[test_df['Parch'] == 9]
# train_dataのParchの"6"のデータ内容確認

train_df[train_df['Parch'] == 6]
# test_dataのParchの"9"は、"6"に置き換える



test_df.loc[test_df[test_df['Parch'] == 9].index, 'Parch'] = 6

test_df[test_df['Parch'] == 9]
# 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked' の変換

# cross_listは、6-1で作成したcross_listをそのまま利用する

cat = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

repl_col = ['surv_Pclass', 'surv_Sex', 'surv_SibSp', 'surv_Parch', 'surv_Embarked'] # 変換後の値入れる列名リスト

for i in range(len(cross_list)):

    df = cross_list[i]

    cat_dic = df['survival_rate'].to_dict()

    test_df[repl_col[i]] = test_df[cat[i]].map(lambda x: cat_dic[x])



# Age の変換

# 年齢をビンわけして'age_binsへ

bins = [0, 4, 9, 14, 19, 29, 39, 49, 59, 69, 80]

test_df['age_bins'] = pd.cut(test_df['Age'], bins)

# ビン毎に該当生存値を"repl_Age"列に追加

age_dic = su_age_cross['survival_rate'].to_dict()

test_df['surv_Age'] = test_df['age_bins'].map(lambda x: age_dic[x]).astype('float64')



# Fare の変換

bins_fare = [-0.001, 7.91, 14.454, 31.0, 263.0]

test_df['Fare_bins'] = pd.cut(test_df['Fare'], bins_fare)

# ビンごとに該当生存値を'repl_Fare'列に追加

fare_dic = su_fa_cross['survival_rate'].to_dict()

test_df['surv_Fare'] = test_df['Fare_bins'].map(lambda x: fare_dic[x]).astype('float64')



# 追加：Cabinの変換

test_df.fillna({'Cabin': 'N'}, inplace=True) # null値を'N'で置き換え

test_df['Cabin_initial'] = test_df['Cabin'].apply(lambda x: x[0]) # Cabin毎の頭文字取り出し

cabin_dic = count_df['survival_rate'].to_dict() # 生存率変換用辞書

# 頭文字毎に生存率を変数化

test_df['surv_Cabin'] = test_df['Cabin_initial'].map(lambda x: cabin_dic[x]).astype('float64')
# cabin以外のcategorical変数をone-hotエンコーディングする

# ver2のコードをそのまま利用



# test_dataのParch 9 のデータを Parch 6 に置き換える。

test_df.loc[test_df[test_df['Parch'] == 9].index, 'Parch'] = 6

# 対象変数取り出し

test_data = test_df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']]

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

# one-hotデータの連結

test_df = pd.concat([test_df, test_data], axis=1)

test_df.head(3)
# Age

# 年齢のビンごとにsurvival_rateを当て込む。

bins = [0, 4, 9, 14, 19, 29, 39, 49, 59, 69, 80]

test_df['age_bins'] = pd.cut(test_df['Age'], bins)

# ビン毎に該当survival_rateを"surv_Age"列に追加

age_dic = su_age_cross['survival_rate'].to_dict()

test_df['surv_Age'] = test_df['age_bins'].map(lambda x: age_dic[x]).astype('float64')



# Fare

# 乗船料のビンごとにsurvival_rateを当て込む

bins_fare = [-0.001, 7.91, 14.454, 31.0, 263.0]

test_df['Fare_bins'] = pd.cut(test_df['Fare'], bins_fare)

# ビンごとにsurvival_rateを"surv_Fare"列に追加

fare_dic = su_fa_cross['survival_rate'].to_dict()

test_df['surv_Fare'] = test_df['Fare_bins'].map(lambda x: fare_dic[x]).astype('float64')
# Age

# minmaxインスタンス

scaler = MinMaxScaler()

# フィッティング（フィッティングはtrain_dfで行う）

scaler = scaler.fit(train_df.Age.values.reshape(-1,1)) # fit()の引数は2次元アレイなので２次元に変換してわたす

# minmax変換

tr_age_test = scaler.transform(test_df.Age.values.reshape(-1,1)).reshape(418,)

# test_dfに反映

test_df['min_max_Age'] = tr_age_test



# Fare

# minmaxインスタンス

scaler2 = MinMaxScaler()

# フィッティング

scaler2 = scaler2.fit(train_df.Fare.values.reshape(-1,1)) # fit()の引数は2次元アレイなので２次元に変換してわたす

# minmax変換

tr_fare_test = scaler2.transform(test_df.Fare.values.reshape(-1,1)).reshape(418,)

# test_dfに反映

test_df['min_max_Fare'] = tr_fare_test
# Nameから変数創出

# 7-1.Nameから変数作成のコードをそのまま適用

title_dic = na_su_cross['survival_rate'].to_dict()

test_df['surv_Name'] = test_df['Name'].apply(get_title).map(lambda x: title_dic[x])
# モデル構築

# Logistic regression

Lr_model = LogisticRegression(C=10, max_iter=5000, random_state=0)

Lr_model.fit(data_x.values, np.ravel(data_y))

# Gradient boosting

GB_model = GradientBoostingClassifier(learning_rate=0.05, n_estimators=300)

GB_model.fit(data_x.values, np.ravel(data_y))



# 予測

test_data = test_df[data_x.columns] # testデータ取得(列を訓練データに揃える)

predict_Lr_result = Lr_model.predict(test_data.values) # Logistic regression予測

predict_GB_result = GB_model.predict(test_data.values) # Gradient boosting予測
# 提出用データの作成と保存

# Rogistic regression

predict_Lr_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived':predict_Lr_result})

predict_Lr_df.to_csv('/kaggle/working/submission_Lr_0327.csv', index=False)

# Gradient boosting

predict_GB_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived':predict_GB_result})

predict_GB_df.to_csv('/kaggle/working/submission_GB_0327.csv', index=False)