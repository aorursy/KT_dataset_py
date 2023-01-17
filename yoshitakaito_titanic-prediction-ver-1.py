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
# matplotとかでの日本語表示用フォントインストール

# !apt -y install fonts-ipafont fonts-ipaexfont
# 他のライブラリインストール

import re

import itertools



from matplotlib import pyplot as plt

from matplotlib import rcParams

# 日本語表示が不安定なので日本語表示一旦諦める。

# rcParams['font.family'] = 'IPAexGothic' # 日本語フォント設定

import seaborn as sns

sns.set()

sns.set_style('whitegrid')

# sns.set_palette('gray')

# ローデータ中身確認

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

train_df.head(3)
train_df.info()

# Age, Cabin, Embarked にnull値あり
# box plot で分布可視化



print(train_df[['Age', 'Fare']].describe()) # 数値

# 可視化

fig = plt.figure()

col = ['Age', 'Fare']

for t in range(len(col)):

    ax = fig.add_subplot(1, 2, t+1)

    ax.boxplot(train_df[train_df[col[t]].notnull()][col[t]])

    ax.set_xlabel(col[t])

plt.show()
# 変数ごとの度数分布を確認

# 積み上げ棒グラフで可視化



# 頻度の算出

count_list = [] # 各カテゴリカル変数の度数分布dfを入れるリスト

cat = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

for i in range(len(cat)):

    df = pd.DataFrame(train_df[cat[i]].value_counts())

    df['ratio'] = df[cat[i]] / df[cat[i]].sum()

    df['ratio'] = df['ratio'].round(2) # 小数点ふた桁

    count_list.append(df)

    

# 可視化

fig = plt.figure(figsize=(10,10))

plt.subplots_adjust(wspace=0.4, hspace=0.6) # グラフ間の余白設定

for t in range(len(cat)):

    df = count_list[t]

    ax = fig.add_subplot(3, 2, t+1)

    index = [str(x) for x in df.index ]# x軸のインデックス用に文字列化

    ax.bar(index, df['ratio'], width=0.3)

    ax.set_ylim(0, 1)

    ax.set_ylabel('Percent(%)')

    ax.set_title(df.columns[0])

plt.show()    
# Name

train_df['Name'].head(10)
# Cabin

train_df['Cabin'].value_counts()
# Ticket

train_df['Ticket'].value_counts()
# Pclass毎、Sex毎、SibSp毎、Parch毎、Embarked毎の各生存状況を確認



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

    # データ数が800もあるので、信頼区間は求めない。

    ax.bar(index, df['survival_rate'], width=0.3)

    ax.set_ylim(0, 1)

    ax.set_ylabel('survival rate')

    ax.set_title(df.index.name)

plt.show()
# Age(年齢)

# 以下のカテゴリでビンわけして生存率を確認



# 子供の生存確率が高いかもしれないので、子供のビンを５歳毎、２０歳以上は１０歳ごとにビンわけする。

bins = [0, 4, 9, 14, 19, 29, 39, 49, 59, 69, 80]

su_age_df = train_df.loc[:,['Survived', 'Age']]

su_age_df['bins'] = pd.cut(su_age_df['Age'], bins)

su_age_cross = pd.crosstab(su_age_df['bins'], su_age_df['Survived'])

su_age_cross['All'] = su_age_cross.sum(axis=1)

su_age_cross['survival_rate'] = su_age_cross[1] / su_age_cross['All']

su_age_cross['survival_rate'] = su_age_cross['survival_rate'].round(2)



# 可視化

fig = plt.figure(figsize=(10,7))

ax = fig.add_subplot(1, 1, 1)

index = [str(x) for x in su_age_cross.index]

ax.bar(index, su_age_cross['survival_rate'], width=0.6)

ax.set_ylim(0, 1)

ax.set_ylabel('survival rate')

ax.set_xlabel("Age's bin")

ax.set_title("survival ratio of age's bin")

plt.show()
# Fare(乗船料)



# ４つの四分位範囲のビンに分類して生存率を確認

su_fa_df = train_df.loc[:, ['Survived', 'Fare']]

su_fa_df['bins'] = pd.qcut(su_fa_df['Fare'], 4)

su_fa_cross = pd.crosstab(su_fa_df['bins'], su_fa_df['Survived'])

su_fa_cross['All'] = su_fa_cross.sum(axis=1)

su_fa_cross['survival_rate'] = su_fa_cross[1] / su_fa_cross['All']

su_fa_cross['survival_rate'] = su_fa_cross['survival_rate'].round(2)



# 可視化

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

index = [str(x) for x in su_fa_cross.index]

ax.bar(index, su_fa_cross['survival_rate'])

ax.set_ylim(0, 1)

ax.set_ylabel('survival rate')

ax.set_title("survival ratio of each Fare's bin")

plt.show()
# Name

# 敬称(Mr. Miss...)と生存率に関係があるか確認

# 女性の生存率が高いので、敬称も変数として利用できると思う。



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

na_su_cross['survival_rate'] = na_su_cross['survival_rate'].round(2)

na_su_cross = na_su_cross.sort_values(by='survival_rate', ascending=False) # ソート



# 可視化

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

ax.bar(na_su_cross.index, na_su_cross['survival_rate'])

ax.set_ylim(0, 1)

ax.set_ylabel('survival rate')

ax.set_title('survival ratio of title')

plt.show()



# 敬称補足

# Masterは７歳または１３までの男の子につける敬称。

# 参考：https://www.reference.com/world-view/age-master-become-mister-c54996c9aec19c32
# Cabin

train_df['Cabin'].isnull().value_counts()
# Ticket



# 生存者のチケットNoチェック

train_df[train_df['Survived'] == 1]['Ticket']
# 非生存者のチケットNoチェック

train_df[train_df['Survived'] == 0]['Ticket']
# 散布図を作成して確認

plt.figure()

plt.scatter(train_df['Age'], train_df['Fare'])

plt.xlabel('Age')

plt.ylabel('Fare')

plt.grid(True)
# crosstabで集計して積み上げ棒グラフで可視化して傾向見る



# ２要素毎にクロス集計

cross_list = []

fea_list = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

comb_list = list(itertools.combinations(fea_list, 2)) # 変数毎の組み合わせリスト作成

for comb in comb_list:

    cross_list.append(pd.crosstab(train_df[comb[0]], train_df[comb[1]], normalize='index')) # 集計結果をindex毎に正規化



# 可視化

fig = plt.figure(figsize=(20,20))

plt.subplots_adjust(wspace=0.4, hspace=0.6)

for t in range(len(cross_list)):

    ax = fig.add_subplot(4, 3, t+1)

    cross_list[t].plot.bar(stacked=True, ax=ax)

plt.show()
# boxplotで可視化して確認する



cat_list = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'] # categorical変数のリスト

con_list = ['Age', 'Fare'] # numeric変数のリスト



# cat_listとcon_listの要素の直積を求めてグループ化

comb_list = list(itertools.product(cat_list, con_list))

fig = plt.figure(figsize=(20,20))

plt.subplots_adjust(wspace=0.4, hspace=0.6)

x = 1 # グラフ表示位置用

for comb in comb_list:

    # boxplot用データ作成

    comb_df = train_df.loc[:, comb] # 対象列データ取り出し

    comb_dic = comb_df.groupby(comb[0]).indices

    label_list = []

    element_list = []

    for key, index in comb_dic.items():

        label_list.append(key)

        df = comb_df.loc[index, comb[1]]

        element_list.append(list(df[df.notnull()])) # null値除外(boxplot用)

    # 可視化

    ax = fig.add_subplot(5, 2, x)

    ax.boxplot(element_list, labels=label_list)

    ax.set_title(comb)

    ax.set_ylabel(comb[1])

    ax.set_xlabel(comb[0])

    x += 1

plt.show()
# train_df['Fare'].sort_values(ascending=False)[:5]
# $512以上を$263に置き換え

# train_df.replace({'Fare': {512.3292: 263.0000}}, inplace=True)
# 確認

# train_df['Fare'].sort_values(ascending=False)[:5]
# Ageが欠損値のデータ内容確認

age_null_df = train_df[train_df['Age'].isnull()]

print(len(age_null_df))

print(age_null_df.head(5))
# 敬称から年齢推測する

# Mr., Mister, Mistress, Mr., Mrs.- 大人として14~最高年齢の平均値で埋める

# Master, Miss - 男の子女の子とみなして0~13歳の子ども達の平均年齢を埋める (Misterが13歳までらしいのでとりあえず13歳で区切る)



# 大人の平均年齢

adult_age = round(train_df[train_df['Age'] >= 14]['Age'].mean(), 1)

# 子供の平均年齢

child_age = round(train_df[train_df['Age'] <= 13]['Age'].mean(), 1)



# 欠損値穴埋め

# 大人

train_df.loc[train_df['Age'].isnull() & train_df['Name'].str.findall(r'.*(Mr\.|Mister|Mistress|Mr\.|Mrs\.).*'), 'Age'] = adult_age

# 子供

train_df.loc[train_df['Age'].isnull() & train_df['Name'].str.findall(r'.*(Master|Miss).*'), 'Age'] = child_age
# 確認

train_df[train_df['Age'].isnull()]

# ↓ Dr.が残っていた
# Dr.には大人年齢で穴埋め

train_df.loc[766, 'Age'] = adult_age
# 再確認

train_df[train_df['Age'].isnull()]
# Embarkedが欠損値のデータ内容確認

train_df[train_df['Embarked'].isnull()]
# 3-2-2 categorical & categoricalの グラフより推測

# Pclass-Embarkedグラフより、class1はEmbarked"S"が６割占め、

# Sed-EmbarkedグラフよりfemaleでEmbark"S"が７割近く占めているので、二人共"S"で補う

train_df.loc[[61, 829], 'Embarked'] = "S"
# 確認

train_df.loc[[61, 829], :]
# 「3-1 目的変数(Survived)と各変数の関係チェック」で求めた変数毎の生存率を利用する。

# 各カテゴリの生存率を１００倍した値を重み付け値として、各カテゴリに当て込む。



# 変数毎にカテゴリの生存率を100倍した値を求める

cross_list = [] # 各変数ごとのcrosstab集計入れるリスト

cat = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

for i in range(len(cat)):

    cross = pd.crosstab(train_df[cat[i]], train_df['Survived'], margins=True)

    cross['生存値'] = cross[1] / cross['All']

    cross['生存値'] = cross['生存値'].round(2) * 100

    cross_list.append(cross)



# 生存率*100の値を変換後の値として、新しい列に入れる

repl_col = ['repl_Pclass', 'repl_Sex', 'repl_SibSp', 'repl_Parch', 'repl_Embarked'] # 変換後の値入れる列名リスト

for i in range(len(cross_list)):

    df = cross_list[i]

    cat_dic = df['生存値'].to_dict()



    train_df[repl_col[i]] = train_df[cat[i]].map(lambda x: cat_dic[x])



train_df[['Survived', 'repl_Pclass', 'repl_Sex', 'repl_SibSp', 'repl_Parch', 'repl_Embarked']].head(5)
# Ageの逆数を取り100倍して Age_verse 列に追加

train_df['Age_verse'] = round(1 / train_df['Age'] * 100 ,2)



train_df[['Survived', 'Age_verse']].head(10)
# 上のSurvivedとAge_verseの数字がイマイチな感じなので、boxplotで状況チェック

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.boxplot([train_df[train_df['Survived'] == 1]['Age_verse'],train_df[train_df['Survived'] == 0]['Age_verse']],

          labels=[1, 0])

ax.set_ylabel('Age_verse')

ax.set_xlabel('Survived')

plt.show()
# su_age_crossの確認

su_age_cross
# ビン毎の生存率(live_rate)を１００倍した値を重み付け値として、年齢ごとに当て込む。



# 生存値作成

su_age_cross['生存値'] = su_age_cross['survival_rate'] * 100



# 年齢をビンわけして'age_binsへ

bins = [0, 4, 9, 14, 19, 29, 39, 49, 59, 69, 80]

train_df['age_bins'] = pd.cut(train_df['Age'], bins)



# ビン毎に該当生存値を"repl_Age"列に追加

age_dic = su_age_cross['生存値'].to_dict()

train_df['repl_Age'] = train_df['age_bins'].map(lambda x: age_dic[x]).astype('float64')



train_df.head(5)
# Fareと他の変数のスケール確認

train_df[['Fare', 'repl_Pclass', 'repl_Sex', 'repl_SibSp', 'repl_Parch', 'repl_Embarked']].describe()
# ビンごとの生存率(survival_rate)を１００倍した値を重み付け値として、Fare毎に当て込む。

# su_fa_crossを利用する



# 生存値作成

su_fa_cross['生存値'] = su_fa_cross['survival_rate'] * 100



# 乗船料を四分位範囲で４つにビンわけして'Fare_bins'へ

train_df['Fare_bins'] = pd.qcut(train_df['Fare'], 4)



# ビンごとに該当生存値を'repl_Fare'列に追加

fare_dic = su_fa_cross['生存値'].to_dict()

train_df['repl_Fare'] = train_df['Fare_bins'].map(lambda x: fare_dic[x]).astype('float64')



train_df[['Survived', 'Fare', 'Fare_bins', 'repl_Fare']].head(10)
# 3-1-3 で求めた生存率df(na_su_cross)を利用する

# na_su_crossに"live_num"列追加

na_su_cross['live_num'] = round(na_su_cross['survival_rate'] * 100, 2)

# 敬称->生存値 変換辞書作成

title_dic = na_su_cross['live_num'].to_dict()

# 'title_live'に生存値を入れる

# get_title()関数は3-1-3で作成した関数

train_df['repl_Name'] = train_df['Name'].apply(get_title).map(lambda x: title_dic[x])



train_df[['Survived', 'repl_Name']].head(5)
# 調整したtrainデータの吐き出し



# 学習用データ： 

# 'repl_Fare', repl_Pclass', 'repl_Sex', 'repl_SibSp', 'repl_Parch', 'repl_Embarked', 'repl_Name', 'repl_Age'

# 上記８変数を特徴量としてモデリングしてみる



tuned_train_df = train_df[['Survived', 'repl_Fare', 'repl_Pclass', 'repl_Sex','repl_SibSp', 

                           'repl_Parch', 'repl_Embarked', 'repl_Name', 'repl_Age']]



tuned_train_df.head(3)



# 調整済み学習データの保存

tuned_train_df.to_csv('/kaggle/working/tuned_train_0324.csv')
import sklearn.preprocessing as preproc
# ペアワイズ交互作用特徴量の作成

# 各変数の値を掛けて新しい変数値として設定



# 対象変数

data = train_df[['repl_Fare', 'repl_Pclass', 'repl_Sex','repl_SibSp',

                 'repl_Parch', 'repl_Embarked', 'repl_Name', 'repl_Age']]

# 交互作用特徴量作成

added_train = preproc.PolynomialFeatures(include_bias=False).fit_transform(data)
# 結果確認

added_train[0]
# 各変数のスケールが違いすぎるので、上記の全変数を標準化してみる。

# また、上のまま進めると、ConvergenceWarning: lbfgs failed to converge (status=1):　

# STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.　のエラーがでて計算できなかった。



# 各変数を標準化

stand_train = preproc.scale(added_train)

stand_train[0]
import pandas as pd

import numpy as np



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
# 独立変数データ

# data_x = tuned_train_df[tuned_train_df.columns[tuned_train_df.columns != 'Survived']].values # 交互作用特徴量作成前用

# data_x = stand_train # 交互作用特徴量含む

# 従属変数データ

data_y = tuned_train_df[['Survived']]

# 訓練データとテストデータに分ける

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression
%%time



# パラメータ設定（とりあえず以下の設定で一番いい結果を選ぶ

param_grid = {'C': [0.001, 0.01, 1, 10,100], 'random_state': [0], 'max_iter':[5000]}



# grid search で良さげなパラメータ見つける

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)

grid_search.fit(x_train, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test, np.ravel(y_test)))

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

grid_search.fit(x_train, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)
from sklearn.ensemble import  GradientBoostingClassifier
%%time



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
# testデータ取り出し

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

test_df.head(3)
test_df.info()


# 5の欠損値処理のコードをそのまま適用する



# 大人の平均年齢

adult_age = round(test_df[test_df['Age'] >= 14]['Age'].mean(), 1)

# 子供の平均年齢

child_age = round(test_df[test_df['Age'] <= 13]['Age'].mean(), 1)



# 欠損値穴埋め

# 大人

test_df.loc[test_df['Age'].isnull() & test_df['Name'].str.findall(r'.*(Mr\.|Mister|Mistress|Mr\.|Mrs\.).*'), 'Age'] = adult_age

# 子供

test_df.loc[test_df['Age'].isnull() & test_df['Name'].str.findall(r'.*(Master|Miss).*'), 'Age'] = child_age
# 確認

test_df[test_df['Age'].isnull()]

# ↓ Ms.が残っていた
# MS.には大人年齢で穴埋め

test_df.loc[88, 'Age'] = adult_age

test_df[test_df['Age'].isnull()]
# Embarkedが欠損値のデータ内容確認

test_df[test_df['Fare'].isnull()]
# Pclassが３なので、Pclass３の平均Fareで穴埋め

test_df.loc[152,'Fare'] = test_df.groupby('Pclass')['Fare'].mean()[3]

test_df.loc[152, 'Fare']
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

# cross_listは、6-1で作成したcross_listwをそのまま利用する

cat = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

repl_col = ['repl_Pclass', 'repl_Sex', 'repl_SibSp', 'repl_Parch', 'repl_Embarked'] # 変換後の値入れる列名リスト

for i in range(len(cross_list)):

    df = cross_list[i]

    cat_dic = df['生存値'].to_dict()

    test_df[repl_col[i]] = test_df[cat[i]].map(lambda x: cat_dic[x])



# Age の変換

# 年齢をビンわけして'age_binsへ

bins = [0, 4, 9, 14, 19, 29, 39, 49, 59, 69, 80]

test_df['age_bins'] = pd.cut(test_df['Age'], bins)

# ビン毎に該当生存値を"repl_Age"列に追加

age_dic = su_age_cross['生存値'].to_dict()

test_df['repl_Age'] = test_df['age_bins'].map(lambda x: age_dic[x]).astype('float64')



# Fare の変換

bins_fare = [-0.001, 7.91, 14.454, 31.0, 512.3292] # 6-2で分けたビン

test_df['Fare_bins'] = pd.cut(test_df['Fare'], bins_fare)

# ビンごとに該当生存値を'repl_Fare'列に追加

fare_dic = su_fa_cross['生存値'].to_dict()

test_df['repl_Fare'] = test_df['Fare_bins'].map(lambda x: fare_dic[x]).astype('float64')
# Nameから変数創出

# 7-1.Nameから変数作成のコードをそのまま適用

title_dic = na_su_cross['live_num'].to_dict()

test_df['repl_Name'] = test_df['Name'].apply(get_title).map(lambda x: title_dic[x])
# 7-2のコードをそのまま利用



# ペアワイズ交互作用特徴量の作成



# 対象変数

data = test_df[['repl_Fare', 'repl_Pclass', 'repl_Sex','repl_SibSp',

                 'repl_Parch', 'repl_Embarked', 'repl_Name', 'repl_Age']]

# 交互作用特徴量作成

added_test = preproc.PolynomialFeatures(include_bias=False).fit_transform(data)



# 各変数を標準化

stand_test = preproc.scale(added_test)
# 調整済みtest_dataの吐き出し

test_df.to_csv('/kaggle/working/test_0324.csv')

np.savetxt('/kaggle/working/stand_test.csv', stand_test)
# train_data でモデルを構築する

# Logsitic regresion C=1



data_y = tuned_train_df[['Survived']]



Lr_model = LogisticRegression(C=1, max_iter=5000, random_state=0)

Lr_model.fit(stand_train, np.ravel(data_y))



# 予測

predict_result = Lr_model.predict(stand_test)
# 提出用データ作成と保存

predict_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predict_result})

predict_df.to_csv('/kaggle/working/submission_0324.csv', index=False)