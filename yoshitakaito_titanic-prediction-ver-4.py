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

    cross['survival_rate'] = cross[1] / cross['All'] # 生存率計算

    cross.sort_values('survival_rate', inplace=True)

    cross_list.append(cross)



# Cabinのイニシャル毎のsurvival_rate追加

# 先頭のアルファベット取り出し

Cabin_data = train_df[train_df['Cabin'].notnull()]

Cabin_data['Cabin_head'] = Cabin_data['Cabin'].apply(lambda x: x[0])

# Cabinの頭文字毎の生存率計算

count_df = Cabin_data.groupby(['Cabin_head'])['Survived'].value_counts().unstack().fillna(0)

count_df['survival_rate']= count_df[1] / (count_df[0]+count_df[1]) # 生存率計算

count_df.sort_values('survival_rate', inplace=True)

# cross_listに追加

cross_list.append(count_df)

# catリストに追加

cat.append('Cabin_head')



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

    ax.set_title(df.index.name + "'s survival rate")

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

su_age_cross['survival_rate'] = su_age_cross['survival_rate']

su_age_cross.sort_values('survival_rate', inplace=True)



# 可視化

fig = plt.figure(figsize=(10,7))

ax = fig.add_subplot(1, 1, 1)

index = [str(x) for x in su_age_cross.index]

ax.bar(index, su_age_cross['survival_rate'], width=0.6)

ax.set_ylim(0, 1)

ax.set_ylabel('survival rate')

ax.set_xlabel("Age's bin")

ax.set_title("survival rate of age's bin")

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

ax.set_title("survival rate of each Fare's bin")

plt.show()
# Name

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

na_su_cross.sort_values(by='survival_rate', inplace=True) # ソート



# 可視化

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

ax.bar(na_su_cross.index, na_su_cross['survival_rate'])

ax.set_ylim(0, 1)

ax.set_ylabel('survival rate')

ax.set_title('survival rate of title')

plt.show()
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
# ラベリング用辞書

Pclass_dic = {3: 0, 2: 1, 1: 3}

Sex_dic = {'male': 0, 'female': 1}

SibSp_dic = {5: 0, 8: 0, 4: 1, 3: 2, 0: 3, 2: 4, 1: 5}

Parch_dic = {4: 0, 6: 0, 5: 1, 0: 2, 2: 3, 1: 4, 3: 5}

Embarked_dic = {'S': 0, 'Q': 1, 'C': 2}

title_dic = {'Mr\.': 0, 'other': 1, 'Master': 2, 'Miss': 3, 'Mrs\.': 4}



# 変換

train_df['title'] = train_df['Name'].apply(get_title) # Nameからtitle取得

tar_col = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'title']

dic_list = [Pclass_dic, Sex_dic, SibSp_dic, Parch_dic, Embarked_dic, title_dic]

for col, dic in zip(tar_col, dic_list):

    col_name = 'lb_' + col

    train_df[col_name] = train_df[col].map(lambda x: dic[x])
# ビンわけ

age_bins = bins = [0, 4, 9, 14, 19, 29, 39, 49, 59, 69, 80]

train_df['Age_bins'] = pd.cut(train_df['Age'], age_bins)

train_df['Fare_bins'] = pd.qcut(train_df['Fare'], 4)

train_df['Age_bins'] = train_df['Age_bins'].astype(str)

train_df['Fare_bins'] = train_df['Fare_bins'].astype(str)



# ラベリング用辞書

Age_dic = {'(69, 80]': 0, '(59, 69]': 1, '(19, 29]': 2, '(39, 49]': 3,

           '(14, 19]': 4, '(49, 59]': 5, '(29, 39]': 6, '(9, 14]': 7,

           '(4, 9]': 8, '(0, 4]': 9}

Fare_dic = {'(-0.001, 7.91]': 0, '(7.91, 14.454]': 1, '(14.454, 31.0]': 2, '(31.0, 512.329]': 3}



# 変換

tar_col = ['Age_bins', 'Fare_bins']

dic_list = [Age_dic, Fare_dic]

for col, dic in zip(tar_col, dic_list):

    col_name = 'lb_' + col

    train_df[col_name] = train_df[col].map(lambda x: dic[x])
# 対象データ抽出

train_data = train_df[['lb_Pclass', 'lb_Sex', 'lb_SibSp', 'lb_Parch', 'lb_Embarked',

                       'lb_title', 'lb_Age_bins', 'lb_Fare_bins']]

# 標準化

scaler = StandardScaler()

scaler = scaler.fit(train_data.values)

stan_train_data = scaler.transform(train_data.values)
# 独立変数

data_x = stan_train_data

# 従属変数

data_y = train_df[['Survived']]

# 訓練データとテストデータに分ける

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=1)
%%time



# パラメータ設定（とりあえず以下の設定で一番いい結果を選ぶ

param_grid = {'C': [0.001, 0.01, 1, 10,100,1000], 'random_state': [0], 'max_iter':[5000, 7000, 10000]}



# grid search で良さげなパラメータ見つける

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)

grid_search.fit(x_train, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)
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
%%time



# パラメータセッティング

param_grid = { 'learning_rate': [0.05, 0.1, 0.2],

               'n_estimators' : [50, 100, 200, 300, 400, 500],

               'min_samples_split': [x for x in range(2,21,4)],

               'max_depth': [2, 4, 6, 8, 10]}



grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)

grid_search.fit(x_train, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)
%%time



# layerサイズは、特徴量の数の３層と、neuralnetoのblogで出ていた数を比較

param_grid = {'hidden_layer_sizes': [(8,8,8,),(150,100,50)],

              'max_iter': [1000,1500,2000],

              'random_state': [0]}



grid_search = GridSearchCV(MLPClassifier(), param_grid, cv=5)

grid_search.fit(x_train, np.ravel(y_train))



# モデリング結果

print('score: ',grid_search.score(x_test, np.ravel(y_test)))

print('best parameter: ', grid_search.best_params_)

print('score for train_date: ', grid_search.best_score_)