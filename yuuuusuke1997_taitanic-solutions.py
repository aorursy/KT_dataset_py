# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

combine = [train_df, test_df]
print(train_df.columns.values)
train_df.head()
test_df.head()
train_df.tail()
train_df.info()

print('_'*40)

test_df.info()
train_df.describe()
# .describe(include='O')でオブジェクトデータ（str）に関する情報取得



# count=データ数, unique=重複を排除したデータ数

# top=最も多く含まれるデータ, freq=そのデータが含まれる個数



train_df.describe(include=['O'])
# ascending=False→降順

# as_index=False→インデックスなし



train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# https://qiita.com/hik0107/items/865b75ae486728cb0006



g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
# col='Pclass'→Pclass属性ごとグラフを描写

# hue='Surived'→'Survived'の色の変更



grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid.map(plt.hist, 'Age', alpha=.5, bins=30)

grid.add_legend()
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)



# palette=色の変更

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
# データセット(行数)=891, 481

# カラム数(列数)=11, 10



# .shape→pandasデータフレームの列数と行数を調べる

print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
for dataset in combine:

    # expand=False→DataFrameとして取得

    # str.extract()→正規表現で分割

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



# pandas.crosstab(index, columns)→カテゴリデータのカテゴリごとのサンプル数（出現回数・頻度）を算出

pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    

    # https://note.nkmk.me/python-pandas-map-replace/

    # map()の引数に辞書（dict）を指定

    # →'Title'と'title_mapping'のkeyが一致する要素がvalueに置換される

    dataset['Title'] = dataset['Title'].map(title_mapping)

    

    # valueがない場合

    # fillna()→欠損値を0で置換

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()
# NameとPassengerIDを削除

# df.drop(name(削除する項目), axis=1)→列を削除



train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]

train_df.shape, test_df.shape
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
# 空リスト作成

guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine:

    print(dataset.isnull().any())
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            # 'Sex'と'Pclass'のに含まれる欠損値を削除

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            

            # 平均

            # age_mean = guess_df.mean()

            # 標準偏差

            # age_std = guess_df.std()

            # rnd.uniform(〇,〇)→〇～〇の間の乱数を取得

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            # 中央値

            age_guess = guess_df.median()

            

            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

    

    for i in range(0, 2):

        for j in range(0, 3):

            # datasetのAgeにある欠損値、datasetのSex(男女)、datasetのPclass(1~3)に中央値を追加

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

    

    # datasetのAgeをfloatをintに変更

    dataset['Age'] = dataset['Age'].astype(int)



train_df.head()
int(21.5/0.5 + 0.5 ) * 0.5
for dataset in combine:

    print(dataset.isnull().any())
# http://ailaby.com/cut_qcut/

# ビニング処理（ビン分割）とは、連続値を任意の境界値で区切りカテゴリ分けして離散値に変換する処理のこと。



# 境界値を指定して分類する

# pd.cut(元データ, bins(分割数))

#第二引数がq=5の場合、train_df['Age']を5分割する

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



# ascending=False→降順に並び替え

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    # datasetのFamilySizeが1でcolumnがIsAloneのとき,１を代入

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]



train_df.head()
for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
# mode()→最頻値を取得

freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    # Embarkedの最頻値、'S'で補完する

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)



train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    # map({})で引数に辞書dictを指定すると、置換される

    # そして置換した要素は、astype(int)でint型に指定

    # ※replace({})でも置換できるが、要素の値がない場合、

    # map()→Nan, replace()→元の値のままになる

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

train_df.head()
# Fareの欠損値を除外した中央値をFaraに補完

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
# 境界値を指定して分類する

# pd.cut(元データ, bins(分割数))

#第二引数がq=5の場合、train_df['Age']を5分割する

# train_df['AgeBand'] = pd.cut(train_df['Age'], 5)



# 値の大きさ順にn等分する

# pd.qcut(元データ, 等分数))

#第二引数がq=2の場合、中央値で分割。

#第二引数がq=4の場合、四分位数で分割。

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]



train_df.head(10)
test_df.head(10)
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

# test_dfのPassengerIDの列を削除して、元のデータに変更を反映させないためコピーする

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
test_df.shape
# ロジスティック回帰モデルのインスタンスを作成

logreg = LogisticRegression()



# 訓練で重みを学習、fitにデータを渡す

logreg.fit(X_train, Y_train)



# 訓練したデータにテストデータを当てはめる

# 説明変数の値からクラスを予測、.predict()

Y_pred = logreg.predict(X_test)



# 精度の確認はscore()

# 重みを指定し、スコアの小数点第二位まで確認

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
# train_dfの'Survived'の列を削除し、coeff_dfとして作成

coeff_df = pd.DataFrame(train_df.columns.delete(0))



coeff_df.columns = ['Feature']



# .coef_でfitした係数を確認

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_ml/py_svm/py_svm_basics/py_svm_basics.html#svm-understanding

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
# http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_ml/py_knn/py_knn_understanding/py_knn_understanding.html

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })



submission.to_csv('submission.csv')