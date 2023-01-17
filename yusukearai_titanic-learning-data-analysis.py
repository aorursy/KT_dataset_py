# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#データの読み込みとデータのNAN値とデータ型把握

import numpy as np

import pandas as pd



# train_xは学習データ、train_yは目的変数、test_xはテストデータ

# pandasのDataFrame, Seriesで保持します。（numpyのarrayで保持することもあります）

train = pd.read_csv('../input/titanic-data/train.csv')

train_x = train.drop(['Survived'], axis=1)

train_y = train['Survived']

test_x = pd.read_csv('../input/titanic-data/test.csv')



print(train_x)

print()

print(test_x)



#全データのNAN値とデータ型を確認-NAN値がある列【Age, Cabin, Embarked, Fare(test)】

train_x.info()

test_x.info()
#pandas_profilingでデータ概要の把握

import numpy as np

import pandas as pd

import pandas_profiling



#データの読み込み

# train_xは学習データ、train_yは目的変数、test_xはテストデータ

# pandasのDataFrame, Seriesで保持します。（numpyのarrayで保持することもあります）

train = pd.read_csv('../input/titanic-data/train.csv')

train_x = train.drop(['Survived'], axis=1)

train_y = train['Survived']

test_x = pd.read_csv('../input/titanic-data/test.csv')



train.profile_report()

#seabornのpairplot関数でデータの分布と外れ値の把握

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



#データの読み込み

train = pd.read_csv('../input/titanic-data/train.csv')

train_x = train.drop(['Survived'], axis=1)

train_y = train['Survived']

test_x = pd.read_csv('../input/titanic-data/test.csv')



cols = ["Pclass", "Age", "SibSp", "Parch", "Survived"]

sns.pairplot(train[cols], size=2.5)

plt.tight_layout()

plt.show()
#ピアソンの積率相関係数の計算と可視化

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



#データの読み込み

train = pd.read_csv('../input/titanic-data/train.csv')

train_x = train.drop(['Survived'], axis=1)

train_y = train['Survived']

test_x = pd.read_csv('../input/titanic-data/test.csv')



cols = ["Pclass", "Age", "SibSp", "Parch", "Survived"]



cm = np.corrcoef(train[cols].values.T)



hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)



plt.tight_layout()

plt.show()
#項目間の関係―データの集計

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#データの読み込み

train = pd.read_csv('../input/titanic-data/train.csv')

train_x = train.drop(['Survived'], axis=1)

train_y = train['Survived']

test_x = pd.read_csv('../input/titanic-data/test.csv')



#groupbyで集計

train[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived", ascending=True).plot.scatter("Pclass", "Survived")

#train[["SibSp", "Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(by="Survived", ascending=True).plot.scatter("SibSp", "Survived")

#train[["Parch", "Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(by="Survived", ascending=True).plot.scatter("Parch", "Survived")
#主成分分析による特徴量削減

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler



#データの読み込み

#現状の数値データのみ扱う

train = pd.read_csv('../input/titanic-data/train.csv')

#影響のすくなそうな特徴量と目的変数を削除

drop_col1 = ["PassengerId", "Name", "Age","Cabin", "Ticket","Sex", "Embarked", "Survived"]

drop_col2 = ["PassengerId", "Name", "Age","Cabin", "Ticket","Sex", "Embarked"]

train_x = train.drop(drop_col1, axis=1)

train_y = train['Survived']

test_x = pd.read_csv('../input/titanic-data/test.csv')

test_x = test_x.drop(drop_col2, axis=1)

#train_x.info()

#test_x.info()

#print(test_x)

#print(train_x)



#主成分分析の前のデータの標準化

#Nullがあっても標準化してくれる

#fitで学習したモデルと同じ特徴量でないとエラー

sc = StandardScaler()

X_train_std = sc.fit_transform(train_x)

X_test_std = sc.transform(test_x)

#print(X_train_std)

#print(X_test_std)



#共分散行列を作成

cov_mat = np.cov(X_train_std.T)

#print(cov_mat)



#固有値と固有ベクトルを計算

eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

#print(eigen_vals)

#print(eigen_vecs)



#分散説明率の累積和　一つ目の成分だけで40％近くを占めている。2つ目の成分と合わせると分散の60％近くになる

#固有値を合計する

tot = sum(eigen_vals)

#print(tot)



#分散説明率を計算。合計から割って割合を出す

var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]

#print(var_exp)



#分散説明率の累積和を取得np.cumsumは配列内の要素を足し合わせていったものを順次配列に記録していく

cum_var_exp = np.cumsum(var_exp)

print(cum_var_exp)



plt.bar(range(1, 5), var_exp, alpha=0.5, align='center',label='individual explained variance')

plt.step(range(1, 5), cum_var_exp, where='mid',label='cumulative explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal component index')

plt.legend(loc='best')

plt.tight_layout()

plt.show()
#scikit-learnによる主成分分析

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression



#決定領域の可視化

def plot_decision_regions(X, y, classifier, resolution=0.02):



    # setup marker generator and color map

    markers = ('s', 'x', 'o', '^', 'v')

    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])



    # plot the decision surface

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),

                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())

    plt.ylim(xx2.min(), xx2.max())



    # plot class samples

    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x=X[y == cl, 0], 

                    y=X[y == cl, 1],

                    alpha=0.6, 

                    c=cmap(idx),

                    edgecolor='black',

                    marker=markers[idx], 

                    label=cl)

        

#データの読み込み

train = pd.read_csv('../input/titanic-data/train.csv')

drop_col1 = ["PassengerId", "Name", "Age","Cabin", "Ticket","Sex", "Embarked", "Survived"]

drop_col2 = ["PassengerId", "Name", "Age","Cabin", "Ticket","Sex", "Embarked"]

train_x = train.drop(drop_col1, axis=1)

train_y = train['Survived']

test_x = pd.read_csv('../input/titanic-data/test.csv')

test_x = test_x.drop(drop_col2, axis=1)



# 列Fareの欠損値をAgeの平均値で穴埋め

test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].mean()) 

# 列Fareの欠損値をAgeの中央値で穴埋め

#data['Age'].fillna(data['Age'].median())

# 列Fareの欠損値をAgeの最頻値で穴埋め

#data['Age'].fillna(data['Age'].mode())   



#欠損値の確認

print(train_x.isnull().sum())

print(test_x.isnull().sum())



sc = StandardScaler()

X_train_std = sc.fit_transform(train_x)

X_test_std = sc.transform(test_x)   



#scikit-learnによる主成分分析

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_std)

X_test_pca = pca.transform(X_test_std)



lr = LogisticRegression()

lr = lr.fit(X_train_pca, train_y)

plot_decision_regions(X_train_pca, train_y, classifier=lr)

plt.xlabel('PC 1')

plt.ylabel('PC 2')

plt.legend(loc='lower left')

plt.tight_layout()

plt.show()
#逐次的特徴選択アルゴリズム―特徴選択による次元削減

#特徴選択による次元削減はモデルの複雑さを低減し過学習を回避する方法

#特徴選択では元の特徴量の一部を選択する。特徴抽出では新しい特徴部分空間を生成するために特徴量の集合から情報を抽出する

#SBSは新しい特徴部分空間に目的の個数の特徴量が含まれるまで特徴量全体から特徴量を逐次的に削除していく

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.base import clone

from sklearn.metrics import accuracy_score

from itertools import combinations

from sklearn.neighbors import KNeighborsClassifier



class SBS():

    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):

        self.scoring = scoring               #特徴量を評価する指標

        self.estimator = clone(estimator)    #推定器

        self.k_features = k_features         #選択する特徴量の個数

        self.test_size = test_size           #テストデータの割合

        self.random_state = random_state     #乱数種の固定



   

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        #X_trainの列の数→すべての特徴量の個数

        dim = X_train.shape[1]

        self.indices_ = tuple(range(dim))

        self.subsets_ = [self.indices_]

        #すべての特徴量を用いてスコアを算出

        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)

        self.scores_ = [score]



        while dim > self.k_features:

            scores = []

            subsets = []



            #combinationsはすべての組み合わせを生成して列挙、self.indices_を元データにして,要素数はr=dim-1の数

            for p in combinations(self.indices_, r=dim - 1):

                score = self._calc_score(X_train, y_train, X_test, y_test, p)

                scores.append(score)

                subsets.append(p)



            best = np.argmax(scores)

            self.indices_ = subsets[best]

            self.subsets_.append(self.indices_)

            dim -= 1



            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]



        return self



    def transform(self, X):

        return X[:, self.indices_]



    def _calc_score(self, X_train, y_train, X_test, y_test, indices):

        self.estimator.fit(X_train[:, indices], y_train)

        y_pred = self.estimator.predict(X_test[:, indices])

        score = self.scoring(y_test, y_pred)

        return score



#データの読み込み

train = pd.read_csv('../input/titanic-data/train.csv')

drop_col1 = ["PassengerId", "Name", "Age","Cabin", "Ticket","Sex", "Embarked", "Survived"]

drop_col2 = ["PassengerId", "Name", "Age","Cabin", "Ticket","Sex", "Embarked"]

train_x = train.drop(drop_col1, axis=1)

train_y = train['Survived']

test_x = pd.read_csv('../input/titanic-data/test.csv')

test_x = test_x.drop(drop_col2, axis=1)



# 列Fareの欠損値をAgeの平均値で穴埋め

test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].mean()) 

# 列Fareの欠損値をAgeの中央値で穴埋め

#data['Age'].fillna(data['Age'].median())

# 列Fareの欠損値をAgeの最頻値で穴埋め

#data['Age'].fillna(data['Age'].mode())   



#欠損値の確認

#print(train_x.isnull().sum())

#print(test_x.isnull().sum())



sc = StandardScaler()

X_train_std = sc.fit_transform(train_x)

X_test_std = sc.transform(test_x)   





knn = KNeighborsClassifier(n_neighbors=5)



# selecting features

sbs = SBS(knn, k_features=1)

sbs.fit(X_train_std, train_y)



# plotting performance of feature subsets

#特徴量の個数のリスト（13, 12, 11,....., 1）

k_feat = [len(k) for k in sbs.subsets_]

print(k_feat)



plt.plot(k_feat, sbs.scores_, marker='o')

plt.ylim([0.5, 1.0])

plt.ylabel('Accuracy')

plt.xlabel('Number of features')

plt.grid()

plt.tight_layout()



plt.show()



#K＝3以降で良い結果を示している。最小限の特徴部分集合はどれか確認する

#sbs.subsets_属性は1番目には3個の要素、2番目には2個の要素、3番目には1個の要素

k3 = list(sbs.subsets_[1])

print(train_x.columns[1:][k3])

#print(k3)

#print(train_x.columns)

#ランダムフォレストで特徴量の重要度にアクセスする

#フォレスト内のすべての決定木から計算された不純度の平均的な減少量として特徴量の重要度を測定できる

#決定木やランダムフォレストなどのツリーベースモデルでは特徴量を標準化または正規化する必要はない

#閾値が変化するだけなので意味はない



import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.base import clone

from itertools import combinations

from sklearn.ensemble import RandomForestClassifier



#データの読み込み

train = pd.read_csv('../input/titanic-data/train.csv')

drop_col1 = ["PassengerId", "Name", "Age","Cabin", "Ticket","Sex", "Embarked", "Survived"]

drop_col2 = ["PassengerId", "Name", "Age","Cabin", "Ticket","Sex", "Embarked"]

train_x = train.drop(drop_col1, axis=1)

train_y = train['Survived']

test_x = pd.read_csv('../input/titanic-data/test.csv')

test_x = test_x.drop(drop_col2, axis=1)



# 列Fareの欠損値をAgeの平均値で穴埋め

test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].mean()) 



#クラスラベル以外の特徴量を変数に代入する

feat_labels = train_x.columns



#ランダムフォレストオブジェクトの生成（決定木の個数＝500）

forest = RandomForestClassifier(n_estimators=500,random_state=1)

forest.fit(train_x, train_y)

#特徴量の重要度の抽出

importances = forest.feature_importances_

#重要度の降順で特徴量のインデックスを抽出。[::-1]は後ろから要素を1こずつ取り出す

indices = np.argsort(importances)[::-1]



#様々な特徴量をそれらの相対的な重要度でランク付けしたグラフが作成される。

#特徴量の重要度が合計して1になるように正規化されている

for f in range(train_x.shape[1]):

    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))



plt.title('Feature Importance')

plt.bar(range(train_x.shape[1]), importances[indices],align='center')

#X軸のメモリを設定する

plt.xticks(range(train_x.shape[1]), feat_labels[indices], rotation=90)

plt.xlim([-1, train_x.shape[1]])

plt.tight_layout()

plt.show()

#教師なし学習

#すべての主成分の分散説明率に関心がある場合はn_componentsパラメタをNoneに設定してPCAクラスを初期化

#そうすると主成分がすべて保持されるようになり、explained_variance_ratio_属性を使って分散説明率にアクセスできる

#分散説明率は合計1、主成分の何個何パーセントの情報を占めているか確認できる→特徴量削減とは異なる

import pandas as pd

from sklearn.preprocessing import StandardScaler

import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from matplotlib.colors import ListedColormap



#データの読み込み

train = pd.read_csv('../input/titanic-data/train.csv')

drop_col1 = ["PassengerId", "Name", "Age","Cabin", "Ticket","Sex", "Embarked", "Survived"]

drop_col2 = ["PassengerId", "Name", "Age","Cabin", "Ticket","Sex", "Embarked"]

train_x = train.drop(drop_col1, axis=1)

train_y = train['Survived']

test_x = pd.read_csv('../input/titanic-data/test.csv')

test_x = test_x.drop(drop_col2, axis=1)



# 列Fareの欠損値をAgeの平均値で穴埋め

test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].mean()) 

# 列Fareの欠損値をAgeの中央値で穴埋め

#data['Age'].fillna(data['Age'].median())

# 列Fareの欠損値をAgeの最頻値で穴埋め

#data['Age'].fillna(data['Age'].mode())   



#欠損値の確認

print(train_x.isnull().sum())

print(test_x.isnull().sum())



sc = StandardScaler()

X_train_std = sc.fit_transform(train_x)

X_test_std = sc.transform(test_x)   

 

pca = PCA(n_components=None)

X_train_pca = pca.fit_transform(X_train_std)

pca.explained_variance_ratio_
#修正中。。。

#教師ありアルゴリズムの線形変換法にLDA（線形判別分析）がある。教師なしはPCA（主成分分析）

#データセットの次元数減らすために使用できる

#データ正規分布に従っていることが前提、クラスの共分散行列が全く同じであること、特徴量が統計的にみて互いに独立していること

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from scipy.linalg import eigh

import matplotlib.pyplot as plt



#データの読み込み

train = pd.read_csv('../input/titanic-data/train.csv')

drop_col1 = ["PassengerId", "Name", "Age","Cabin", "Ticket","Sex", "Embarked", "Survived"]

drop_col2 = ["PassengerId", "Name", "Age","Cabin", "Ticket","Sex", "Embarked"]

train_x = train.drop(drop_col1, axis=1)

train_y = train['Survived']

test_x = pd.read_csv('../input/titanic-data/test.csv')

test_x = test_x.drop(drop_col2, axis=1)



# 列Fareの欠損値をAgeの平均値で穴埋め

test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].mean()) 

# 列Fareの欠損値をAgeの中央値で穴埋め

#data['Age'].fillna(data['Age'].median())

# 列Fareの欠損値をAgeの最頻値で穴埋め

#data['Age'].fillna(data['Age'].mode())   



sc = StandardScaler()

X_train_std = sc.fit_transform(train_x)

X_test_std = sc.transform(test_x)   



#set_printoptions(precision=4)で小数点以下4桁までの指定

np.set_printoptions(precision=4)



mean_vecs = []

for label in range(0, 2):

    mean_vecs.append(np.mean(X_train_std[train_y== label], axis=0))

    print('MV %s: %s\n' % (label, mean_vecs[label - 1]))



print(mean_vecs)

print()    



d = 4 

#d行d列の行列を0で初期化

S_W = np.zeros((d, d))

for label, mv in zip(range(1, 4), mean_vecs):

    class_scatter = np.zeros((d, d))  # scatter matrix for each class

    #rowに1行13列のデータが入る（1回ごと）

    for row in X_train_std[train_y == label]:

        row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # make column vectors

        class_scatter += (row - mv).dot((row - mv).T)

    S_W += class_scatter                          # sum class scatter matrices



print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))

print('Class label distribution: %s' % np.bincount(train_y)[1:])

print() 



d = 4  # number of features

S_W = np.zeros((d, d))

for label, mv in zip(range(1, 4), mean_vecs):

    class_scatter = np.cov(X_train_std[train_y == label].T)

    S_W += class_scatter

print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0],S_W.shape[1]))

print() 



mean_overall = np.mean(X_train_std, axis=0)

d = 4  # number of features

S_B = np.zeros((d, d))

for i, mean_vec in enumerate(mean_vecs):

    n = train_x[train_y == i + 1, :].shape[0]

    mean_vec = mean_vec.reshape(d, 1)  # make column vector

    mean_overall = mean_overall.reshape(d, 1)  # make column vector

    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)



print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))

print() 



#inv関数で逆行列、dot関数で行列積、eig関数で固有値を計算

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))



# 固有値と固有ベクトルのタプルを作成する

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])

               for i in range(len(eigen_vals))]



# Sort the (eigenvalue, eigenvector) tuples from high to low

eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)



#固有対を計算した後は固有値を大きいものから降順で並べ替えた結果を出力する

print('Eigenvalues in descending order:\n')

for eigen_val in eigen_pairs:

    print(eigen_val[0])

    

#固有値の実数部の総和を求める

tot = sum(eigen_vals.real)

#分散説明率とその累積和を計算

discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]

cum_discr = np.cumsum(discr)



#最初の2つの線形判別だけでWineトレーニングデータセット内の有益な情報を100%補足している

plt.bar(range(1, 14), discr, alpha=0.5, align='center',label='individual "discriminability"')

plt.step(range(1, 14), cum_discr, where='mid',label='cumulative "discriminability"')

plt.ylabel('"discriminability" ratio')

plt.xlabel('Linear Discriminants')

plt.ylim([-0.1, 1.1])

plt.legend(loc='best')

plt.tight_layout()

plt.show()



#2つの固有ベクトルから変換行列を作成

w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,eigen_pairs[1][1][:, np.newaxis].real))

print('Matrix W:\n', w)



#標準化したトレーニングで―タに変換行列を掛ける

X_train_lda = X_train_std.dot(w)

colors = ['r', 'b', 'g']

markers = ['s', 'x', 'o']



#新しい特徴空間にサンプルを射影する

for l, c, m in zip(np.unique(y_train), colors, markers):

    plt.scatter(X_train_lda[y_train == l, 0], X_train_lda[y_train == l, 1] * (-1), c=c, label=l, marker=m)



plt.xlabel('LD 1')

plt.ylabel('LD 2')

plt.legend(loc='lower right')

plt.tight_layout()

plt.show()






