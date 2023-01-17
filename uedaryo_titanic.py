#　ライブラリ、データのインポート

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib as mpl

!pip install pydotplus

import pydotplus

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.tree import export_graphviz

from mlxtend.plotting import plot_decision_regions

from IPython.display import Image

%matplotlib inline



path = '/kaggle/input/titanic/'

train = pd.read_csv(path + 'train.csv')

test = pd.read_csv(path + 'test.csv')

submit = pd.read_csv(path + 'gender_submission.csv')
#欠損データの表示

print('学習データ\n',train.isnull().sum(),'\n')

print('テストデータ\n',test.isnull().sum())
#欠損値を中央値で埋める

train['Age'] = train['Age'].fillna(train['Age'].median())

test['Age'] = test['Age'].fillna(test['Age'].median())

test['Fare'] = test['Fare'].fillna(test['Age'].median())
#maleを1,femaleを0に変換

train["Sex"] = train["Sex"].map({'male':0, 'female':1})

test["Sex"] = test["Sex"].map({'male':0, 'female':1})
#年齢ヒストグラム

plt.hist(train['Age'])
#運賃ヒストグラム

plt.hist(train['Fare'],bins=30)
#性別をクロス集計

pd.crosstab(train['Sex'], train['Survived'])
#相関係数をヒートマップ表示

sns.heatmap(train[['Survived','Age','Sex','Fare','Pclass', 'SibSp','Parch']].corr(), cmap= sns.color_palette('coolwarm', 10), annot=True, fmt='.2f', vmin = -1, vmax = 1)
# データを説明変数と目的変数に分離する、カラムを使うものだけ抽出

columns = ['Age','Fare']

X_train = train[columns]

y_train = train['Survived']

X_test = test[columns]
#ジニ不純度を指標に決定木学習

tree = DecisionTreeClassifier(criterion='gini', max_depth=10)

tree.fit(X_train, y_train)

plot_decision_regions(X_train.to_numpy(), y_train.to_numpy(), clf=tree)

plt.xlabel('Age')

plt.ylabel('Fare')

plt.legend(loc='upper left')

plt.show()
#学習後の木のモデルを表示

dot_data = export_graphviz(tree, filled=True, rounded=True, class_names=['0','1'],feature_names=columns, out_file=None)

tree_graph = pydotplus.graph_from_dot_data(dot_data)

tree_graph.write_png('tree.png')

from IPython.display import Image, display_png

display_png(Image("tree.png"))
#訓練データのスコアを表示

pred_train = tree.predict(X_train)

accuracy_score(y_train, pred_train)
#提出

pred_test = tree.predict(X_test)

submit['Survived'] = pred_test

submit[['PassengerId', 'Survived']].to_csv('decision_tree.csv',index=False)