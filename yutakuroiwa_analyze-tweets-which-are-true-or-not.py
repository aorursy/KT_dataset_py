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
#NLPを使ってみよう
import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.linear_model import LogisticRegression
#データの観察とちょっとした実験
train_df = pd.read_csv("../input/nlpgettingstarted/train.csv")
test_df = pd.read_csv("../input/nlpgettingstarted/test.csv")
train_df.shape
#データの大きさをチェック
train_df.head(10)
#データの概要をのぞいてみる
#真のツイートは1、偽のツイートは0をtargetに表示
train_df[train_df["target"]==0]["text"]
#偽のツイートのtext内容を見てみる
train_df[train_df["target"]==1]["text"]
#真のツイートのtext内容を見てみる
train_df["text"][0:5]
count_vectorizer = feature_extraction.text.CountVectorizer()
#countvectorizerを導入
example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])
#訓練データの冒頭5つのデータを正規化する
count_vectorizer.get_feature_names()
#学習している語を表示してみる
len(count_vectorizer.get_feature_names())
#5つのツイートでは54語を収集している
print(example_train_vectors)
#上のボキャブラリリストに対し、上から順に出現場所と回数を出力
#語彙の発生箇所以外は0となる、疎行列
print(example_train_vectors.todense())
#ボキャブラリがそれぞれのツイートの中で何回登場したかを行列形式で表示
#todenseというコマンドは初めて見た。
#モデルの構築
#text全体に先の処理を施す
train_vectors = count_vectorizer.fit_transform(train_df["text"])
#testデータにも施す。※同じトークンを使うため、fitは作用させないらしい。
test_vectors = count_vectorizer.transform(test_df["text"])
#Ridge回帰を採用。
clf = linear_model.RidgeClassifier()
#モデルがうまく作用しているか、クロスバリデーション法で調べてみる。
#クロスバリデーション法とは、全体のデータを何分割化し、一つをテストデータ、残りを訓練データとする学習を、すべての分割で回す方法。
scores_clf = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")
scores_clf
np.average(scores_clf)
clf.fit(train_vectors, train_df["target"])
lr = LogisticRegression(max_iter = 10000)
scores = model_selection.cross_val_score(lr, train_vectors, train_df["target"], cv=3, scoring = "f1")
scores
np.average(scores)