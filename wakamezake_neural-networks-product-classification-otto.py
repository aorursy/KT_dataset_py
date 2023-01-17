import numpy as np
import pandas as pd
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submit = pd.read_csv("../input/sampleSubmission.csv")
train.head()
train.describe()
train.dtypes
test.head()
submit.head()
# 訓練データ 特徴名取り出し
columns = train.columns[1:-1]
# idとtarget列を取り除いたものをXとする
X = train[columns]
print(columns)
X.head()
# 訓練データのラベル部分
target = train['target']
target.head()
# 予測されるラベル名の取り出し
# 上記targetは2次元データであるため1次元のデータに変換する
# https://deepage.net/features/numpy-ravel.html
y = np.ravel(target)
print(y)
# yの重複なしを調べる
labels = np.unique(y)
labels
# 訓練データは 61878個の種類があり、それぞれ93個の特徴量を持っている
X.shape
from sklearn.neural_network import MLPClassifier
# 
model = MLPClassifier(solver= 'lbfgs', hidden_layer_sizes= (30, 10), alpha= 1e-5, random_state=1)
# 用意した訓練データ Xと出力ラベル yを学習する 
model.fit(X, target)
# 訓練データと同じく特徴量93種類のみ取り出す
_test = test[columns]
_test.head()
# テストデータは 144368個の種類があり、それぞれ93個の特徴量を持っている
_test.shape
# テストデータを使ってどの出力ラベルになりうるか予測する
# predict_probaは出力ラベルそれぞれの確立を吐き出す
test_prob = model.predict_proba(_test)
test_prob
test_prob.shape
# 試しにあるデータの出力結果を見てみる
# 出力ラベルは 9種類のラベル
print(labels)
# それぞれ9種類のどれにあたるか、それぞれの確立が出力にあたる
print(test_prob[0])
# 今回の場合 class4 が予測結果になる
print("Class_{}".format(np.argmax(test_prob[0])+1))
submit[labels] = test_prob
submit.head()
submit.to_csv('./otto_prediction.csv', index = False)
