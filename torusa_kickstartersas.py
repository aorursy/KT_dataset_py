# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

#import pandas as pd

#import numpy as np

import matplotlib.pyplot as plt



import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D #3D散布図の描画



# lib model

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, precision_recall_fscore_support



# lib 前処理

from sklearn.model_selection import train_test_split # ホールドアウト法に関する関数

from sklearn.model_selection import KFold # 交差検証法に関する関数

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../input/ks-projects-201801.csv')
y_col = 'state'



x_cols = ['main_category', 'currency','goal']



#カテゴリ変数を、ダミー変数にする

X = pd.get_dummies(df[x_cols], drop_first=True)



#目的変数を successfulのフラグに変更

y = pd.get_dummies(df[y_col])['successful']
X.columns
df.describe()
#  無相関化を行うための一連の処理

cov = np.cov(X, rowvar=0 ) # 分散・共分散を求める

_, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

X_decorr = np.dot(S.T, X.T).T #データを無相関化
#  白色化を行うための一連の処理

stdsc = StandardScaler()

stdsc.fit(X_decorr)

X_whitening  = stdsc.transform(X_decorr) # 無相関化したデータに対して、さらに標準化
# 全データのうち、何%をテストデータにするか（今回は20%に設定）

test_size = 0.2



# ホールドアウト法を実行（テストデータはランダム選択）

X_train, X_test, y_train, y_test = train_test_split(X_whitening, y, test_size=test_size, random_state=1234) 
#ロジスティック回帰

#clf = SGDClassifier(loss='log', penalty='none', max_iter=1000, fit_intercept=True, random_state=1234)

#clf = SGDClassifier(loss='log', penalty='none')



#SVM

C = 5

clf = SVC(C=C, kernel="linear")



#clf.fit(x, y)

clf.fit(X_train, y_train)
# ラベルを予測

#y_est = clf.predict(x)

y_train_est = clf.predict(X_train)





# state正答率を表示

print('訓練誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_train_est)))
display(pd.value_counts(y_train_est))
display(pd.value_counts(y_train))
# 予測値と正解のクロス集計

conf_mat = pd.DataFrame(confusion_matrix(y_train, y_train_est), 

                        index=['正解 = 失敗', '正解 = 成功'], 

                        columns=['予測 = 失敗', '予測 = 成功'])

conf_mat
# Precision, Recall, F1-scoreを計算

precision, recall,f1_score, _ = precision_recall_fscore_support(y_train, y_train_est)



# 成功/失敗 での Precision, Recall, F1-scoreを表示

print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))

print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
#汎化誤差

y_test_est = clf.predict(X_test)





# 予測値と正解のクロス集計

conf_mat = pd.DataFrame(confusion_matrix(y_test, y_test_est), 

                        index=['正解 = 失敗', '正解 = 成功'], 

                        columns=['予測 = 失敗', '予測 = 成功'])

display(conf_mat)





# state正答率を表示

print('訓練誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_test_est)))



# Precision, Recall, F1-scoreを計算

precision, recall,f1_score, _ = precision_recall_fscore_support(y_test, y_test_est)



# 成功/失敗 での Precision, Recall, F1-scoreを表示

print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))

print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))