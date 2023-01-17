%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D #3D散布図の描画
# lib model

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, precision_recall_fscore_support

from sklearn.svm import SVC

from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier
# lib 前処理

from sklearn.model_selection import train_test_split # ホールドアウト法に関する関数

from sklearn.model_selection import KFold # 交差検証法に関する関数

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
# データセット

df_raw = pd.read_csv('../input/ks-projects-201801.csv')
#欠損値

display(df_raw.isnull().apply(lambda col: col.value_counts(), axis=0).fillna(0).astype(np.int))
df = df_raw
y_col = 'state'



x_cols = ['category', 'currency','goal']



#カテゴリ変数を、ダミー変数にする

X = pd.get_dummies(df[x_cols], drop_first=True)



#successfulのフラグを目的変数 y とする

y = pd.get_dummies(df[y_col])['successful'].values
# 全データのうち、何%をテストデータにするか（今回は20%に設定）

test_size = 0.2



# ホールドアウト法を実行（テストデータはランダム選択）

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234) 
#決定木モデル

clf = DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_split=3, min_samples_leaf=3, random_state=1234)

clf = clf.fit(X_train, y_train)
# ラベルを予測

y_train_est = clf.predict(X_train)





# state正答率を表示

print('訓練誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_train_est)))



#汎化誤差

y_test_est = clf.predict(X_test)



# state正答率を表示

print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_test_est)))
df = df_raw



y_col = 'state'

x_cols = ['main_category', 'currency','goal']



#カテゴリ変数を、ダミー変数にする

X = pd.get_dummies(df[x_cols], drop_first=True)



#successfulのフラグを目的変数 y とする

y = pd.get_dummies(df[y_col])['successful'].values



# 全データのうち、何%をテストデータにするか（今回は20%に設定）

test_size = 0.2



# ホールドアウト法を実行（テストデータはランダム選択）

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234) 



#ランダムフォレスト

clf = RandomForestClassifier(n_estimators=15, max_depth=None, criterion="gini",

                                                 min_samples_leaf=2, min_samples_split=2, random_state=1234)

clf.fit(X_train, y_train)
# ラベルを予測

y_train_est = clf.predict(X_train)





# state正答率を表示

print('訓練誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_train_est)))



#汎化誤差

y_test_est = clf.predict(X_test)



# state正答率を表示

print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_test_est)))
df = df_raw



#募集日数カラムを追加

df['dt_launched'] = pd.to_datetime(df.launched)

df['dt_deadline'] = pd.to_datetime(df.deadline)

df['days'] = (df.dt_deadline - df.dt_launched).dt.days

df['apd'] = df.goal/(df.days + 1) #daysの値が0のデータがあるため、その対策の +1
y_col = 'state'



x_cols = ['category', 'currency','goal', 'days', 'apd']





#カテゴリ変数を、ダミー変数にする

X = pd.get_dummies(df[x_cols], drop_first=True)



#successfulのフラグを目的変数 y とする

y = pd.get_dummies(df[y_col])['successful'].values
# 全データのうち、何%をテストデータにするか（今回は20%に設定）

test_size = 0.2



# ホールドアウト法を実行（テストデータはランダム選択）

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234) 
#決定木モデル

clf = DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_split=3, min_samples_leaf=3, random_state=1234)

clf = clf.fit(X_train, y_train)

clf_DTC = clf



# ラベルを予測

y_train_est = clf_DTC.predict(X_train)

y_test_est = clf_DTC.predict(X_test)



# state正答率を表示

print('決定木モデル')

print('訓練誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_train_est)))

print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_test_est)))

#ランダムフォレスト1

clf = RandomForestClassifier(n_estimators=20, max_depth=None, criterion="gini",

                                                 min_samples_leaf=3, min_samples_split=3, random_state=1234)

clf.fit(X_train, y_train)

clf_RFC1 = clf.fit(X_train, y_train)



# ラベルを予測

y_train_est = clf_RFC1.predict(X_train)

y_test_est = clf_RFC1.predict(X_test)



# state正答率を表示

print('ランダムフォレストモデル')

print('訓練誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_train_est)))

print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_test_est)))

#ランダムフォレスト2

clf = RandomForestClassifier(n_estimators=40, max_depth=None, criterion="gini",

                                                 min_samples_leaf=3, min_samples_split=4, random_state=1234)

clf.fit(X_train, y_train)

clf_RFC2 = clf.fit(X_train, y_train)



# ラベルを予測

y_train_est = clf_RFC2.predict(X_train)

y_test_est = clf_RFC2.predict(X_test)



# state正答率を表示

print('ランダムフォレストモデル')

print('訓練誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_train_est)))

print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_test_est)))

#ランダムフォレスト3

clf = RandomForestClassifier(n_estimators=40, max_depth=None, criterion="gini",

                                                 min_samples_leaf=3, min_samples_split=8, random_state=1234)

clf.fit(X_train, y_train)

clf_RFC3 = clf.fit(X_train, y_train)



# ラベルを予測

y_train_est = clf_RFC3.predict(X_train)

y_test_est = clf_RFC3.predict(X_test)



# state正答率を表示

print('ランダムフォレストモデル')

print('訓練誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_train_est)))

print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_test_est)))



#ランダムフォレスト4

clf = RandomForestClassifier(n_estimators=40, max_depth=None, criterion="gini",

                                                 min_samples_leaf=3, min_samples_split=12, random_state=1234)

clf.fit(X_train, y_train)

clf_RFC4 = clf.fit(X_train, y_train)



# ラベルを予測

y_train_est = clf_RFC4.predict(X_train)

y_test_est = clf_RFC4.predict(X_test)



# state正答率を表示

print('ランダムフォレストモデル')

print('訓練誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_train_est)))

print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_test_est)))
#アダブースト1

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3,min_samples_leaf=2,

                                                min_samples_split=2, random_state=1234, criterion="gini"),

                         n_estimators=10, random_state=1234)

clf.fit(X_train, y_train)

clf_ABC1 = clf



# ラベルを予測

y_train_est = clf_ABC1.predict(X_train)

y_test_est = clf_ABC1.predict(X_test)





# state正答率を表示

print('ランダムフォレストモデル')

print('訓練誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_train_est)))

print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_test_est)))
#アダブースト2

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=None,min_samples_leaf=2,

                                                min_samples_split=2, random_state=1234, criterion="gini"),

                         n_estimators=40, random_state=1234)

clf.fit(X_train, y_train)

clf_ABC2 = clf



# ラベルを予測

y_train_est = clf_ABC2.predict(X_train)

y_test_est = clf_ABC2.predict(X_test)





# state正答率を表示

print('ランダムフォレストモデル')

print('訓練誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_train_est)))

print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_test_est)))
#アダブースト3

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10,min_samples_leaf=2,

                                                min_samples_split=2, random_state=1234, criterion="gini"),

                         n_estimators=40, random_state=1234)

clf.fit(X_train, y_train)

clf_ABC3 = clf



# ラベルを予測

y_train_est = clf_ABC3.predict(X_train)

y_test_est = clf_ABC3.predict(X_test)





# state正答率を表示

print('ランダムフォレストモデル')

print('訓練誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_train_est)))

print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_test_est)))
#アダブースト4

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10,min_samples_leaf=3,

                                                min_samples_split=4, random_state=1234, criterion="gini"),

                         n_estimators=40, random_state=1234)

clf.fit(X_train, y_train)

clf_ABC4 = clf



# ラベルを予測

y_train_est = clf_ABC4.predict(X_train)

y_test_est = clf_ABC4.predict(X_test)





# state正答率を表示

print('ランダムフォレストモデル')

print('訓練誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_train_est)))

print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_test_est)))
#アダブースト5

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10,min_samples_leaf=3,

                                                min_samples_split=8, random_state=1234, criterion="gini"),

                         n_estimators=40, random_state=1234)

clf.fit(X_train, y_train)

clf_ABC5 = clf



# ラベルを予測

y_train_est = clf_ABC5.predict(X_train)

y_test_est = clf_ABC5.predict(X_test)





# state正答率を表示

print('ランダムフォレストモデル')

print('訓練誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_train_est)))

print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_test_est)))