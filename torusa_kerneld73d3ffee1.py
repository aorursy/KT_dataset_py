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

# lib 前処理

from sklearn.model_selection import train_test_split # ホールドアウト法に関する関数

from sklearn.model_selection import KFold # 交差検証法に関する関数

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
# データセット

df_raw = pd.read_csv('../input/ks-projects-201801.csv')
#欠損値

display(df_raw.isnull().apply(lambda col: col.value_counts(), axis=0).fillna(0).astype(np.int))
df = df_raw.sample(n=5000, random_state=1234)
y_col = 'state'



x_cols = ['main_category', 'currency','goal']



#カテゴリ変数を、ダミー変数にする

X = pd.get_dummies(df[x_cols], drop_first=True)



#successfulのフラグを目的変数 y とする

y = pd.get_dummies(df[y_col])['successful']
#  無相関化を行うための一連の処理

cov = np.cov(X, rowvar=0 ) # 分散・共分散を求める

_, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

X_decorr = np.dot(S.T, X.T).T #データを無相関化
#  白色化を行うための一連の処理

stdsc = StandardScaler()

stdsc.fit(X_decorr)

X_whitening  = stdsc.transform(X_decorr) # 無相関化したデータに対して、さらに標準化
X = X_whitening



# 全データのうち、何%をテストデータにするか（今回は20%に設定）

test_size = 0.2



# ホールドアウト法を実行（テストデータはランダム選択）

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234) 
#SVM

C = 5

clf = SVC(C=C, kernel="linear")



clf.fit(X_train, y_train)
# ラベルを予測

y_train_est = clf.predict(X_train)





# state正答率を表示

print('訓練誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_train_est)))
display(pd.value_counts(y_train_est))
display(pd.value_counts(y_train))
# 予測値と正解のクロス集計

conf_mat = pd.DataFrame(confusion_matrix(y_train, y_train_est), 

                        index=['正解 = 失敗', '正解 = 成功'], 

                        columns=['予測 = 失敗', '予測 = 成功'])

display(conf_mat)
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

print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_test_est)))



# Precision, Recall, F1-scoreを計算

precision, recall,f1_score, _ = precision_recall_fscore_support(y_test, y_test_est)



# 成功/失敗 での Precision, Recall, F1-scoreを表示

print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))

print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
#SVM

C = 5

clf = SVC(C=C, kernel="rbf")



clf.fit(X_train, y_train)





# ラベルを予測

y_train_est = clf.predict(X_train)





# state正答率を表示

print('訓練誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_train_est)))

display(pd.value_counts(y_train_est))

display(pd.value_counts(y_train))
# 予測値と正解のクロス集計

conf_mat = pd.DataFrame(confusion_matrix(y_train, y_train_est), 

                        index=['正解 = 失敗', '正解 = 成功'], 

                        columns=['予測 = 失敗', '予測 = 成功'])

display(conf_mat)



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

print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_test_est)))



# Precision, Recall, F1-scoreを計算

precision, recall,f1_score, _ = precision_recall_fscore_support(y_test, y_test_est)



# 成功/失敗 での Precision, Recall, F1-scoreを表示

print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))

print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
df = df_raw.sample(n=1000, random_state=1234)
y_col = 'state'



x_cols = ['main_category', 'currency','goal']



#カテゴリ変数を、ダミー変数にする

X = pd.get_dummies(df[x_cols], drop_first=True).values



#目的変数を successfulのフラグに変更

y = pd.get_dummies(df[y_col])['successful'].values
#白色化

cov = np.cov(X, rowvar=0 ) # 分散・共分散を求める

_, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

X_decorr = np.dot(S.T, X.T).T #データを無相関化



stdsc = StandardScaler()

stdsc.fit(X_decorr)

X_whitening  = stdsc.transform(X_decorr) # 無相関化したデータに対して、さらに標準化
X = X_whitening

n_split = 5 # グループ数を設定（今回は5分割）



cross_valid_mae = 0

split_num = 1



#SVMパラメータ

C = 5



# テスト役を交代させながら学習と評価を繰り返す

for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X):

    X_train, y_train = X[train_idx], y[train_idx] #学習用データ

    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

    

    #ロジスティック回帰

    clf = SVC(C=C, kernel="linear")

    clf.fit(X_train, y_train)



    # テストデータに対する予測を実行

    y_pred_test = clf.predict(X_test)

    

    # テストデータに対するMAEを計算

    mae = mean_absolute_error(y_test, y_pred_test)

    print("Fold %s"%split_num)

    print("MAE = %s"%round(mae, 3))

    print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_pred_test)))

    print()

    

    cross_valid_mae += mae #後で平均を取るためにMAEを加算

    split_num += 1



# MAEの平均値を最終的な汎化誤差値とする

final_mae = cross_valid_mae / n_split

print("Cross Validation MAE = %s"%round(final_mae, 3))
X = X_whitening

n_split = 5 # グループ数を設定（今回は5分割）



cross_valid_mae = 0

split_num = 1



#SVMパラメータ

C = 1



# テスト役を交代させながら学習と評価を繰り返す

for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X):

    X_train, y_train = X[train_idx], y[train_idx] #学習用データ

    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

    

    #ロジスティック回帰

    clf = SVC(C=C, kernel="linear")

    clf.fit(X_train, y_train)



    # テストデータに対する予測を実行

    y_pred_test = clf.predict(X_test)

    

    # テストデータに対するMAEを計算

    mae = mean_absolute_error(y_test, y_pred_test)

    print("Fold %s"%split_num)

    print("MAE = %s"%round(mae, 3))

    print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_pred_test)))

    print()

    

    cross_valid_mae += mae #後で平均を取るためにMAEを加算

    split_num += 1



# MAEの平均値を最終的な汎化誤差値とする

final_mae = cross_valid_mae / n_split

print("Cross Validation MAE = %s"%round(final_mae, 3))
X = X_whitening

n_split = 5 # グループ数を設定（今回は5分割）



cross_valid_mae = 0

split_num = 1



#SVMパラメータ

C = 10



# テスト役を交代させながら学習と評価を繰り返す

for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X):

    X_train, y_train = X[train_idx], y[train_idx] #学習用データ

    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

    

    #ロジスティック回帰

    clf = SVC(C=C, kernel="linear")

    clf.fit(X_train, y_train)



    # テストデータに対する予測を実行

    y_pred_test = clf.predict(X_test)

    

    # テストデータに対するMAEを計算

    mae = mean_absolute_error(y_test, y_pred_test)

    print("Fold %s"%split_num)

    print("MAE = %s"%round(mae, 3))

    print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_pred_test)))

    print()

    

    cross_valid_mae += mae #後で平均を取るためにMAEを加算

    split_num += 1



# MAEの平均値を最終的な汎化誤差値とする

final_mae = cross_valid_mae / n_split

print("Cross Validation MAE = %s"%round(final_mae, 3))
X = X_whitening

n_split = 5 # グループ数を設定（今回は5分割）



cross_valid_mae = 0

split_num = 1



#SVMパラメータ

C = 100



# テスト役を交代させながら学習と評価を繰り返す

for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X):

    X_train, y_train = X[train_idx], y[train_idx] #学習用データ

    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

    

    #ロジスティック回帰

    clf = SVC(C=C, kernel="linear")

    clf.fit(X_train, y_train)



    # テストデータに対する予測を実行

    y_pred_test = clf.predict(X_test)

    

    # テストデータに対するMAEを計算

    mae = mean_absolute_error(y_test, y_pred_test)

    print("Fold %s"%split_num)

    print("MAE = %s"%round(mae, 3))

    print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_pred_test)))

    print()

    

    cross_valid_mae += mae #後で平均を取るためにMAEを加算

    split_num += 1



# MAEの平均値を最終的な汎化誤差値とする

final_mae = cross_valid_mae / n_split

print("Cross Validation MAE = %s"%round(final_mae, 3))
df = df_raw



#募集日数カラムを追加

df['dt_launched'] = pd.to_datetime(df.launched)

df['dt_deadline'] = pd.to_datetime(df.deadline)

df['days'] = (df_raw.dt_deadline - df_raw.dt_launched).dt.days
df = df.sample(n=1000, random_state=1234)
y_col = 'state'



x_cols = ['main_category', 'currency','goal', 'days']



#カテゴリ変数を、ダミー変数にする

X = pd.get_dummies(df[x_cols], drop_first=True)



#successfulのフラグを目的変数 y とする

y = pd.get_dummies(df[y_col])['successful'].values
#  無相関化を行うための一連の処理

cov = np.cov(X, rowvar=0 ) # 分散・共分散を求める

_, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

X_decorr = np.dot(S.T, X.T).T #データを無相関化



#  白色化を行うための一連の処理

stdsc = StandardScaler()

stdsc.fit(X_decorr)

X_whitening  = stdsc.transform(X_decorr) # 無相関化したデータに対して、さらに標準化

X = X_whitening

n_split = 5 # グループ数を設定（今回は5分割）



cross_valid_mae = 0

split_num = 1



#SVMパラメータ

C = 10



# テスト役を交代させながら学習と評価を繰り返す

for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X):

    X_train, y_train = X[train_idx], y[train_idx] #学習用データ

    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

    

    #ロジスティック回帰

    clf = SVC(C=C, kernel="linear")

    clf.fit(X_train, y_train)



    # テストデータに対する予測を実行

    y_pred_test = clf.predict(X_test)

    

    # テストデータに対するMAEを計算

    mae = mean_absolute_error(y_test, y_pred_test)

    print("Fold %s"%split_num)

    print("MAE = %s"%round(mae, 3))

    print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_pred_test)))

    print()

    

    cross_valid_mae += mae #後で平均を取るためにMAEを加算

    split_num += 1



# MAEの平均値を最終的な汎化誤差値とする

final_mae = cross_valid_mae / n_split

print("Cross Validation MAE = %s"%round(final_mae, 3))
#df = df_raw

df = df_raw.sample(n=50000, random_state=1234)





#募集日数カラムを追加

df['dt_launched'] = pd.to_datetime(df.launched)

df['dt_deadline'] = pd.to_datetime(df.deadline)

df['days'] = (df_raw.dt_deadline - df_raw.dt_launched).dt.days



y_col = 'state'



#x_cols = ['currency','goal', 'days']

x_cols = ['goal','days']





#カテゴリ変数を、ダミー変数にする

X = pd.get_dummies(df[x_cols], drop_first=True)



#successfulのフラグを目的変数 y とする

y = pd.get_dummies(df[y_col])['successful'].values
X.shape
#  無相関化を行うための一連の処理

cov = np.cov(X, rowvar=0 ) # 分散・共分散を求める

_, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

X_decorr = np.dot(S.T, X.T).T #データを無相関化



#  白色化を行うための一連の処理

stdsc = StandardScaler()

stdsc.fit(X_decorr)

X_whitening  = stdsc.transform(X_decorr) # 無相関化したデータに対して、さらに標準化

X = X_whitening

n_split = 5 # グループ数を設定（今回は5分割）



cross_valid_mae = 0

split_num = 1



#SVMパラメータ

C = 10



# テスト役を交代させながら学習と評価を繰り返す

for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X):

    X_train, y_train = X[train_idx], y[train_idx] #学習用データ

    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

    

    #ロジスティック回帰

    clf = SVC(C=C, kernel="linear")

    clf.fit(X_train, y_train)



    # テストデータに対する予測を実行

    y_pred_test = clf.predict(X_test)

    

    # テストデータに対するMAEを計算

    mae = mean_absolute_error(y_test, y_pred_test)

    print("Fold %s"%split_num)

    print("MAE = %s"%round(mae, 3))

    print('汎化誤差：正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_pred_test)))

    print()

    

    cross_valid_mae += mae #後で平均を取るためにMAEを加算

    split_num += 1



# MAEの平均値を最終的な汎化誤差値とする

final_mae = cross_valid_mae / n_split

print("Cross Validation MAE = %s"%round(final_mae, 3))