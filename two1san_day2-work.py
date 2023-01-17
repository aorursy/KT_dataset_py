import pandas as pd
# CSVファイル読み込み

csv = pd.read_csv("../input/ks-projects-201801.csv")
csv.head() # 内容確認
# stateをフラグに分解

dummy = pd.get_dummies(csv, columns=['state'])

dummy = dummy.fillna(0)
dummy.corr().style.background_gradient(cmap="autumn_r")
success = dummy[dummy["state_successful"] == 1]

failed = dummy[dummy["state_failed"]==1]

sof = pd.concat([success,failed])

drop_columns = ["state_canceled","state_failed","state_live","state_suspended","state_undefined"]

sof = sof.drop(drop_columns, axis=1)
sof.corr().style.background_gradient(cmap="autumn_r")
# ダミー変数map制作

def get_swap_list(l):

    return {v: i for i, v in enumerate(l)}



# main_category

mc_list = list(sof.main_category.unique())

mc_map = get_swap_list(mc_list)

# category

c_list = list(sof.category.unique())

c_map = get_swap_list(c_list)

# currency

currency_list = list(sof.currency.unique())

currency_map = get_swap_list(currency_list)

# country

country_list = list(sof.country.unique())

country_map = get_swap_list(country_list)
# ダミー変数に置き換え

sof.category = sof.category.map(c_map)

sof.main_category = sof.main_category.map(mc_map)

sof.currency = sof.currency.map(currency_map)

sof.country = sof.country.map(country_map)
sof.corr().style.background_gradient(cmap="autumn_r")
sof.corr().style.background_gradient(cmap="autumn_r")
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
Y = sof.state_successful.values

X = sof.drop(["state_successful","name","launched","deadline"], axis=1).values

clf = SGDClassifier(loss='log', penalty='none', max_iter=1000, fit_intercept=True, random_state=1234)

clf.fit(X, Y)



# 重みを取得して表示

w0 = clf.intercept_[0]

w1 = clf.coef_[0, 0]

w2 = clf.coef_[0, 1]

w3 = clf.coef_[0, 2]

w4 = clf.coef_[0, 3]

w5 = clf.coef_[0, 4]

w6 = clf.coef_[0, 5]

print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}, w6 = {:.3f}'.format(w0, w1, w2, w3, w4, w5, w6))
# ラベルを予測

y_est = clf.predict(X)



# 対数尤度を表示

print('対数尤度 = {:.3f}'.format(- log_loss(Y, y_est)))



# 正答率を表示

print('正答率 = {:.3f}%'.format(100 * accuracy_score(Y, y_est)))
# 予測値と正解のクロス集計

conf_mat = pd.DataFrame(confusion_matrix(Y, y_est), 

                        index=['正解 = 資金調達失敗', '正解 = 資金調達成功'], 

                        columns=['予測 = 資金調達失敗', '予測 = 資金調達成功'])

conf_mat
from sklearn.metrics import precision_score, recall_score, f1_score

print("精度：{}".format(precision_score(Y, y_est)))

print("検出率：{}".format(recall_score(Y, y_est)))

print("F値：{}".format(f1_score(Y, y_est)))
from sklearn.model_selection import KFold, cross_val_score



# 交差検証

kf = KFold(n_splits=5, random_state=30, shuffle=True)

cv_result = cross_val_score(clf, X, Y, cv=kf)
print(cv_result)

print("平均精度：{}".format(cv_result.mean()))
sof.drop(["name","launched","deadline"], axis=1).info()
sof2 = sof.drop(["name","launched","deadline"], axis=1)

sof2.hist()

plt.show()
from sklearn.model_selection import train_test_split



np.random.seed(1)



# データフレームの7割を学習に使い、３割をテストに使う

sof_train,sof_test=train_test_split(sof2, test_size=0.3)

train_X = sof_train[sof2.columns[0:-1]] # 訓練データ説明変数群

train_Y = sof_train[sof2.columns[-1]] # 訓練データ目的変数

test_X = sof_test[sof2.columns[0:-1]] # テストデータ説明変数群

test_Y = sof_test[sof2.columns[-1]] # テストデータ目的変数



# ロジスティック回帰モデル作成

clf.fit(train_X, train_Y)



# ラベルを予測

y_est = clf.predict(test_X)
print('対数尤度 = {:.3f}'.format(- log_loss(test_Y, y_est)))

print('正答率 = {:.3f}%'.format(100 * accuracy_score(test_Y, y_est)))

print("精度：{}".format(precision_score(test_Y, y_est)))

print("検出率：{}".format(recall_score(test_Y, y_est)))

print("F値：{}".format(f1_score(test_Y, y_est)))
# 交差検証

kf = KFold(n_splits=5, random_state=30, shuffle=True)

cv_result = cross_val_score(clf, test_X, test_Y, cv=kf)
print(cv_result)

print("平均精度：{}".format(cv_result.mean()))
from sklearn.svm import SVC



"""

# SVMモデル作成

svm=SVC(C=256, gamma=0.1, kernel="rbf")

svm.fit(train_X, train_Y)

# ラベルを予測

predict_linSVM= svm.predict(test_X)

"""
# CSVファイル読み込み

csv = pd.read_csv("../input/ks-projects-201801.csv")



# 目的変数：stateとし、二値化する

dummy = pd.get_dummies(csv, columns=['state'])

dummy = dummy.fillna(0)

# 成功または失敗の行のみを使用し、それ以外は削除

success = dummy[dummy["state_successful"] == 1]

failed = dummy[dummy["state_failed"]==1]

sof = pd.concat([success,failed])

drop_columns = ["state_canceled","state_failed","state_live","state_suspended","state_undefined"]

sof = sof.drop(drop_columns, axis=1)



# ダミー変数map制作

def get_swap_list(l):

    return {v: i for i, v in enumerate(l)}

# main_category

mc_list = list(sof.main_category.unique())

mc_map = get_swap_list(mc_list)

# category

c_list = list(sof.category.unique())

c_map = get_swap_list(c_list)

# currency

currency_list = list(sof.currency.unique())

currency_map = get_swap_list(currency_list)

# country

country_list = list(sof.country.unique())

country_map = get_swap_list(country_list)



# ダミー変数に置き換え

sof.category = sof.category.map(c_map)

sof.main_category = sof.main_category.map(mc_map)

sof.currency = sof.currency.map(currency_map)

sof.country = sof.country.map(country_map)
# プロジェクト期間も加味

#sof["days"] = pd.to_numeric(pd.to_datetime(sof.deadline) - pd.to_datetime(sof.launched))
# 不要な列を削除

sof = sof.drop(["ID", "name","launched","deadline"], axis=1)
#sof.corr().style.background_gradient(cmap="autumn_r")
# category x main_category : 0.32

#sof = sof.drop(["main_category"], axis=1)

# currency x country: 0.87

#sof = sof.drop(["country"], axis=1)

#sof.corr().style.background_gradient(cmap="autumn_r")
# goal x usd_goal_real: 0.95

#sof = sof.drop(["usd_goal_real"], axis=1)
#sof.corr().style.background_gradient(cmap="autumn_r")
# pledged x backers	: 0.72　⇒ いずれも目的変数との間に無視できない弱い相関(0.1以上)があるためキープ

# pledged x usd pledged	: 0.86

#sof = sof.drop(["usd pledged"], axis=1)
#sof.corr().style.background_gradient(cmap="autumn_r")
# pledged x usd_pledged_real : 0.95　⇒ いずれも目的変数との間に無視できない弱い相関(0.1以上)があるが、ラベル間の相関が強い(0.9以上)ため削除

#sof = sof.drop(["usd_pledged_real"], axis=1)
#sof.corr().style.background_gradient(cmap="autumn

                                     
# goal

#sof = sof.drop(["goal"], axis=1)
sof.head()
sof = sof.loc[:, ['state_successful', 'category', 'main_category', 'currency', 'goal', 'pledged', 'backers',

       'country', 'usd pledged', 'usd_pledged_real', 'usd_goal_real']] # stateを最前列に移動
# 正規化

#sof = (sof - sof.min()) / (sof.max() - sof.min())
np.random.seed(1)



# データフレームの7割を学習に使い、３割をテストに使う

sof_train,sof_test=train_test_split(sof, test_size=0.3)

train_X = sof_train[sof.columns[1:]] # 訓練データ説明変数群

train_Y = sof_train[sof.columns[0]] # 訓練データ目的変数

test_X = sof_test[sof.columns[1:]] # テストデータ説明変数群

test_Y = sof_test[sof.columns[0]] # テストデータ目的変数



# ロジスティック回帰モデル作成

clf.fit(train_X, train_Y)



# ラベルを予測

y_est = clf.predict(test_X)
print('対数尤度 = {:.3f}'.format(- log_loss(test_Y, y_est)))

print('正答率 = {:.3f}%'.format(100 * accuracy_score(test_Y, y_est)))

print("精度：{}".format(precision_score(test_Y, y_est)))

print("検出率：{}".format(recall_score(test_Y, y_est)))

print("F値：{}".format(f1_score(test_Y, y_est)))
# 交差検証

kf = KFold(n_splits=5, random_state=30, shuffle=True)

cv_result = cross_val_score(clf, test_X, test_Y, cv=kf)
print(cv_result)

print("平均精度：{}".format(cv_result.mean()))
# 予測値と正解のクロス集計

conf_mat = pd.DataFrame(confusion_matrix(test_Y, y_est), 

                        index=['正解 = 資金調達失敗', '正解 = 資金調達成功'], 

                        columns=['予測 = 資金調達失敗', '予測 = 資金調達成功'])

conf_mat
# SVMモデル作成

svm=SVC(C=256, gamma=0.1, kernel="rbf")

svm.fit(train_X, train_Y)

# ラベルを予測

predict_linSVM= svm.predict(test_X)
print('対数尤度 = {:.3f}'.format(- log_loss(test_Y, predict_linSVM)))

print('正答率 = {:.3f}%'.format(100 * accuracy_score(test_Y, predict_linSVM)))

print("精度：{}".format(precision_score(test_Y, predict_linSVM)))

print("検出率：{}".format(recall_score(test_Y, predict_linSVM)))

print("F値：{}".format(f1_score(test_Y, predict_linSVM)))
# 予測値と正解のクロス集計

conf_mat = pd.DataFrame(confusion_matrix(test_Y, predict_linSVM), 

                        index=['正解 = 資金調達失敗', '正解 = 資金調達成功'], 

                        columns=['予測 = 資金調達失敗', '予測 = 資金調達成功'])

conf_mat