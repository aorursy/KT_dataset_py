import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.svm import SVC
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



sof.head()
sof.describe()
# 説明変数にプロジェクト期間を追加

sof["days"] = pd.to_numeric(pd.to_datetime(sof.deadline) - pd.to_datetime(sof.launched))
# 不要な列を削除

useless_columns = ["ID", "name","launched","deadline"]

sof = sof.drop(useless_columns, axis=1)
# プロジェクト開始時点で値が確定しない説明変数列を削除

anonymous_columns = ["pledged", "backers","usd pledged","usd_pledged_real"]

sof = sof.drop(anonymous_columns, axis=1)
# 目的変数を最前列に移動

sof = sof.loc[:, ['state_successful', 'category', 'main_category', 'currency',

       'goal', 'country', 'usd_goal_real', 'days']]
# 相関係数表示

sns.heatmap(sof.corr(), cmap="coolwarm", annot=True)
# "currency(0.059)" vs "country(0.057)" : 0.87

sof = sof.drop(["country"], axis=1)
# "goal(0.025)" vs "usd_goal_real(0.024)" : 0.95

sof = sof.drop(["goal"], axis=1) # ほぼ同じなので、*_realを残す
# 散布図

sns.pairplot(sof)

# sns.swarmplot("state_successful", "usd_goal_real", "state_successful", sof, dodge=True)
sof[sof["state_successful"]==1].describe()
# 正規化

# sof = (sof - sof.min()) / (sof.max() - sof.min())

# 標準化

from sklearn.preprocessing import StandardScaler

amount_columns = ["usd_goal_real", "days"]

stdsc = StandardScaler() 

sof["usd_goal_real"] = stdsc.fit_transform(sof[["usd_goal_real"]].values)

sof["days"] = stdsc.fit_transform(sof[["days"]].values)
np.random.seed(1)



# データフレームの7割を学習に使い、３割をテストに使う

sof_train,sof_test=train_test_split(sof, test_size=0.3)

train_X = sof_train[sof.columns[1:]] # 訓練データ説明変数群

train_Y = sof_train[sof.columns[0]] # 訓練データ目的変数

test_X = sof_test[sof.columns[1:]] # テストデータ説明変数群

test_Y = sof_test[sof.columns[0]] # テストデータ目的変数
# ロジスティック回帰モデル作成

clf = SGDClassifier(loss='log', penalty='none', max_iter=1000, fit_intercept=True, random_state=1234, tol=0.001)

# 訓練

clf.fit(train_X, train_Y)

# 予測

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
from sklearn.model_selection import GridSearchCV

# グリッドサーチ

kernel=['linear','rbf']

C=np.arange(0.0000000000000001,1.0, 0.25)

gamma=np.arange(0.0000000000000001, 1.0, 0.25)

params={'kernel':kernel, 'C':C, 'gamma':gamma, 'max_iter':[500]}

GS_svm=GridSearchCV(estimator=SVC(), param_grid=params, verbose=True, cv=2)

GS_svm.fit(train_X,train_Y)

print(GS_svm.best_score_)

print(GS_svm.best_estimator_)
# SVMモデル作成

svm=SVC(C=0.2500000000000001, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma=0.2500000000000001,

  kernel='rbf', max_iter=1000, probability=False, random_state=None,

  shrinking=True, tol=0.001, verbose=False)

svm.fit(train_X, train_Y)

# ラベルを予測

predict_SVM= svm.predict(test_X)
print('対数尤度 = {:.3f}'.format(- log_loss(test_Y, predict_SVM)))

print('正答率 = {:.3f}%'.format(100 * accuracy_score(test_Y, predict_SVM)))

print("精度：{}".format(precision_score(test_Y, predict_SVM)))

print("検出率：{}".format(recall_score(test_Y, predict_SVM)))

print("F値：{}".format(f1_score(test_Y, predict_SVM)))
# 予測値と正解のクロス集計

conf_mat = pd.DataFrame(confusion_matrix(test_Y, predict_SVM), 

                        index=['正解 = 資金調達失敗', '正解 = 資金調達成功'], 

                        columns=['予測 = 資金調達失敗', '予測 = 資金調達成功'])

conf_mat
# 交差検証

svm_result = cross_val_score(svm, test_X, test_Y, cv=kf)
print(svm_result)

print("平均正解率：{}".format(svm_result.mean()))