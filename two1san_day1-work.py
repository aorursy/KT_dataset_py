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
# プロジェクト期間も加味

sof.days = pd.to_numeric(pd.to_datetime(sof.deadline) - pd.to_datetime(sof.launched))
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