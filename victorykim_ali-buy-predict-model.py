import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
user_data = pd.read_csv("../input/user_data.csv")

item_data = pd.read_csv("../input/item_data.csv")

data_train = pd.read_csv("../input/data_train.csv")

data_eval = pd.read_csv("../input/data_eval.csv")

data_test = pd.read_csv("../input/data_test.csv")
# 划分数据集  predict_date:预测日期

def split_x_y(data, predict_date):

    end_date = "2014-12-1" + str(int(predict_date[-1])+1)

    labels = user_data[(user_data["time"] >= predict_date)&(user_data["time"] < end_date)&(user_data["behavior_type"] == 4)][["user_id", "item_id"]].drop_duplicates()

    labels["is_buy"] = 1

    data = pd.merge(data, labels, how="left", on=["user_id", "item_id"])

    data["is_buy"] = data["is_buy"].fillna(0)

    x_train = data.drop(["user_id", "item_id", "item_category", "is_buy"], axis=1)

    y_train = data["is_buy"]

    return (x_train, y_train)     
# 训练集

(x_train,y_train) = split_x_y(data_train, "2014-12-17")

# 验证集

(x_eval,y_eval) = split_x_y(data_eval, "2014-12-18")

# 测试集 data_test
import xgboost as xgb

xgb_train = xgb.DMatrix(x_train,y_train)

xgb_eval = xgb.DMatrix(x_eval,y_eval)

xbg_test = xgb.DMatrix(data_test.drop(["user_id", "item_id", "item_category"], axis=1))
params = {

    'objective': 'rank:pairwise', # 学习目标

    'eval_metric': 'auc', # 评价指标

    'gamma': 0.1, # 最小损失函数下降值

    'min_child_weight': 1.1, # 子集观察值的最小权重和

    'max_depth': 6, # 树的最大深度

    'lambda': 10, # L2正则化项

    'subsample': 0.7, # 树采样率

    'colsample_bytree': 0.7, # 特征采样率

    'eta': 0.01, # 学习率

    'tree_method':'exact', # 算法类别

    'seed':0

}
watchlist = [(xgb_train,'train'),(xgb_eval,'validate')]

model_xgb = xgb.train(params,xgb_train,num_boost_round=2000,evals=watchlist,early_stopping_rounds=100)
from xgboost import plot_importance

plot_importance(model_xgb)

plt.show()
model_xgb = xgb.train(params,xgb_train,num_boost_round=model_xgb.best_iteration)
from sklearn.preprocessing import MinMaxScaler

eval_data = user_data[user_data["time"] < "2014-12-18"][["user_id", "item_id", "item_category"]].drop_duplicates()

print(len(eval_data))

eval_data["pred"] = model_xgb.predict(xgb.DMatrix(x_eval))

eval_data["pred"] = MinMaxScaler().fit_transform(eval_data["pred"].values.reshape(-1, 1))
threshold = eval_data[["pred"]].sort_values(by="pred", ascending=False).iloc[550][0]

y_pred = eval_data["pred"].tolist()

for i in range(len(y_pred)):

    if y_pred[i] >= threshold:

        y_pred[i] = 1

    else:

        y_pred[i] = 0

y_eval = y_eval.tolist()
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

print("Validate set accuracy score: {:.4f}".format(accuracy_score(y_pred, y_eval)))

print("Validate set F1 score : {:.4f}".format(f1_score(y_pred, y_eval)))

confusion_matrix(y_pred, y_eval) 
model_xgb = xgb.train(params,xgb_eval,num_boost_round=model_xgb.best_iteration)
from xgboost import plot_importance

plot_importance(model_xgb)

plt.show()
from sklearn.preprocessing import MinMaxScaler

item_list = item_data["item_id"].unique().tolist()

predict = data_test[["user_id", "item_id"]]

predict["label"] = model_xgb.predict(xbg_test)

predict["label"] = MinMaxScaler().fit_transform(predict["label"].values.reshape(-1, 1))

predict = predict[predict["item_id"].isin(item_list)].sort_values(by="label", ascending=False)
result = predict.head(550)[["user_id", "item_id"]].drop_duplicates()

result.to_csv("submission.csv", index=False)