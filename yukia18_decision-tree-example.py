!pip install pydotplus
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.tree import DecisionTreeRegressor, export_graphviz

# この3つはモデルの可視化に使います
import pydotplus
from IPython.display import Image
from graphviz import Digraph

%matplotlib inline
sns.set()
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
SEED = 42
seed_everything(SEED)

train_df = pd.read_csv("../input/ailab-ml-training-2/train.csv")
test_df = pd.read_csv("../input/ailab-ml-training-2/test.csv")
sample_submission_df = pd.read_csv("../input/ailab-ml-training-2/sample_submission.csv")
# trainとtestを行方向に結合します。
# FEする際に処理が一度で済むので楽チンです。
full_df = pd.concat([train_df, test_df], axis=0)
full_df.head()
# 名前の長さを特徴量にします。連続変数です。
full_df["name_length"] = full_df["name"].apply(lambda x: len(x))

# item_condition_idを特徴量にします。カテゴリカル変数です。
# pd.get_dummiesを利用するとカテゴリカル変数をOneHotにできます。
full_df = pd.get_dummies(full_df, columns=["item_condition_id"])
full_df.head()
# 特徴量とするカラム名をリストで保持しておきます。
features = ["shipping",
            "name_length", 
            "item_condition_id_1",
            "item_condition_id_2",
            "item_condition_id_3",
            "item_condition_id_4",
            "item_condition_id_5"]

# full_dfをtrain/testに戻します。
# 今回はidカラムを利用できます。
train_df = full_df.loc[full_df["train_id"].notnull(), ["train_id", "price"] + features]
test_df = full_df.loc[full_df["test_id"].notnull(), ["test_id"] + features]

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
# 学習データdevと検証データvalに分割します。
dev_df, val_df = train_test_split(train_df, test_size=0.2, random_state=SEED, shuffle=True)
dev_df = dev_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
# 今回は"決定木"と呼ばれる機械学習モデルを利用します。
# ランダムフォレストや勾配ブースティング木(GBDT)も決定木ベースの手法です。
clf = DecisionTreeRegressor(criterion="mse", max_depth=5, min_samples_leaf=100000, random_state=SEED)
clf.fit(dev_df[features], dev_df["price"])
def rmsle(y_true, y_pred):
    return mean_squared_log_error(y_true, y_pred) ** 0.5


print("dev rmsle: {:.3f}".format(rmsle(dev_df["price"], clf.predict(dev_df[features]))))
print("val rmsle: {:.3f}".format(rmsle(val_df["price"], clf.predict(val_df[features]))))
# 学習済みの決定木を可視化してみます。
dot_data = export_graphviz(clf, out_file=None, feature_names=features, filled=True, proportion=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
# 決定木では特徴量重要度(feature importance)を算出できます。
# ある特徴量が目的変数の予測にどれだけ貢献しているかを示す指標です。

fimp = pd.DataFrame()
fimp["feature"] = features
fimp["importance"] = clf.feature_importances_
fimp = fimp.sort_values(by="importance", ascending=False)
sns.barplot(x="importance", y="feature", data=fimp)
plt.title("feature importance")
plt.show()
predictions = clf.predict(test_df[features])
sample_submission_df.head()
sub_df = pd.DataFrame()
sub_df["test_id"] = test_df["test_id"].astype(int)
sub_df["price"] = predictions
sub_df.to_csv("submission.csv", index=False)
