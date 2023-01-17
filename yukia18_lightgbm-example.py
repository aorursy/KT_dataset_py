import os

import random

import warnings



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from category_encoders import OrdinalEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_squared_log_error



import lightgbm as lgb



warnings.filterwarnings("ignore")

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
# Decision Tree Exampleではtrain/testを結合してFEしていました。

# 実は私はそのやり方はしません。

# trainにあってtestにないカテゴリとか、意識して処理をしたいので。



train_df.head()
# 名前の長さを特徴量にします。連続変数です。

train_df["name_length"] = train_df["name"].apply(lambda x: len(x))

test_df["name_length"] = test_df["name"].apply(lambda x: len(x))



# item_condition_idを特徴量にします。カテゴリカル変数です。

# 元々数値ラベルなので何もしません。



# brand_nameを特徴量にします。カテゴリカル変数です。

# OrdinalEncoderを使うと数値ラベルに変換できます。

# 学習データに存在しテストデータに存在しない(unknown)、欠損している(missing)は、NaNを返す設定にします。

oe = OrdinalEncoder(cols=["brand_name"], handle_unknown="return_nan", handle_missing="return_nan")

train_df["brand_name"] = oe.fit_transform(train_df["brand_name"])

test_df["brand_name"] = oe.transform(test_df["brand_name"])



# shippingを特徴量にします。バイナリ変数です。

# 元々数値ラベルなので何もしません。
train_df.head()
# 特徴量とするカラム名をリストで保持しておきます。

features = ["item_condition_id",

            "brand_name",

            "shipping",

            "name_length"]



# カテゴリカル変数として扱うカラム名もリストで保持しておきます。

cat_features = ["item_condition_id",

                "brand_name",

                "shipping"]



train_df = train_df[["train_id", "price"] + features].reset_index(drop=True)

test_df = test_df[["test_id"] + features].reset_index(drop=True)
# 学習データdevと検証データvalに分割します。

dev_df, val_df = train_test_split(train_df, test_size=0.2, random_state=SEED, shuffle=True)

dev_df = dev_df.reset_index(drop=True)

val_df = val_df.reset_index(drop=True)
# lightgbmの学習を行っていきます。

# 代表的なハイパーパラメータを説明します。

# 詳しくはDocumentationを見てください。

# https://lightgbm.readthedocs.io/en/latest/Parameters.html



params = {

    # boosting_type: アルゴリズムの種類です。"gbdt"にしとけばいいです。

    # objective: 目的関数です。今回は回帰ですので"rmse"を使用します。

    # metric: 評価関数です。今回は目的関数と同じ"rmse"です。

    "boosting_type": "gbdt",

    "objective": "rmse",

    "metric": "rmse",

    

    # num_leaves: 葉の数の最大値です。決定木のmax_depthと類似していますが、あちらは木の深さであることに気をつけてください。私は大体16~128を調べます。

    # min_data_in_leaf: 葉に属する最小のデータ数です。20から始めて、過学習気味なら増やす感じです。

    "num_leaves": 32,

    "min_data_in_leaf": 20,

    

    # num_iterations: GBDTは複数の決定木を逐次的に組み合わせて学習を行う機械学習モデルです。

    #                 num_iterationsはイテレート数を指定しますが、適当な大きな値に設定して良いです。

    # early_stopping_round: validデータのmetricが何回改善しなかったら学習を止めるか設定します。

    # learning_rate: 学習率です。0.1~0.001あたりを使います。GBDTはNNと違って学習率を下げればほぼ確実に精度が上がります。

    "num_iterations": 10000,

    "early_stopping_round": 200,

    "learning_rate": 0.1,

    

    # bagging_fraction: ひとつの木が扱うデータの割合です。bagging_fractionの割合だけランダムサンプルします。

    #                   複数の木を組み合わせた時に、それぞれの木を微妙に異なるデータで学習させることで、全体の多様性を向上させます。

    #                   私は0.7~0.9あたりを使います。小さい方が過学習しにくいです。

    # bagging_freq: 上記のbaggingをいくつ行うかです(多分)。私はいつも5にしてます。

    # feature_fraction: ひとつの木が扱う特徴量の割合です。feature_fractionの割合だけランダムサンプルします。

    #                   私は0.5~0.9あたりを使います。今は特徴量が少ししかないので1.0にしています。

    "bagging_fraction": 0.8,

    "bagging_freq": 5,

    "feature_fraction": 1.0,

}
# lightgbmでは、学習時はlgb.Datasetにデータを格納します。

# カテゴリカル変数として扱って欲しいカラム名をここで指定します。

dev_data = lgb.Dataset(dev_df[features], dev_df["price"], categorical_feature=cat_features)

val_data = lgb.Dataset(val_df[features], val_df["price"], categorical_feature=cat_features)



# 学習を行います。

# verbose_eval毎に途中経過を出力します。

clf = lgb.train(params, dev_data, valid_sets=[dev_data, val_data], verbose_eval=100)
# 予測を行います。

dev_pred = clf.predict(dev_df[features], num_iteration=clf.best_iteration)

val_pred = clf.predict(val_df[features], num_iteration=clf.best_iteration)



# 価格として負の値はあり得ないので、後処理します。

dev_pred = np.where(dev_pred < 0, 0.0, dev_pred)

val_pred = np.where(val_pred < 0, 0.0, val_pred)
def rmsle(y_true, y_pred):

    return mean_squared_log_error(y_true, y_pred) ** 0.5





print("dev rmsle: {:.3f}".format(rmsle(dev_df["price"], dev_pred)))

print("val rmsle: {:.3f}".format(rmsle(val_df["price"], val_pred)))
# 決定木では特徴量重要度(feature importance)を算出できます。

# ある特徴量が目的変数の予測にどれだけ貢献しているかを示す指標です。



fimp = pd.DataFrame()

fimp["feature"] = features

fimp["importance"] = clf.feature_importance()

fimp = fimp.sort_values(by="importance", ascending=False)

sns.barplot(x="importance", y="feature", data=fimp)

plt.title("feature importance")

plt.show()
predictions = clf.predict(test_df[features], num_iteration=clf.best_iteration)

predictions = np.where(predictions < 0, 0.0, predictions)
sub_df = pd.DataFrame()

sub_df["test_id"] = test_df["test_id"].astype(int)

sub_df["price"] = predictions

sub_df.to_csv("submission.csv", index=False)
sub_df.head()