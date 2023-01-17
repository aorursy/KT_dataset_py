# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

uselog = pd.read_csv('../input/use_log.csv')

uselog.isnull().sum()
customer = pd.read_csv('../input/customer_join.csv')

customer.isnull().sum()
# 顧客のグループ化

# 必要な変数に絞り込む

customer_clustering = customer[["mean", "median", "max", "min", "membership_period"]]

customer_clustering.head()
# K-means法　変数間の距離をベースにグループ化

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

# 標準化

sc = StandardScaler()

customer_clustering_sc = sc.fit_transform(customer_clustering)



# モデル構築 (4クラスタ)

kmeans = KMeans(n_clusters = 4, random_state = 0)

clusters = kmeans.fit(customer_clustering_sc)

customer_clustering["cluster"] = clusters.labels_

print(customer_clustering["cluster"].unique())

customer_clustering.head()
# クラスタリング結果分析　データ件数の把握

customer_clustering.columns = ["月内平均値", "月内中央値", "月内最大値", "月内最小値", "会員期間", "cluster"]

customer_clustering.groupby("cluster").count()
customer_clustering.groupby("cluster").mean()
# クラスタリング結果を可視化

# 5つの変数を二次元上にプロットする場合、「次元削除」をする。（教師なし学習の一種） // 主成分(PCA)

from sklearn.decomposition import PCA

X = customer_clustering_sc

# モデル定義

pca = PCA(n_components = 2)

# 主成分分析の実行

pca.fit(X)

x_pca = pca.transform(X)

# 2次元に削減したデータを、データフレームに格納

pca_df = pd.DataFrame(x_pca)

# クラスタリング結果を付与

pca_df["cluster"] = customer_clustering["cluster"]
import matplotlib.pyplot as plt

%matplotlib inline

# グループ毎に散布図をプロット

for i in customer_clustering["cluster"].unique():

    tmp = pca_df.loc[pca_df["cluster"] == i]

    # 散布図

    plt.scatter(tmp[0], tmp[1])
# 退会顧客の特定

# axis (引数 0: 縦方向に 1: 横方向に)

customer_clustering = pd.concat([customer_clustering, customer], axis = 1)

# cluster, is_deleted毎にcustomer_idの件数を集計

customer_clustering.groupby(["cluster", "is_deleted"], as_index = False).count()[["cluster", "is_deleted", "customer_id"]]
# 定期利用しているかどうか

customer_clustering.groupby(["cluster", "routine_flg"], as_index = False).count()[["cluster", "routine_flg", "customer_id"]]
# 教師あり学習の回帰・・あらかじめ正解がわかっているデータを用いて予測を行う

# 2018/5~10の6ヶ月の利用データと、2018年11月の利用回数を教師データとして学習



# 特定の顧客の特定の月のデータを作成する必要がある

# 年月、顧客毎に集計

uselog["usedate"] = pd.to_datetime(uselog["usedate"])

uselog["年月"] = uselog["usedate"].dt.strftime("%Y%m")

uselog_months = uselog.groupby(["年月", "customer_id"], as_index = False).count()

uselog_months.rename(columns={"log_id":"count"}, inplace = True)

del uselog_months["usedate"]

uselog_months.head()
# 当月から過去5ヶ月分の利用回数と、翌月の利用回数を付与

year_months = list(uselog_months["年月"].unique())

predict_data = pd.DataFrame()

# 2018年10月から2019年3月まで

for i in range(6, len(year_months)):

    tmp = uselog_months.loc[uselog_months["年月"] == year_months[i]]

    tmp.rename(columns = {"count":"count_pred"}, inplace = True)

    # 過去6ヶ月分の利用データを取得し、列に追加

    for j in range(1, 7):

        tmp_before = uselog_months.loc[uselog_months["年月"] == year_months[i - j]]

        del tmp_before["年月"]

        tmp_before.rename(columns={"count":"count_{}".format(j - 1)}, inplace = True)

        tmp = pd.merge(tmp, tmp_before, on="customer_id", how="left")

    predict_data = pd.concat([predict_data, tmp], ignore_index = True)

predict_data.head()
# 欠損値の対応 → 対象顧客は、6ヶ月以上滞在している顧客に絞られる

predict_data = predict_data.dropna()

# 歯抜けになるので、indexを初期化

predict_data = predict_data.reset_index(drop = True)

predict_data.head()
# 特徴となるデータとして、会員期間を付与

predict_data = pd.merge(predict_data, customer[["customer_id", "start_date"]], on="customer_id", how="left")

predict_data.head()
# 年月とstart_dateの差から、会員期間を月単位に

predict_data["now_date"] = pd.to_datetime(predict_data["年月"], format="%Y%m")

predict_data["start_date"] = pd.to_datetime(predict_data["start_date"])

from dateutil.relativedelta import relativedelta

predict_data["period"] = None

for i in range(len(predict_data)):

    delta = relativedelta(predict_data["now_date"][i], predict_data["start_date"][i])

    predict_data["period"][i] = delta.years*12 + delta.months

predict_data.head()
# LinearRegression 線形回帰モデル

# 2018年4月以降に新規に入った顧客に絞ってモデル作成

predict_data = predict_data.loc[predict_data["start_date"] >= pd.to_datetime("20180401")]

from sklearn import linear_model

import sklearn.model_selection

model = linear_model.LinearRegression()

X = predict_data[["count_0", "count_1", "count_2", "count_3", "count_4", "count_5", "period"]]

y = predict_data["count_pred"]

# 学習用データと評価用データに分割　無指定の場合、学習用データ75% 評価用データ25%に分割

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)

# 学習用データを用いてモデル作成

model.fit(X_train, y_train)



# 分割する理由・・機械学習は、あくまで未知のデータを予測するのが目的。学習に用いたデータに過剰適合すると未知なデータに対応できなくなる（過学習状態）。

# そのため、学習用データで学習を行い、モデルにとっては未知のデータである評価用データで精度の検証を行う
print(model.score(X_train, y_train))

print(model.score(X_test, y_test))
# モデルに寄与している変数を確認

# 精度の高い予測モデルを構築しても、それがどのようなモデルなのかを理解しないと、説明できない

coef = pd.DataFrame({"feature_names": X.columns, "coefficient": model.coef_})

coef
# 2人の顧客データ

# 1人目：　6ヶ月前から、1ヶ月毎に7回、８回、６回、４回、4回、3回きている

# 2人目：　ゞ　6回、4回、3回、3回、2回、2回きている

x1 = [3,4,4,6,8,7,8]

x2 = [2,2,3,3,4,6,8]

x_pred = [x1, x2]
# modelを用いて予測を行う

model.predict(x_pred)