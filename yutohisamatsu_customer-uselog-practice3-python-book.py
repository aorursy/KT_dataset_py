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
customer = pd.read_csv('../input/dl5-data/customer_join.csv')

uselog_months = pd.read_csv('../input/dl5-data/use_log_months.csv')
# 当月と過去1ヶ月の利用回数を集計したデータを作成

year_months = list(uselog_months["年月"].unique())

uselog = pd.DataFrame()

for i in range(1, len(year_months)):

    tmp = uselog_months.loc[uselog_months["年月"] == year_months[i]]

    tmp.rename(columns={"count": "count_0"}, inplace=True)

    tmp_before = uselog_months.loc[uselog_months["年月"] == year_months[i-1]]

    del tmp_before["年月"]

    tmp_before.rename(columns={"count": "count_1"}, inplace=True)

    tmp = pd.merge(tmp, tmp_before, on="customer_id", how="left")

    uselog = pd.concat([uselog, tmp], ignore_index=True)

uselog.head()
# 退会前月の退会顧客データを作成・・・退会する月の一ヶ月前までに退会申請をするから

from dateutil.relativedelta import relativedelta

exit_customer = customer.loc[customer["is_deleted"] == 1]

exit_customer["exit_date"] = None

exit_customer["end_date"] = pd.to_datetime(exit_customer["end_date"])

for i in range(len(exit_customer)):

    exit_customer["exit_date"].iloc[i] = exit_customer["end_date"].iloc[i] - relativedelta(months = 1)

exit_customer["年月"] = exit_customer["exit_date"].dt.strftime("%Y%m")

# 型変換(キャスト)・・結合の為に

uselog["年月"] = uselog["年月"].astype(str)

exit_uselog = pd.merge(uselog, exit_customer, on=["customer_id", "年月"], how="left")

print(len(uselog))

exit_uselog.head()
# データ件数は、uselogをベースにしているので、33851件。

# subset ["name"]列に含まれる欠損値を指定してその行を削除

exit_uselog = exit_uselog.dropna(subset=["name"])

print(len(exit_uselog))

print(len(exit_uselog["customer_id"].unique()))

exit_uselog.head()
# 継続顧客

conti_customer = customer.loc[customer["is_deleted"] == 0]

conti_uselog = pd.merge(uselog, conti_customer, on=["customer_id"], how="left")

print(len(conti_uselog))

conti_uselog = conti_uselog.dropna(subset=["name"])

print(len(conti_uselog))
# データのサンプル数を均衡にする。

# データのシャッフル

conti_uselog = conti_uselog.sample(frac=1).reset_index(drop=True)

# customer_idが重複しているデータは最初のデータのみ取得

conti_uselog = conti_uselog.drop_duplicates(subset="customer_id")

print(len(conti_uselog))

conti_uselog.head()
# 継続顧客のデータと退会顧客のデータを縦に結合

predict_data = pd.concat([conti_uselog, exit_uselog], ignore_index=True)

print(len(predict_data))

predict_data.head()
predict_data["period"] = 0

predict_data["now_date"] = pd.to_datetime(predict_data["年月"], format="%Y%m")

predict_data["start_date"] = pd.to_datetime(predict_data["start_date"])

for i in range(len(predict_data)):

    delta = relativedelta(predict_data["now_date"][i], predict_data["start_date"][i])

    predict_data["period"][i] = int(delta.years*12 + delta.months)

predict_data.head()
predict_data.isna().sum()
predict_data = predict_data.dropna(subset=["count_1"])

predict_data.isna().sum()
target_col = ["campaign_name", "class_name", "gender", "count_1", "routine_flg", "period", "is_deleted"]

predict_data = predict_data[target_col]

predict_data.head()
# カテゴリカル変数を用いてダミー変数を作成

predict_data = pd.get_dummies(predict_data)

predict_data.head()
# 不要な列を削除

del predict_data["campaign_name_通常"]

del predict_data["class_name_ナイト"]

del predict_data["gender_M"]

predict_data.head()
# 決定木を使用するためのライブラリ

from sklearn.tree import DecisionTreeClassifier

# 学習用データと評価データを分割する際のライブラリ

import sklearn.model_selection



# データの件数を揃える

exit = predict_data.loc[predict_data["is_deleted"] == 1]

conti = predict_data.loc[predict_data["is_deleted"] == 0].sample(len(exit))



X = pd.concat([exit, conti], ignore_index=True)

# is_deleted列を目的変数

y = X["is_deleted"]

# is_deleted列を削除したデータを説明変数

del X["is_deleted"]

# 学習用/評価用にデータ分割

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)



# モデル定義

model = DecisionTreeClassifier(random_state=0)

# 学習用データを指定し、モデル構築

model.fit(X_train, y_train)

# 評価データの予測を行う

y_test_pred = model.predict(X_test)

print(y_test_pred)
# 実際に正解との比較

results_test = pd.DataFrame({"y_test":y_test, "y_pred":y_test_pred})

results_test.head()
correct = len(results_test.loc[results_test["y_test"]==results_test["y_pred"]])

data_count = len(results_test)

score_test = correct / data_count

print(score_test)
# それぞれのデータを用いた際の精度

print(model.score(X_test, y_test))

print(model.score(X_train, y_train))
# 決定木・・最も綺麗に0と1を分割できる説明変数およびその条件を探す作業を、木構造状に派生させていく手法。

# 分割していく木構造の深さを浅くしてしまえば、モデルは簡易化できる

X = pd.concat([exit, conti], ignore_index=True)

y = X["is_deleted"]

del X["is_deleted"]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)



# 5階層の決定木の深さ

model = DecisionTreeClassifier(random_state=0, max_depth=5)

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(model.score(X_train, y_train))
importance = pd.DataFrame({"feature_names": X.columns, "coefficient": model.feature_importances_})

importance