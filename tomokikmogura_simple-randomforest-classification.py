import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer

from sklearn.metrics import classification_report
# 3つの csv ファイルを読み込む

train = pd.read_csv("/kaggle/input/club-data-set/club_churn_train.csv")

test = pd.read_csv("/kaggle/input/club-data-set/club_churn_test.csv")

real_y = pd.read_csv("/kaggle/input/club-data-set/real_y_test_2.csv")
train  # 中身の確認
test  # 中身の確認
real_y  # 中身の確認
# 必要な列だけ抽出

real_y = real_y.loc[:, ["Unnamed: 0.1", "MEMBERSHIP_STATUS"]]



# 結合する

test = pd.merge(test, real_y, left_on="Unnamed: 0", right_on="Unnamed: 0.1")

test  # 中身の確認
# AGENT_CODE, MEMBER_OCCUPATION_CD: object に

train.AGENT_CODE = train.AGENT_CODE.astype("object")

test.AGENT_CODE = test.AGENT_CODE.astype("object")

train.MEMBER_OCCUPATION_CD = train.MEMBER_OCCUPATION_CD.astype("object")

test.MEMBER_OCCUPATION_CD = test.MEMBER_OCCUPATION_CD.astype("object")



# START_DATE, END_DATE: 日付型に

train.START_DATE = pd.to_datetime(train.START_DATE, format="%Y%m%d")

test.START_DATE = pd.to_datetime(test.START_DATE, format="%Y%m%d")

    

f = lambda x: np.nan if np.isnan(x) else pd.to_datetime(int(x), format="%Y%m%d")

train.END_DATE = train.END_DATE.apply(f)

test.END_DATE = test.END_DATE.apply(f)
train.drop(columns=["Unnamed: 0"], inplace=True)

test.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], inplace=True)
train
test
label = "MEMBERSHIP_STATUS"

y_train = train[label]

x_train = train.drop(columns=[label])

y_test = test[label]

x_test = test.drop(columns=[label])
x_train  # 中身の確認
y_train  # 中身の確認
x_test  # 中身の確認
y_test  # 中身の確認
unuseful_features = ["START_DATE", "END_DATE", "MEMBERSHIP_NUMBER", "AGENT_CODE"]

x_train.drop(columns=unuseful_features, inplace=True)

x_test.drop(columns=unuseful_features, inplace=True)
x_train  # 中身の確認
x_test  # 中身の確認
x_train.info()
x_test.info()
# stragegy = "most_frequent" が最頻値, "median" は中央値による補完

imputer = DataFrameMapper([

    (["MEMBER_MARITAL_STATUS"], [SimpleImputer(strategy="most_frequent")]),

    (["MEMBER_GENDER"], [SimpleImputer(strategy="most_frequent")]),

    (["MEMBER_OCCUPATION_CD"], [SimpleImputer(strategy="most_frequent")]),

    (["MEMBER_ANNUAL_INCOME"], [SimpleImputer(strategy="median")])

    ], input_df=True, df_out=True, default=None

)



# 訓練データの学習

imputer.fit(x_train)



# 補完処理

x_train = imputer.transform(x_train)

x_test = imputer.transform(x_test)

x_train.info()
x_test.info()
encoder = DataFrameMapper([

    (['MEMBER_MARITAL_STATUS'], LabelEncoder()),

    (['MEMBER_GENDER'], LabelEncoder()),

    (['MEMBER_OCCUPATION_CD'], LabelEncoder()),

    (['MEMBERSHIP_PACKAGE'], LabelEncoder()),

    (['PAYMENT_MODE'], LabelEncoder())

    ], input_df=True, df_out=True, default=None

)



# 訓練データの学習

encoder.fit(x_train)



# 補完処理

x_train = encoder.transform(x_train)

x_test = encoder.transform(x_test)
x_train  # 中身の確認
x_test.dtypes  # 中身の確認
clf = RandomForestClassifier(random_state=42)  # 分類器

clf.fit(x_train, y_train)  # 訓練を行う
y_train_pred = clf.predict(x_train)  # 訓練データの予測値

y_test_pred = clf.predict(x_test)  # テストデータの予測値
# 訓練データに対して

print(classification_report(y_train, y_train_pred, labels=["INFORCE", "CANCELLED"]))
# 訓練データに対して

print(classification_report(y_test, y_test_pred, labels=["INFORCE", "CANCELLED"]))