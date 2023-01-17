import os



import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
LABEL = "Is_CANCELLED"
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# 3つの csv ファイルを読み込む

train = pd.read_csv("/kaggle/input/techcom-ai-competition/train.csv")

test = pd.read_csv("/kaggle/input/techcom-ai-competition/test.csv")

sample_sub = pd.read_csv("/kaggle/input/techcom-ai-competition/sample_submission.csv")
train  # 中身の確認
test  # 中身の確認
sample_sub  # 中身の確認
# 置換前の教師ラベルを確認する

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)

sns.countplot(x="MEMBERSHIP_STATUS", data=train, order=["INFORCE", "CANCELLED"])

plt.title("Label, before replacement")

sns.despine()



# ラベルの置換処理（値の数値化→カラムのリネーム）

train.MEMBERSHIP_STATUS = train.MEMBERSHIP_STATUS.apply(lambda x: 0 if x == "INFORCE" else 1)

train.rename(columns={"MEMBERSHIP_STATUS":LABEL}, inplace=True)



# 置換後の教師ラベルを確認する

plt.subplot(1, 2, 2)

sns.countplot(x=LABEL, data=train)

plt.title("Label, after replacement")

sns.despine()
train  # 中身の確認
# AGENT_CODE, MEMBER_OCCUPATION_CD: object に

train.AGENT_CODE = train.AGENT_CODE.astype("object")

test.AGENT_CODE = test.AGENT_CODE.astype("object")

train.MEMBER_OCCUPATION_CD = train.MEMBER_OCCUPATION_CD.astype("object")

test.MEMBER_OCCUPATION_CD = test.MEMBER_OCCUPATION_CD.astype("object")



# START_DATE, END_DATE: 日付型に

train.START_DATE = pd.to_datetime(train.START_DATE, format="%Y%m%d")

test.START_DATE = pd.to_datetime(test.START_DATE, format="%Y%m%d")



# MEMBERSHIP_NUMBER はインデックスに

train.set_index("MEMBERSHIP_NUMBER", inplace=True)

test.set_index("MEMBERSHIP_NUMBER", inplace=True)



    

# f = lambda x: np.nan if np.isnan(x) else pd.to_datetime(int(x), format="%Y%m%d")

# train.END_DATE = train.END_DATE.apply(f)

# test.END_DATE = test.END_DATE.apply(f)
y = train[LABEL]

X = train.drop(columns=[LABEL])



# x_ が特徴量 y_ が教師ラベル

# trian が学習用データで valid が評価用データ

x_train, x_valid, y_train, y_valid = train_test_split(

    X,  # 特徴量

    y,  # 教師ラベル

    test_size=0.2,  # テストデータの割合

    random_state=0,  # 指定しないと毎回違う結果になる

    stratify=y  # stratify に教師ラベルを指定すると分割後も元の割合が保たれる

)
x_train  # 中身の確認
y_train  # 中身の確認
x_valid  # 中身の確認
y_valid  # 中身の確認
# 学習用データの教師ラベルを確認する

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)

sns.countplot(x=y_train)

plt.title("Training Data")

sns.despine()



# 評価用データの教師ラベルを確認する

plt.subplot(1, 2, 2)

sns.countplot(x=y_valid)

plt.title("Validation data")

sns.despine()
unuseful_features = ["START_DATE", "END_DATE", "AGENT_CODE"]

x_train.drop(columns=unuseful_features, inplace=True)

x_valid.drop(columns=unuseful_features, inplace=True)

unuseful_features = ["START_DATE", "AGENT_CODE"]

test.drop(columns=unuseful_features, inplace=True)
x_train  # 中身の確認
x_valid  # 中身の確認
test
x_train.info()
x_valid.info()
test.info()
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

x_valid = imputer.transform(x_valid)

test = imputer.transform(test)
x_train.info()
x_valid.info()
test.info()
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

x_valid = encoder.transform(x_valid)

test = encoder.transform(test)
clf = RandomForestClassifier(random_state=42)  # 分類器

clf.fit(x_train, y_train)  # 訓練を行う
y_train_pred = clf.predict(x_train)  # 訓練データの予測値

y_valid_pred = clf.predict(x_valid)  # テストデータの予測値
# 学習用データに対して

print(classification_report(y_train, y_train_pred))
# 評価用データに対して

print(classification_report(y_valid, y_valid_pred))
sample_sub = pd.read_csv("/kaggle/input/techcom-ai-competition/sample_submission.csv")

sample_sub
clf.predict(test)
test  # 中身の再確認
my_prediction = pd.DataFrame()

my_prediction["MEMBERSHIP_NUMBER"] = test.index

my_prediction[LABEL] = clf.predict(test)

my_prediction
# もともと付いているサンプルのデータは邪魔なので削除

sample_sub.drop(columns=[LABEL], inplace=True)



# MEMBERSHIP_NUMBER をキーに自分の予測結果と join する

my_sub = pd.merge(sample_sub, my_prediction, on="MEMBERSHIP_NUMBER")

my_sub
sns.countplot(x=LABEL, data=my_sub)

sns.despine()
my_sub.to_csv("submission.csv", index=False, header=True)