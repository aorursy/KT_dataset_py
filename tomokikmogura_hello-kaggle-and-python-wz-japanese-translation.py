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
"""
pandas の read_csv() は引数にファイルパスを、read_excel() は更に sheet_name という引数に対象シートも指定します。
= は代入演算子なので、 = の左辺に書いた変数に関数の戻り値が代入されます。"""

# read_csv()
train = pd.read_csv('/kaggle/input/club-data-set/club_churn_train.csv')
test = pd.read_csv('/kaggle/input/club-data-set/club_churn_test.csv')
real_y_test = pd.read_csv('/kaggle/input/club-data-set/real_y_test_2.csv')

# read_excel
problem_statement = pd.read_excel("/kaggle/input/club-data-set/Assignment.xlsx",
                                  sheet_name="Problem statement")
problem_statement
def to_string(value):
    """文字列への変換。

    引数を str 型に変換して返す。ただし numpy.nan が渡されたらブランクを返す。
    """
    return '' if str(value)==str(np.nan) else str(value)

for index, row in problem_statement.iterrows():
    print(' '.join([to_string(row[0]), to_string(row[1])]))
train
test
real_y_test
# 各データフレームの次元数とカラム名をざざっと確認します
print(f'club_churn_train.csv の次元数（行数, 列数）は {train.shape} です')
print(f'club_churn_train.csv の列名一覧：\n{train.columns} ')
print(f'\nclub_churn_test.csv の次元数は {test.shape} です')
print(f'club_churn_test.csv の列名一覧：\n{test.columns} ')
print(f'\nreal_y_test_2.csv の次元数は {real_y_test.shape} です')
print(f'real_y_test_2.csv の列名一覧：\n{real_y_test.columns} ')
train.info()
# 第一引数に変換対象のオブジェクトを指定し、format に日付の書式を Python が認識できる形式で指定
train.START_DATE = pd.to_datetime(train.START_DATE, format='%Y%m%d')
train.END_DATE = pd.to_datetime(train.END_DATE, format='%Y%m%d')
train.info()
test.START_DATE = pd.to_datetime(test.START_DATE, format='%Y%m%d')
test.END_DATE = pd.to_datetime(test.END_DATE, format='%Y%m%d')
test.info()
real_y_test.info()
train.describe()
test.describe()
train.MEMBER_MARITAL_STATUS.unique()  # データフレーム.列名.unique()
train.MEMBER_MARITAL_STATUS.value_counts()  # 要は データフレーム.列名 とすれば表の中の特定の列を指定できます
train.MEMBER_MARITAL_STATUS.nunique()
train.MEMBER_MARITAL_STATUS.value_counts(dropna=False)
train.MEMBER_MARITAL_STATUS.nunique(dropna=False)
for col in train.select_dtypes(exclude='number').columns:
    print(f'列）{col}：')
    print(f'{train[col].nunique(dropna=False)} 種類のカテゴリが存在します：')  # データフレーム[列名] でも個々の列を指定できます
    print(f'{train[col].value_counts(dropna=False)}\n')
import matplotlib.pyplot as plt  # 代表的な可視化のパッケージ
import seaborn as sns  # matplotlib を使いやすくする wrapper

# Notebook 上にグラフを直に描画したい時のおまじない
%matplotlib inline
sns.scatterplot(x="ANNUAL_FEES", y="ADDITIONAL_MEMBERS", data=train)
plt.title("ANNUAL_FEES vs ADDITIONAL_MEMBERS")
sns.scatterplot(x="ANNUAL_FEES", y="ADDITIONAL_MEMBERS", hue="MEMBERSHIP_STATUS", data=train)
plt.title("ANNUAL_FEES vs ADDITIONAL_MEMBERS, colored by MEMBERSHIP_STATUS")
sns.distplot(train.MEMBER_AGE_AT_ISSUE)  
plt.title("MEMBER_AGE_AT_ISSUE of training set")  # グラフのタイトル
sns.distplot(train.MEMBER_AGE_AT_ISSUE, hue="MEMBERSHIP_STATUS")  
plt.title("MEMBER_AGE_AT_ISSUE of training set, colored by MEMBERSHIP_STATUS")  # グラフのタイトル
# MEMBERSHIP_STATUS の一覧
membership_statuses = np.sort(train.MEMBERSHIP_STATUS.unique())
# 描画処理
ax = None
for membership_status in membership_statuses:
    # MEMBERSHIP_STATUS のカテゴリ毎にグラフを描画する
    if ax is None:
        ax = sns.distplot(train.loc[train.MEMBERSHIP_STATUS==membership_status, "MEMBER_AGE_AT_ISSUE"], kde=False)
    else:
        sns.distplot(train.loc[train.MEMBERSHIP_STATUS==membership_status, "MEMBER_AGE_AT_ISSUE"], ax=ax, kde=False)
# 凡例の描画
plt.legend(membership_statuses,  # 凡例
           loc='upper right',  # 描画位置
           frameon=False)  # 凡例の枠の有無
plt.figure(figsize=(14, 14))  # figsize=(width, height), unit=inches

i = 1
numeric_columns = train.select_dtypes('number').columns
n_numeric_columns = len(numeric_columns.tolist())

for col in numeric_columns:  # 連続量のカラムを col に順次代入
    
    # 描画位置の指定（行数、列数、左上から何番目の位置か？）
    plt.subplot(1 + n_numeric_columns/3, 3, i)
    
    # i 番目の位置に現在の col のヒストグラムを書く（MEMBERSHIP_STATUS で色分け）
    ax = None
    for membership_status in membership_statuses:
        if ax is None:
            ax = sns.distplot(train.loc[train.MEMBERSHIP_STATUS==membership_status, col], kde=False)
        else:
            sns.distplot(train.loc[train.MEMBERSHIP_STATUS==membership_status, col], ax=ax, kde=False)
    plt.legend(membership_statuses,  # 凡例
           loc='upper right',  # 描画位置
           frameon=False)  # 凡例の枠の有無
    i = i + 1
plt.figure(figsize=(14, 10))  # figsize=(width, height), unit=inches
sns.pairplot(train)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
sns.countplot(train.MEMBERSHIP_STATUS, order=membership_statuses)
plt.title("Label of training set")
plt.subplot(1, 2, 2)
sns.countplot(real_y_test.MEMBERSHIP_STATUS, order=membership_statuses)
plt.title("Label of test set")
# AGENT_CODE 毎の件数 N_CUSTOMERS を持ったデータフレームを作る
train_n_customers = train \
                   .groupby('AGENT_CODE') \
                   .MEMBERSHIP_NUMBER \
                   .count() \
                   .reset_index() \
                   .rename(columns={'AGENT_CODE':'AGENT_CODE',
                                    'MEMBERSHIP_NUMBER':'N_CUSTOMERS'})
print("train_n_customers を表示します")
display(train_n_customers)
# 元のデータと AGENT_CODE をキーに内部結合する
train = train.merge(right = train_n_customers, 
                    on = 'AGENT_CODE', 
                    how = 'inner')
display("train を表示します")
display(train)
# 前半の処理が終了した時点
train.groupby('AGENT_CODE').MEMBERSHIP_NUMBER.count()
# reset_index() した結果
train.groupby('AGENT_CODE').MEMBERSHIP_NUMBER.count().reset_index()
# リネームまで完了した結果
train_n_customers = train \
                   .groupby('AGENT_CODE') \
                   .MEMBERSHIP_NUMBER \
                   .count() \
                   .reset_index() \
                   .rename(columns={'AGENT_CODE':'AGENT_CODE',
                                    'MEMBERSHIP_NUMBER':'N_CUSTOMERS'})
train_n_customers
test_n_customers = test \
                  .groupby('AGENT_CODE') \
                  .MEMBERSHIP_NUMBER \
                  .count() \
                  .reset_index() \
                  .rename(columns={'AGENT_CODE':'AGENT_CODE',
                                   'MEMBERSHIP_NUMBER':'N_CUSTOMERS'})
test = test.merge(right = test_n_customers, 
                  on = 'AGENT_CODE', 
                  how = 'inner')
membership_statuses = train.MEMBERSHIP_STATUS.unique()
plt.figure(figsize=(8, 4))

# 訓練データの描画処理
plt.subplot(1, 2, 1)
ax = None
for membership_status in membership_statuses:
    # MEMBERSHIP_STATUS のカテゴリ毎にグラフを描画する
    if ax is None:
        ax = sns.distplot(train.loc[train.MEMBERSHIP_STATUS==membership_status, "N_CUSTOMERS"], kde=False)
    else:
        sns.distplot(train.loc[train.MEMBERSHIP_STATUS==membership_status, "N_CUSTOMERS"], ax=ax, kde=False)
# 凡例の描画
plt.legend(membership_statuses,  # 凡例
           loc='upper right',  # 描画位置
           frameon=False)  # 凡例の枠の有無
plt.title('N_CUSTOMERS of training set')

# テストデータの描画処理
plt.subplot(1, 2, 2)
ax = None
for membership_status in membership_statuses:
    # MEMBERSHIP_STATUS のカテゴリ毎にグラフを描画する
    if ax is None:
        ax = sns.distplot(test.loc[real_y_test.MEMBERSHIP_STATUS==membership_status, "N_CUSTOMERS"], kde=False)
    else:
        sns.distplot(test.loc[real_y_test.MEMBERSHIP_STATUS==membership_status, "N_CUSTOMERS"], ax=ax, kde=False)
# 凡例の描画
plt.legend(membership_statuses,  # 凡例
           loc='upper right',  # 描画位置
           frameon=False)  # 凡例の枠の有無
plt.title('N_CUSTOMERS of test set')
sns.despine()
# 特徴量とラベルを別々のデータフレームに分ける
informative_features = [
    "MEMBERSHIP_TERM_YEARS",
    "ANNUAL_FEES",
    "MEMBER_MARITAL_STATUS",
    "MEMBER_GENDER",
    "MEMBER_ANNUAL_INCOME",
    "MEMBER_OCCUPATION_CD",
    "MEMBERSHIP_PACKAGE",
    "MEMBER_AGE_AT_ISSUE",
    "ADDITIONAL_MEMBERS",
    "PAYMENT_MODE",
    "N_CUSTOMERS"
]
label = "MEMBERSHIP_STATUS"

# x_... に特徴量、y_... にラベル
# ... は訓練データが train で性能評価用データが test
x_train = train.loc[:, informative_features]
y_train = train[label]
x_test = test.loc[:, informative_features]
y_test = real_y_test[label]

x_train.MEMBER_OCCUPATION_CD = x_train.MEMBER_OCCUPATION_CD.astype('object')
x_test.MEMBER_OCCUPATION_CD = x_test.MEMBER_OCCUPATION_CD.astype('object')


print('x_train の先頭10件:')
display(x_train.head())

print('y_train の先頭10件:')
display(y_train.head())

print('x_test の先頭10件:')
display(x_test.head())

print('y_test の先頭10件:')
display(y_test.head())
sample = pd.DataFrame({
    "Interview impression":["not bad", "not bad", np.nan, "not bad", "good", np.nan, "good"],  # 欠損値がある！
    "Math score": [1, 8, 9, 9, 5, 10, 6],
    "Toeic score":[910, 715, 745, 650, 435, 815, 830],
    "Test result":["pass", "failure", "failure", "failure", "pass", "failure", "pass"]
    })
sample
x_train_sample = sample.loc[0:4, ["Interview impression", "Math score", "Toeic score"]]
y_train_sample = sample.loc[0:4, ["Test result"]]
x_test_sample = sample.loc[5:, ["Interview impression", "Math score", "Toeic score"]]
y_test_sample = sample.loc[5:, ["Test result"]]

print("訓練データ")
display(x_train_sample)
display(y_train_sample)

print("テストデータ")
display(x_test_sample)
display(y_test_sample)
# Interview impression を再確認します
print("訓練データの Interview impression")
display(x_train_sample["Interview impression"])
print("テストデータの Interview impression")
display(x_test_sample["Interview impression"])
# 欠損値を補完する
x_train_sample["Interview impression"].fillna('not bad', inplace=True)  # 欠損値を置換する
x_test_sample["Interview impression"].fillna('not bad', inplace=True)  # 欠損値を置換する
print("訓練データの Interview impression （欠損値補完済）")
display(x_train_sample["Interview impression"])
print("テストデータの Interview impression （欠損値補完済）")
display(x_test_sample["Interview impression"])
x_train_sample["Interview impression"] = x_train_sample["Interview impression"].apply(lambda x: 1 if x=="good" else 0)
x_test_sample["Interview impression"] = x_test_sample["Interview impression"].apply(lambda x: 1 if x=="good" else 0)
print("訓練データの Interview impression （エンコード済）")
display(x_train_sample["Interview impression"])
print("テストデータの Interview impression （エンコード済）")
display(x_test_sample["Interview impression"])
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

x_train_copy = x_train.copy(deep=True) 
x_test_copy = x_test.copy(deep=True)

"""ヌル値の補完
今回はヌル値があった場合、最頻値で埋めることにします。SimpleImputer が便利です。
"""
imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(x_train_copy)
x_train_imputed = pd.DataFrame(imputer.transform(x_train_copy),
                               columns=x_train_copy.columns.tolist())
x_test_imputed = pd.DataFrame(imputer.transform(x_test_copy),
                              columns=x_test_copy.columns.tolist())

"""カテゴリ変数のエンコード（数値への置換） 
文字はそのまま扱えないので数値を割り振ります。カテゴリ化には LabelEncoder が使えます。
"""
# データフレームからカテゴリの列名を取り出し1つずつエンコードします
categorical = x_train_copy.select_dtypes(exclude='number').columns.tolist()  # カテゴリの列名
for col in categorical:
    # LabelEncoder のインスタンスを作成
    encoder = LabelEncoder()
    # 現在処理中の列に対して LebelEncoder を fit() させる（今回の場合はカテゴリに数値を割り振っている）
    encoder.fit(x_train_imputed[col])
    # fit() で学習した結果に基づいてデータを加工する
    x_train_imputed[col] = encoder.transform(x_train_imputed[col])
    x_test_imputed[col] = encoder.transform(x_test_imputed[col])
    
"""ランダムフォレストの訓練と評価
"""
# RandomForestClassifier のインスタンスを作成する
clf = RandomForestClassifier(random_state=0)
# 前処理した特徴量とラベルに対して fit() させて分類器を構築する
clf.fit(x_train_imputed, y_train)
# predict() で訓練済の分類器にラベルを予測させる
y_train_predict = clf.predict(x_train_imputed)
y_test_predict = clf.predict(x_test_imputed)

# 正解率の表示
print(f'訓練データに対する正解率：{sum(y_train==y_train_predict)/y_train.shape[0]}')
print(f'テストデータに対する正解率：{sum(y_test==y_test_predict)/y_test.shape[0]}')
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

x_train_copy = x_train.copy(deep=True) 
x_test_copy = x_test.copy(deep=True)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, x_train_copy.select_dtypes(include="number").columns.tolist()),
        ('cat', categorical_transformer, x_train_copy.select_dtypes(exclude="number").columns.tolist())])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf2 = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', RandomForestClassifier(random_state=0))])
clf2.fit(x_train_copy, y_train)
y_train_predict = clf2.predict(x_train)
y_test_predict = clf2.predict(x_test)

# 正解率の表示
print(f'訓練データに対する正解率：{sum(y_train==y_train_predict)/y_train.shape[0]}')
print(f'テストデータに対する正解率：{sum(y_test==y_test_predict)/y_test.shape[0]}')
from sklearn.model_selection import GridSearchCV

x_train_copy = x_train.copy(deep=True) 
x_test_copy = x_test.copy(deep=True)

"""ヌル値の補完
今回はヌル値があった場合、最頻値で埋めることにします。SimpleImputer が便利です。
"""
imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(x_train_copy)
x_train_imputed = pd.DataFrame(imputer.transform(x_train_copy),
                               columns=x_train_copy.columns.tolist())
x_test_imputed = pd.DataFrame(imputer.transform(x_test_copy),
                              columns=x_test_copy.columns.tolist())

"""カテゴリ変数のエンコード（数値への置換） 
文字はそのまま扱えないので数値を割り振ります。カテゴリ化には LabelEncoder が使えます。
"""
# データフレームからカテゴリの列名を取り出し1つずつエンコードします
categorical = x_train_copy.select_dtypes(exclude='number').columns.tolist()  # カテゴリの列名
for col in categorical:
    # LabelEncoder のインスタンスを作成
    encoder = LabelEncoder()
    # 現在処理中の列に対して LebelEncoder を fit() させる（今回の場合はカテゴリに数値を割り振っている）
    encoder.fit(x_train_imputed[col])
    # fit() で学習した結果に基づいてデータを加工する
    x_train_imputed[col] = encoder.transform(x_train_imputed[col])
    x_test_imputed[col] = encoder.transform(x_test_imputed[col])

clf3 = RandomForestClassifier(random_state=0)

# 探索したいハイパーパラメータを辞書形式で指定します
param_grid = {
    "n_estimators": [10, 100, 1000],
    "max_features":['auto', 'sqrt'],
    "criterion": ["gini", "entropy"],
    "max_depth":[3, 4, 5, 6, None],
    "bootstrap": [True, False]
}

# GridSearchCV を使うと param_grid で指定したすべてのハイパーパラメータの
# 組み合わせに対して cross validation を行い最もスコアが良い組み合わせを
# 探索してくれます
searcher = GridSearchCV(estimator=clf3, 
                        param_grid=param_grid,
                        cv=5, 
                        # random_state=0, 
                        n_jobs=-1,
                        scoring="accuracy")
searcher.fit(x_train_imputed, y_train)
y_train_predict = searcher.predict(x_train_imputed)
y_test_predict = searcher.predict(x_test_imputed)

# 正解率の表示
print(f'訓練データに対する正解率：{sum(y_train==y_train_predict)/y_train.shape[0]}')
print(f'テストデータに対する正解率：{sum(y_test==y_test_predict)/y_test.shape[0]}')
searcher.best_params_