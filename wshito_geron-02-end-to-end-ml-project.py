import os             # パスの接続にosモジュールを使用

import pandas as pd   # データ操作ライブラリであるpandasモジュールをpdというエイリアスでロード



HOUSING_PATH = "../input"  # データファイルのパスをHOUSING_PATH変数に設定



def load_housing_data(housing_path=HOUSING_PATH):        # load_housing_data関数を定義

    csv_path = os.path.join(housing_path, "housing.csv") # osモジュールを使用してファイルのパスを構築

    return pd.read_csv(csv_path)                         # csvファイルを読み込み
housing = load_housing_data() # 先ほど定義したload_housing_data関数を実行し結果をhousing変数に読み込む



housing.head()  # head()メソッドで最初の数行を表示
housing.info() # DataFrameの構造を表示
housing['ocean_proximity'].value_counts() # カテゴリ毎のカウント数を表示
housing.describe()  # DataFrameの統計サマリを表示
%matplotlib inline

# 上の1行はJupyterノートブックでのみ必要

import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20,15))

plt.show()  # Jupyterノートブックではhist()メソッドが直接描画を出力するので，show()メソッドはなくてもOK
import numpy as np



def split_data (data, test_ratios=[0.5, 0.25, 0.25], seed=None):

    if sum(test_ratios) != 1.0:

        print("test_ratios must sum up to 1.0")

        return

    if seed:

        np.random.seed(seed)

    shuffled_indices = np.random.permutation(len(data))

    accum = 0

    beg = 0

    result = []

    size = len(data)

    for i, v in enumerate(test_ratios):

        accum += v

        to = int(size*accum)

        result.append(data.iloc[shuffled_indices[beg:to]])

        beg = to

    return result
training, validation, test = split_data(housing, [0.5, 0.3, 0.2])

print(len(training), ", ", len(validation), ", ", len(test))

print(len(training) + len(validation)+  len(test), "==", len(housing))



training, validation, test = split_data(housing, [0.5, 0.25, 0.25], 1234) # seedを指定

training.head()
training2, validation2, test2 = split_data(housing, [0.5, 0.25, 0.25], 1234) # seedを指定

training2.head()
housing.plot(y=["median_income"], kind='hist')
housing["median_income"].max()
housing["income_cat"] = np.ceil(housing["median_income"] * 2.0 / 3.0) # 圧縮した値を新たにincome_cat列に追加

housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True) # where(条件, 条件Falseの時の値か処理, 置き換えるか否か)
housing.plot(y=["income_cat"], kind='hist')
from sklearn.model_selection import StratifiedShuffleSplit



# 1回目の1/2分割で訓練データセットが確定

split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

for train_index, half_index in split.split(housing, housing["income_cat"]):

    strat_train_set = housing.loc[train_index]

    half = housing.loc[half_index]

# 2回目の1/2分割で検査データセットとテストデータセットを作成

split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=99)

for test_index, validation_index in split.split(half, half["income_cat"]):

    strat_validation_set = housing.loc[validation_index]

    strat_test_set = housing.loc[test_index]



    
strat_train_set.plot(y=["income_cat"], kind='hist')
strat_validation_set.plot(y=["income_cat"], kind='hist')
strat_test_set.plot(y=["income_cat"], kind='hist')
# median_house_value列と層化抽出のために一時的に作成したincome_cat列を取り除いたデータフレームのコピーを返す

housing = strat_train_set.drop(["median_house_value", "income_cat"], axis=1)



# median_house_value列データのコピーを返す

housing_labels = strat_train_set["median_house_value"].copy()
housing.isnull().any()
housing.isnull().sum()
housing.shape