import pandas as pd
# ファイルパスを設定

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

# csvデータを読み込む

melbourne_data = pd.read_csv(melbourne_file_path) 
#　データの最初の5行

melbourne_data.head()
# 要約統計量の出力

melbourne_data.describe()
# count: 要素の個数

# mean: 算術平均

# std: 標準偏差

# min: 最小値

# max: 最大値

# 50%: 中央値（median）

# 25%, 75%: 1/4分位数、3/4分位数
## 練習問題
import pandas as pd



# ファイルパスを設定

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



# csvデータを読み込む

home_data = pd.read_csv(iowa_file_path)
home_data.head()
home_data.columns
home_data.describe()
avg_lot_size = home_data.describe()['LotArea']['mean']

newest_home_age = 2020 - home_data.describe()['YearBuilt']['max']
print(avg_lot_size)
print(newest_home_age)