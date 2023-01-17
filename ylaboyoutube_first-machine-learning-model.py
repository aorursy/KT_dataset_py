import pandas as pd



# ファイルパスを指定

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

#データを読み込み

melbourne_data = pd.read_csv(melbourne_file_path) 

#カラムの確認

melbourne_data.columns
#データの先頭5行を確認

melbourne_data.head()
# nullであるデータの確認

melbourne_data.isnull().sum()
# nullのデータを削除

melbourne_data = melbourne_data.dropna()
# 予測したい値をyとする

y = melbourne_data.Price
# 使用する特徴量をリストに格納

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
# Xに使用する特徴量のみを持ったデータフレームを代入

X = melbourne_data[melbourne_features]
X.head()
# 基本統計量をチェック

X.describe()
# 回帰木をインポート

from sklearn.tree import DecisionTreeRegressor



# random_stateで乱数を固定する。ここを固定しないと毎回違う結果が得られてしまう

melbourne_model = DecisionTreeRegressor(random_state=1)



# 学習を行う

melbourne_model.fit(X, y)
print(X.head())

print("予測値は…")

print(melbourne_model.predict(X.head()))