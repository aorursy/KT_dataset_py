#ライブラリをインポート
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#CSVファイル読み込み。
df_train = pd.read_csv('data_house/train.csv')
#読み込んだデータの列名一覧を表示。
df_train.columns
#欠損値が多い列を探す。
missing_data = df_train.isnull().sum().sort_values(ascending=False)
missing_data = pd.concat([missing_data], axis=1, keys=['missing_data'])
missing_data.head(20)
#欠損値が多い列を除外。
df_train = df_train.drop((missing_data[missing_data['missing_data'] > 30]).index,1)
df_train
#ヒートマップ表示。
plt.figure(figsize=(20,20))
sns.heatmap(df_train.corr(), cmap='Reds',annot = True)
#ヒートマップを見て、いくつかの列を選択。
#.fillna(df_train.mean())の部分は「欠損値している部分は、平均値を使って代用する」という意味です。
df_train = df_train.loc[:,[ 'GrLivArea', 'GarageArea', 'MasVnrArea', 'TotalBsmtSF','SalePrice' ]].fillna(df_train.mean())
df_train
#再度ヒートマップで確認。
sns.heatmap(df_train.corr(), cmap='Reds',annot = True)
# X_trainにSalePriceを除いたtrain_dfを代入。
X_train = df_train.drop("SalePrice", axis=1)

# Y_trainにSalePriceのみが入ったtrain_dfを代入。
Y_train = df_train["SalePrice"]
#テストデータからも、訓練データから同じ列を抽出。
X_test = pd.read_csv('data_house/test.csv')
Id = X_test["Id"]
X_test = X_test.loc[:,['GrLivArea', 'GarageArea', 'MasVnrArea','TotalBsmtSF']].fillna(X_test.mean())
X_test
#モデルを使って予測値を出す。
from sklearn.linear_model import ElasticNet

el = ElasticNet(alpha=0.0005)
el.fit(X_train , Y_train)

Y_pred = el.predict(X_test)
#予測値をcsvファイルとして出力。
Predict = pd.DataFrame({
        "Id": Id,
        "SalePrice": Y_pred
    })

Predict.to_csv('data_house/el.predict.csv', index=False, encoding="SHIFT-JISx0213")