%matplotlib inline 
import pandas as pd
import numpy as np
from IPython.display import display
from dateutil.parser import parse
import matplotlib.pyplot as plt
from IPython.core.display import display
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA #主成分分析用ライブラリ
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D #3D散布図の描画
import itertools #組み合わせを求めるときに使う
from sklearn.linear_model import LinearRegression
import seaborn as sns
# データの読み込み
df_data = pd.read_csv("../input/kc_house_data.csv")
print(df_data.columns)
display(df_data.head())
display(df_data.tail())
# coutn missing
pd.DataFrame(df_data.isnull().sum(), columns=["num of missing"])
# 欠損データなし
# データの型確認
df_data.info()
# データの数値の個数
print(df_data.shape)
print(df_data.nunique())
# date列の変換
df_data["date"] = [ parse(i[:-7]).date() for i in df_data["date"]]
display(df_data.head())
# 基礎統計
df_data.describe().round(1)
# priceのhistogram
ax=df_data['price'].hist(rwidth=100000,bins=20)
ax.set_title('price')
plt.show()
df_en=df_data.drop(['id','date'],axis=1)
df_en1=df_data.drop(['id','date'],axis=1)
# priceを対数で確認
s_price_log = np.log(df_en1['price'])
s_price_log.plot.hist(x='price')
# priceの対数化
df_log= df_en1
df_log["price"] = df_en1["price"].apply( lambda x: np.log(x) )
# 基礎統計（price対数データ）
df_en1.describe().round(1)
# price（対数データ）と全変数の掛け合わせ
cols = [x for x in df_en1.columns if x not in ('id', 'price', 'date')]
fig, axes = plt.subplots(len(cols), 2, figsize=(10,100))
for i, col in enumerate(cols):
    df_en1[col].plot.hist(ax=axes[i, 0])
    df_en1.plot.scatter(x=col, y = 'price', ax=axes[i, 1])
# 全変数同士の相関の確認
cor = df_en1.corr().style.background_gradient().format("{:.2f}")
cor 
#lat,long,priceの関係性可視化

X = df_en1["lat"]
Y = df_en1["long"]
Z = df_en1["zipcode"]

fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel("lat")
ax.set_ylabel("long")
ax.set_zlabel("zipcode")

ax.scatter3D(X,Y,Z)
plt.show()
# lat、long確認（priceをカラーリング）
plt.figure(figsize = (15,10))
g = sns.FacetGrid(data=df_data, hue='price',size= 5, aspect=2)
g.map(plt.scatter, "long", "lat")
plt.show()
# northエリア判別の新変数作成
north_array = np.zeros((df_en.shape[0],1),float)

for i in range(df_en.shape[0]):
    if df_en.iat[i, 15] < 47.5000 and df_en.iat[i, 15] >= 47.1000:
        north_array[i, 0] = 0
    elif df_en.iat[i, 15] < 47.8000 and df_en.iat[i, 15] >= 47.5000:
        north_array[i, 0] = 1
        
north_array_df = pd.DataFrame(north_array)
north_array_df.columns = ["north"]
print(north_array_df)
# データ合体
df_en = pd.concat([df_en,north_array_df], axis=1)
df_en1 = pd.concat([df_en1,north_array_df], axis=1)
print(df_en.columns)
print(df_en1.columns)
#相関確認
cor = df_en1.corr().style.background_gradient().format("{:.2f}")
cor 

# ★north（0.52）のほうが、元のlat（0.45）より説明力がUPしたので、変数として採用
#　zipcode,latおよび、多重共線性が出たsqft_above,sqft_basementを除外
df_en=df_en.drop(['sqft_above','sqft_basement','zipcode','lat'],axis=1)
df_en1=df_en1.drop(['sqft_above','sqft_basement','zipcode','lat'],axis=1)
print(df_en.columns)
print(df_en1.columns)
#多重共線性の確認
from sklearn.linear_model import LinearRegression
df_vif = df_en.drop(["price"],axis=1)
for cname in df_vif.columns:  
    y=df_vif[cname]
    X=df_vif.drop(cname, axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    rsquared = regr.score(X,y)
    #print(cname,":" ,1/(1-np.power(rsquared,2)))
    if rsquared == 1:
        print(cname,X.columns[(regr.coef_> 0.5) | (regr.coef_ < -0.5)])
# 変数の選択
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm 

count = 1
for i in range(5):
    combi = itertools.combinations(df_en1.drop(["price"],axis=1).columns, i+1) 
    for v in combi:
        y = df_en["price"]
        X = sm.add_constant(df_en[list(v)])
        model = sm.OLS(y, X).fit()
        if count == 1:
            min_aic = model.aic
            min_var = list(v)
        if min_aic > model.aic:
            min_aic = model.aic
            min_var = list(v)
        count += 1
        print("AIC:",round(model.aic), "変数:",list(v))
print("====minimam AIC====")
print(min_var,min_aic)

# ★　====minimam AIC====['sqft_living', 'waterfront', 'grade', 'yr_built', 'north'] 590053.189844908
# LinerRegresshionで、選択した説明変数の決定係数および、説明変数の傾きを確認
y=df_en1["price"].values
X=df_en1[['sqft_living', 'waterfront', 'grade', 'yr_built', 'north']].values
regr = LinearRegression(fit_intercept=True)
regr.fit(X, y)
print("決定係数=%s"%regr.score(X,y))
print("傾き=%s"%regr.coef_,"切片=%s"%regr.intercept_)
# yr_builtデータ確認
plt.figure(figsize = (15,10))
g = sns.FacetGrid(data=df_data,hue='price',size= 10, aspect=2)
g.map(plt.scatter, "yr_built", "yr_renovated")
plt.show()

# 仮説a：yr_builtが新しいほどpriceが高いことが顕著に表れていない⇒仮説棄却
# 仮説b：リノベーションをした住宅のほうが価格が高いわけではない⇒仮説棄却
# どちらの仮説も棄却したため、yr_builtはこのままでOK
# 勾配降下法のcross validationによる検証
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

X_train,X_test,y_train,y_test = train_test_split(np.array(X),np.array(y),test_size=0.3,random_state=1234)

kf = KFold(n_splits=5, random_state=1234, shuffle=True)

df_result = pd.DataFrame()
models = []

for i,(train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    X_train_train, X_train_val = X_train[train_index], X_train[val_index]
    y_train_train, y_train_val = y_train[train_index], y_train[val_index]
    
    regr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1,
     max_depth=1, random_state=0, loss='ls')
    
    regr.fit(X_train_train, y_train_train)
    models.append(regr)
    y_pred = regr.predict(X_train_val)
    df = pd.DataFrame({"y_val":y_train_val, "y_pred":y_pred})
    df_result = pd.concat([df_result, df], axis=0)

# validation dataによる評価指標の算出
    y_val = df_result["y_val"]
    y_pred = df_result["y_pred"]
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred) # ここだけとりあえず見る！
    print(i)
    print("MSE=%s"%round(mse,3) )
    print("RMSE=%s"%round(np.sqrt(mse), 3) )

import numpy as np
import matplotlib.pyplot as plt
print("MAE=%s"%round(mae,3) )
#　モデルの精度評価
y_pred = models[1].predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE=%s"%round(mse,3) )
print("RMSE=%s"%round(np.sqrt(mse), 3) )
print("MAE=%s"%round(mae,3) )