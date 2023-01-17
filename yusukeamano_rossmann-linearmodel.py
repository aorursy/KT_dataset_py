import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate
from sklearn.linear_model import LinearRegression


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#reading data files
store=pd.read_csv("../input/rossmann-store-sales/store.csv")
train=pd.read_csv("../input/rossmann-store-sales/train.csv")
test =pd.read_csv("../input/rossmann-store-sales/test.csv")
train_means = train.groupby([ 'Store', 'DayOfWeek', 'Promo' ])['Sales'].mean().reset_index()
print("とりあえず適当にグルーピングした平均値を予測値とすることにする．\n"
      "train データを店，曜日，プロモーション有無で集計\n",
      'shape', train_means.shape)
display(train_means.head())
print("test データに結合")
test = pd.merge(test, train_means,
                on = ['Store','DayOfWeek','Promo'], how='left')
test.fillna(train.Sales.mean(), inplace=True)
display(test.head())
#自己検証のためにRMSPEを算出
def gen_RMSPE(pred, ans):
    tmp_0 = (pred - ans)/ans
    tmp_1 = tmp_0[np.isfinite(tmp_0)] #ansが0のレコード(=tmp0がinfのレコード)は無視する
    return(np.sqrt(np.power(tmp_1,2).sum()/tmp_1.shape[0]))
val_train = pd.merge(train, train_means,
                on = ['Store','DayOfWeek','Promo'], how='left')
val_train.head()
gen_RMSPE(val_train["Sales_y"],val_train["Sales_x"])
#アイデア１
#"Open"が0のときは当然ながら0
open_sales = pd.DataFrame({"Sales":train["Sales"],"Open":train["Open"]})
open_sales.groupby("Open").mean()
x_train = train["Open"]
y_train = train["Sales"]
#線形回帰（単回帰）する
from sklearn.linear_model import LinearRegression

#reshapeしてあげる。よくわからんが、こうしないとエラーがでる、、、
x_train = train[["Open"]]

#線形回帰実行（普通の重回帰）
model = LinearRegression()
model.fit(x_train,y_train)
#"Open"での条件付き期待値と一致
print(model.coef_,model.intercept_)
pred = model.predict(x_train)
gen_RMSPE(model.predict(x_train),y_train)
#アイデア２
#Storeをすべてダミー変数として、読み込む
x_train2 = train[['Store', 'DayOfWeek', 'Promo', "Sales"]]
x_train2 = pd.concat([x_train2.drop('Store', axis=1), pd.get_dummies(x_train2['Store']).iloc[:, :-1]], axis=1)
x_train2.head()
#実は、trainにあってtestにない、Storeがあるので、回帰する前にちゃんと抜いてあげる
x_test2 = test[['Store', 'DayOfWeek', 'Promo']]
idx = x_test2["Store"].unique()
len(idx)
idx = idx[0:len(idx)-1]
idx_list = idx.tolist()
idx_list.append("DayOfWeek")
idx_list.append("Promo")
idx_list.append("Sales")
x_train2_modifyed = x_train2[idx_list]
#'DayOfWeek','Promo'をダミー化
x_train2_modifyed  = pd.concat([x_train2_modifyed .drop('DayOfWeek', axis=1), pd.get_dummies(x_train2_modifyed ['DayOfWeek']).iloc[:, :-1]], axis=1)
x_train2_modifyed  = pd.concat([x_train2_modifyed .drop('Promo', axis=1), pd.get_dummies(x_train2_modifyed ['Promo']).iloc[:, :-1]], axis=1)
x_train2_modifyed.head()
y_train2_modifyed = x_train2_modifyed[["Sales"]]
x_train2_modifyed = x_train2_modifyed.drop(["Sales"], axis = 1)
#testもデータを作成
x_test2_modifyed = pd.concat([x_test2.drop('Store', axis=1), pd.get_dummies(x_test2['Store']).iloc[:, :-1]], axis=1)
x_test2_modifyed = pd.concat([x_test2_modifyed.drop('DayOfWeek', axis=1), pd.get_dummies(x_test2_modifyed['DayOfWeek']).iloc[:, :-1]], axis=1)
x_test2_modifyed = pd.concat([x_test2_modifyed.drop('Promo', axis=1), pd.get_dummies(x_test2_modifyed['Promo']).iloc[:, :-1]], axis=1)
print(x_train2_modifyed.shape,x_test2_modifyed.shape)
x_train2_modifyed.head()
x_test2_modifyed.head()
#３．線形回帰する
from sklearn.linear_model import LinearRegression

#線形回帰実行（普通の重回帰）
model = LinearRegression()
model.fit(x_train2_modifyed,y_train2_modifyed)
pred = model.predict(x_test2_modifyed)
#1次元配列に変換
pred_linear_1 = np.ravel(pred)
submission1 =test["Id"]
submission1["Sales"]= pd.Series(np.ravel(pred))
#提出用にデータを結合
submission_linear = pd.DataFrame({
        "Id": test["Id"][:41088],
        "Sales": pred_linear_1
    })

#"Id"順に並べ替え
submission_1 = submission_linear.sort_values('Id')

#0より小さい予測値を0に変換
submission_1["Sales"] = submission_1["Sales"].apply(lambda x:max(x,0))

submission_1.to_csv('./submission_dummy.csv', index = False )
pred_self = model.predict(x_train2_modifyed)
train["pred_Sales"] = pred_self
train_analysis = train[train["Sales"]>0]
train_analysis["RMSPE"] = np.power((train_analysis["pred_Sales"] - train_analysis["Sales"])/train_analysis["Sales"], 2)
train_analysis.head()
train_analysis["RMSPE"] .mean()
#RMSPE順に降順ソート
train_analysis.sort_values('RMSPE',ascending=False).head(20)
#storeのデータと結合
train_analysis=train_analysis.merge(store,on=["Store"],how="inner")
train_analysis.head()
#月、週、日を分離
train_analysis["Date"]=pd.to_datetime(train_analysis["Date"])
train_analysis["Year"]=train_analysis["Date"].dt.year
train_analysis["Month"]=train_analysis["Date"].dt.month
train_analysis["Day"]=train_analysis["Date"].dt.day
train_analysis["Week"]=train_analysis["Date"].dt.week%4
#グラフ化してみる
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display import display
import math

#RMSPEは対数変換した
train_analysis["RMSPE_Log10"]=train_analysis["RMSPE"].apply(lambda x:math.log10(x))
Sales_RMSPE =pd.DataFrame({"Customers":train_analysis["Sales"],"RMSPE_Log10":train_analysis["RMSPE_Log10"]})
print(plt.scatter(Sales_RMSPE["Customers"],Sales_RMSPE["RMSPE_Log10"], marker = "."))
plt.figure(figsize=(10,10))
sns.set(style="whitegrid")
sns.boxplot(data=train_analysis,x="DayOfWeek",y="RMSPE_Log10")
print("曜日ごとに箱ひげ図を描画。日曜だけ誤差が大きい。しかし曜日はすでにダミーで入れているので、日曜におけるほかの要素が関係しているはず。")
print("深堀してみる。RMSPEでソート後、日曜を抽出")
import collections
collections.Counter(train_analysis["DayOfWeek"])
#train_analysis_sunday = train_analysis[train_analysis["DayOfWeek"]==7]
print("と考えたが、日曜かつ売上＞０のレコード数が小さくて全体に対する寄与が小さいので無視する。")
plt.figure(figsize=(10,10))
sns.set(style="whitegrid")
sns.boxplot(data=train_analysis,x="Month",y="RMSPE_Log10")
print("月ごとに箱ひげ図を描画。12月の誤差が大きい？")
plt.figure(figsize=(10,10))
sns.set(style="whitegrid")
sns.boxplot(data=train_analysis,x="Assortment",y="RMSPE_Log10")
print("Assortmentごとに箱ひげ図を描画。bの誤差が大きい")
#'Date',"Assortment"を変数に含める。
tmp3 = train[['Store', 'DayOfWeek', 'Promo', "Sales",'Date']]
tmp3 = tmp3.merge(store,on=["Store"],how="inner")
tmp3 = tmp3[['Store', 'DayOfWeek', 'Promo', "Sales",'Date',"Assortment"]]
tmp3.head()
#Monthを抽出する
tmp3["Date"]=pd.to_datetime(tmp3["Date"])
tmp3["Month"]=tmp3["Date"].dt.month
tmp3_1 = tmp3.drop(["Date"], axis = 1)
tmp3_1.head()
collections.Counter(tmp3_1["Month"])
#"Month"と"Assortment"をダミー変数にする
#"Month"から
tmp3_1["Month"] = np.where(tmp3_1["Month"] == 12,1,0)
print(tmp3_1["Month"].sum())
collections.Counter(tmp3_1["Assortment"])
#"Assortment"
tmp3_1["Assortment"] = np.where(tmp3_1["Assortment"] == "b",1,0)
print(tmp3_1["Assortment"].sum())
train3 =pd.concat([tmp3_1.drop('Store', axis=1), pd.get_dummies(tmp3_1['Store']).iloc[:, :-1]], axis=1)
#実は、trainにあってtestにない、Storeがあるので、回帰する前にちゃんと抜いてあげる
x_test2 = test[['Store', 'DayOfWeek', 'Promo']]
idx = x_test2["Store"].unique()
len(idx)
idx = idx[0:len(idx)-1]
idx_list = idx.tolist()
idx_list.append("DayOfWeek")
idx_list.append("Promo")
idx_list.append("Sales")
idx_list.append("Month")
idx_list.append("Assortment")
train3_modifyed = train3[idx_list]
train3_modifyed  = pd.concat([train3_modifyed .drop('DayOfWeek', axis=1), pd.get_dummies(train3_modifyed ['DayOfWeek']).iloc[:, :-1]], axis=1)
train3_modifyed  = pd.concat([train3_modifyed .drop('Promo', axis=1), pd.get_dummies(train3_modifyed ['Promo']).iloc[:, :-1]], axis=1)
train3_modifyed.head()
y_train3_modifyed = train3_modifyed[["Sales"]]
x_train3_modifyed = train3_modifyed.drop(["Sales"], axis = 1)
model3 = LinearRegression()
model3.fit(x_train3_modifyed,y_train3_modifyed)
