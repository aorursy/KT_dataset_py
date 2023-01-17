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
#1.データ、ライブラリの読み込み
# data wrangling
import numpy as np
import pandas as pd
#import pandas_profiling as pdp
from collections import Counter

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display import display

# modeling
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate

# evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#正規表現
import re
#データの読み込み
train = pd.read_csv('../input/rossmann-store-sales/train.csv')
test = pd.read_csv('../input/rossmann-store-sales/test.csv')
store = pd.read_csv('../input/rossmann-store-sales/store.csv')

#いつでも元データを取り出せるようにしておく（0:train, 1:test, 2:store）
def gen_data():
    return([train,test,store])
#加工用にデータを分離する
x_train = gen_data()[0]
x_test = gen_data()[1]
x_store = gen_data()[2]
#2.1.特徴量の構成（trainの加工）
#データの外形を確認
display(x_train.head())
#欠損値、データタイプの確認
display(x_train.info())
#trainには欠損なし。"Date"と"StateHoliday"がobjectなので、線形モデルに取り込めるように変換する。
#"Date"の処理
#"Date" は時系列データとしてではなく、"DayOfWeek"（月曜：1～日曜：7）のみ取り込むこととにする。
x_train = pd.concat([x_train.drop("DayOfWeek", axis=1), pd.get_dummies(x_train["DayOfWeek"]).iloc[:, :-1]], axis=1)
x_train = x_train.drop("Date",axis=1)
x_train.head()
#"StateHoliday"の処理
#内容確認
x_train["StateHoliday"] = x_train["StateHoliday"].astype(str)
import collections
print(collections.Counter(x_train["StateHoliday"]))

#'StateHoliday'はダミー変数で置き換える。(One hot encoding)
x_train = pd.concat([x_train.drop('StateHoliday', axis=1), pd.get_dummies(x_train['StateHoliday']).iloc[:, :-1]], axis=1)
x_train.head()
#trainへの処理は終わり
x_train.info()
#2.2.特徴量の構成（testの加工）

#test　「open(float64)」に欠損あり。
display(x_test.info())

#最頻値で置き換える。
x_test['Open'].fillna(x_test['Open'].mode()[0], inplace=True)
#"Date"、"StateHoliday"の処理（trainと同じ）
#"Date"の処理
x_test = pd.concat([x_test.drop("DayOfWeek", axis=1), pd.get_dummies(x_test["DayOfWeek"]).iloc[:, :-1]], axis=1)
x_test = x_test.drop("Date",axis=1)
display(x_test.head())
#"StateHoliday"の処理
x_test = pd.concat([x_test.drop('StateHoliday', axis=1), pd.get_dummies(x_test['StateHoliday']).iloc[:, :-1]], axis=1)
display(x_test.head())
#stateholidayの種類が足りないので追加してあげる
#testへの処理は終わり
x_test["a"]=0
x_test["b"]=0
display(x_test.head())
#2.3.特徴量の構成（storeの加工）
x_store.info()
#"CompetitionDistance"、"CompetitionOpenSinceMonth"、"CompetitionOpenSinceYear"、
#"Promo2SinceWeek"、"Promo2SinceYear"、"PromoInterval"に欠損あり
#欠損値は次のとおり置き換える。
#いずれも雑な気がするので、後でちゃんと考えること、、、

#"CompetitionDistance"は平均で置き換える
x_store['CompetitionDistance'].fillna(x_store['CompetitionDistance'].mean(), inplace=True)

#"CompetitionOpenSinceMonth"と"CompetitionOpenSinceYear"は最頻値で置き換える。
x_store['CompetitionOpenSinceMonth'].fillna(x_store['CompetitionOpenSinceMonth'].mode()[0], inplace=True)
x_store['CompetitionOpenSinceYear'].fillna(x_store['CompetitionOpenSinceYear'].mode()[0], inplace=True)

#Note: "Promo2SinceWeek","Promo2SinceYear","PromoInterval"は"Promo2"が「0」の場合、自動的に空白となる。
#"Promo2SinceWeek"→0, "Promo2SinceYear"→2016, "PromoInterval"→0　と置換する。※2015年9月17日時点で始まっていないの意
x_store['Promo2SinceWeek'].fillna(0, inplace=True)
x_store['Promo2SinceYear'].fillna(2016, inplace=True)
x_store['PromoInterval'].fillna(0, inplace=True)
#オブジェクトタイプの変数をすべてダミー変数にする
#これも雑なので後でちゃんと考える、、、
x_store = pd.concat([x_store.drop('StoreType', axis=1), pd.get_dummies(x_store['StoreType']).iloc[:, :-1]], axis=1)
x_store = pd.concat([x_store.drop('Assortment', axis=1), pd.get_dummies(x_store['Assortment']).iloc[:, :-1]], axis=1)
x_store = pd.concat([x_store.drop('PromoInterval', axis=1), pd.get_dummies(x_store['PromoInterval']).iloc[:, :-1]], axis=1)
x_store.head()
#3. 線形回帰で予測値を算出する

#train・testとstoreを結合
#trainとstoreを結合すると何故か"Id"が消える、、、
x_train_store = pd.merge(x_train, x_store, how = 'inner', on = 'Store')

x_test_store = pd.merge(x_test, x_store, how = 'inner', on = 'Store')
#trainから"Sales"と"Customers"を分離
y_train_store = x_train_store[["Sales"]]
x_train_store = x_train_store.drop(["Sales","Customers"], axis = 1)

#"Store"も回帰変数には含めないので落とす
x_train_store = x_train_store.drop(["Store"], axis = 1)
#testから"Id"を分離
id_test = x_test_store["Id"]
#"Store"も回帰変数には含めないので落とす
x_test_store = x_test_store.drop(["Id","Store"], axis = 1)
#３．線形回帰する
from sklearn.linear_model import LinearRegression

#線形回帰実行（普通の重回帰）
model = LinearRegression()
model.fit(x_train_store,y_train_store)
#線形回帰による予測
pred_linear_1 = model.predict(x_test_store)
#1次元配列に変換
pred_linear_1 = np.ravel(pred_linear_1)
#提出用にデータを結合
submission_linear = pd.DataFrame({
        "Id": id_test,
        "Sales": pred_linear_1
    })

#"Id"順に並べ替え
submission_linear = submission_linear.sort_values('Id')

#0より小さい予測値を0に変換
submission_linear["Sales"] = submission_linear["Sales"].apply(lambda x:max(x,0))

submission_linear.to_csv('./submission_linear.csv', index = False )
#回帰係数が大きすぎる？
pd.DataFrame({"変数":np.ravel(x_train_store.columns.values), "係数":np.ravel(model.coef_)})
#RMSPEを算出
def gen_RMSPE(pred, ans):
    tmp_0 = (pred - ans)/ans
    tmp_1 = tmp_0[np.isfinite(tmp_0)] #ansが0のレコード(=tmp0がinfのレコード)は無視する
    return(np.sqrt(np.power(tmp_1,2).sum()/tmp_1.shape[0]))
from sklearn.metrics import mean_squared_log_error
#各foldのスコアを保存するリスト
scores_RMSPE = []

#クロスバリデーションを行う
#学習データを4つに分割し、うち1つをバリデーションデータとすることをバリデーションデータを変えて繰り返す
kf = KFold(n_splits=4, shuffle =True, random_state=71)

for tr_idx, va_idx in kf.split(x_train_store):
    #学習データを学習データとバリデーションデータに分ける
    tr_x, va_x = x_train_store.iloc[tr_idx], x_train_store.iloc[va_idx]
    tr_y, va_y = y_train_store.iloc[tr_idx], y_train_store.iloc[va_idx]
    
    #モデルの学習を行う 
    model_cv = LinearRegression()
    model_cv.fit(tr_x,tr_y)
    
    #バリデーションデータの予測値を確率で出力する
    va_pred = pd.DataFrame(model_cv.predict(va_x)[:len(model_cv.predict(va_x))])
    va_pred[0] = va_pred[0].apply(lambda x:0 if x<0 else x)
    #pd.DataFrame({"Sales_pred":model_cv.predict(va_x)[:len(model_cv.predict(va_x))]})
    
    #バリデーションデータでスコアを計算する
    RMSPE = gen_RMSPE(va_pred[0], va_y["Sales"])
        
    #そのfoldスコアを保持する
    scores_RMSPE.append(RMSPE)
scores_RMSPE
print(pd.DataFrame({"変数":np.ravel(x_train_store.columns.values), "係数":np.ravel(model_cv.coef_)}))
print({"定数項":model_cv.intercept_})

#変数選択
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

#変数選択はとりあえずLASSOで、正則化パラメータは10とする（適当）。

scaler = StandardScaler()
clf = Lasso(alpha=10)

#30秒くらいかかった
scaler.fit(x_train_store)
clf.fit(scaler.transform(x_train_store), y_train_store)
#一部の変数の寄与が0になった
pd.DataFrame({"変数":np.ravel(x_train_store.columns.values), "係数":np.ravel(clf.coef_)})
#線形回帰(Lasso)による予測
scaler.fit(x_test_store)
pred_linear_2 = clf.predict(scaler.transform(x_test_store))
#1次元配列に変換
pred_linear_2 = np.ravel(pred_linear_2)

#提出用にデータを結合
submission_linear_lasso = pd.DataFrame({
        "Id": id_test,
        "Sales": pred_linear_2
    })

#"Id"順に並べ替え
submission_linear_lasso = submission_linear_lasso.sort_values('Id')

#0より小さい予測値を0に変換
submission_linear_lasso["Sales"] = submission_linear_lasso["Sales"].apply(lambda x:max(x,0))

submission_linear_lasso.to_csv('./submission_linear_lasso.csv', index = False )
pred_linear_2
#train・testとstoreを結合
#trainとstoreを結合すると何故か"Id"が消える、、、
x_train_store = pd.merge(x_train, x_store, how = 'inner', on = 'Store')
x_test_store = pd.merge(x_test, x_store, how = 'inner', on = 'Store')

#単価を算出する
x_train_store["Unit"] = x_train_store["Sales"] /x_train_store["Customers"] 
#"Customers"=0 のときInfになるので0に置き換える。
x_train_store["Unit"] = x_train_store["Unit"] .fillna(0)
#trainから"Sales", "Customers", "Unit"を分離、うち"Sales"は捨てる
y_train_store = x_train_store.loc[:,["Customers","Unit"]]
x_train_store = x_train_store.drop(["Sales","Customers","Unit"], axis = 1)

#"Store"も回帰変数には含めないので落とす
x_train_store = x_train_store.drop(["Store"], axis = 1)
#変数選択はとりあえずLASSOで、正則化パラメータは10とする（適当）。

scaler = StandardScaler()
clf_cust = Lasso(alpha=10) #適当
clf_unit = Lasso(alpha=0.1) #適当。10だと退化してしまったので、、

#30秒くらいかかった
scaler.fit(x_train_store)
clf_cust.fit(scaler.transform(x_train_store), y_train_store["Customers"])
clf_unit.fit(scaler.transform(x_train_store), y_train_store["Unit"])
#testから"Id"を分離
id_test = x_test_store["Id"]
#"Store"も回帰変数には含めないので落とす
x_test_store = x_test_store.drop(["Id","Store"], axis = 1)

#CostomerとUnitの積を予測値とする。
pred_linear_3 = clf_cust.predict(scaler.transform(x_test_store))*clf_unit.predict(scaler.transform(x_test_store))
#1次元配列に変換
pred_linear_3 = np.ravel(pred_linear_3)

#提出用にデータを結合
submission_linear_lasso_unit = pd.DataFrame({
        "Id": id_test,
        "Sales": pred_linear_3
    })

#"Id"順に並べ替え
submission_linear_lasso_unit = submission_linear_lasso_unit.sort_values('Id')

#0より小さい予測値を0に変換
submission_linear_lasso_unit["Sales"] = submission_linear_lasso_unit["Sales"].apply(lambda x:max(x,0))

submission_linear_lasso_unit.to_csv('./submission_linear_lasso_unit.csv', index = False )
# submission_linear_lasso_unitのスコアは0.43454となった。
#加工用にデータを分離する
train2 = gen_data()[0]
test2 = gen_data()[1]
store2 = gen_data()[2]
plt.figure(figsize=(10,10))
sns.set(style="whitegrid")
sns.boxplot(data=train2,x="DayOfWeek",y="Sales")
print("曜日ごとに箱ひげ図を描画し、外れ値の検証をした。曜日によってまちまちだが20000超は外れ値扱いでよさそう。")
#曜日ごとのばらつきを確認
sales_DayOfWeek_df=pd.DataFrame({"Avg SalesPerDoW":train2["Sales"],"DayOfWeek":train2["DayOfWeek"]})
AvgSalesDayOfWeek=sales_DayOfWeek_df.groupby("DayOfWeek").mean()
print("曜日ごとに一定のばらつきがある。")
print(plt.plot(AvgSalesDayOfWeek, marker = "o"))
#外れ値を特定の値で置き換える(Clipping)
train2["Sales"]=train2["Sales"].apply(lambda x: 20000 if x>20000 else x)
#Dateの情報を活用する
train2["Date"]=pd.to_datetime(train2["Date"])
train2["Year"]=train2["Date"].dt.year
train2["Month"]=train2["Date"].dt.month
train2["Day"]=train2["Date"].dt.day
#その月の第何週かに無理やり読み替える
train2["Week"]=train2["Date"].dt.week%4
#季節別
train2["Season"] = np.where(train2["Month"].isin([3,4]),"Spring",np.where(train2["Month"].isin([5,6,7,8]), "Summer",np.where(train2["Month"].isin ([9,10,11]),"Fall",np.where(train2["Month"].isin ([12,1,2]),"Winter","None"))))
#月別売上高のばらつきを確認
sales_time_df=pd.DataFrame({"Avg SalesPerMonth":train2["Sales"],"Month":train2["Month"]})
AvgCustomerperMonth=sales_time_df.groupby("Month").mean()
print("月ごとに一定のばらつきがある。")
print(plt.plot(AvgCustomerperMonth, marker = "o"))
train_store2=store2.merge(train2,on=["Store"],how="inner")
#store type別のばらつきを確認
stype_df=pd.DataFrame({"Avg storetype":train_store2["Sales"],"StoreType":train_store2["StoreType"]})
Avgstoretype=stype_df.groupby("StoreType").mean()
print("「b」だけ明らかに高い")
print(plt.plot(Avgstoretype, marker = "o"))
#Assortment別のばらつきを確認
Assortment_df=pd.DataFrame({"Avg Assortment":train_store2["Sales"],"Assortment":train_store2["Assortment"]})
AvgAssortment=Assortment_df.groupby("Assortment").mean()
print("「b」が高いが各々差分がある")
print(plt.plot(AvgAssortment, marker = "o"))
#kaggleのカーネルでは動かない、why...

#StateHoliday別のばらつきを確認
#StateHoliday_df=pd.DataFrame({"Avg StateHoliday":train_store2["Sales"],"StateHoliday":train_store2["StateHoliday"]})
#AvgStateHoliday=StateHoliday_df.groupby("StateHoliday").mean()
#print("「0」が高く、あとは一律に低い")
#print(plt.plot(AvgStateHoliday, marker = "o"))
#Promo2別のばらつきを確認
Promo2_df=pd.DataFrame({"Avg Promo2":train_store2["Sales"],"Promo2":train_store2["Promo2"]})
AvgPromo2=Promo2_df.groupby("Promo2").mean()
print("以外にも「0」が高い")
print(plt.plot(AvgPromo2, marker = "o"))
#欠損値になっている箇所は一旦無視する
train_store2.info()
drop_list = ["CompetitionDistance","CompetitionOpenSinceMonth","CompetitionOpenSinceYear","Promo2SinceWeek","Promo2SinceWeek","Promo2SinceYear","PromoInterval"]
feature = train_store2.drop(drop_list, axis=1)
#特徴量の構成
#"Month"は12月かそれ以外か
feature["Month"] = feature["Month"].apply(lambda x: 1 if x==12 else 0)
#"Assortment"は「b」かそれ以外か   ##ダミー変数にする
#feature = pd.concat([feature.drop('Assortment', axis=1), pd.get_dummies(feature['Assortment']).iloc[:, :-1]], axis=1)
feature["Assortment"] = feature["Assortment"].apply(lambda x: 1 if x=="b" else 0)
#"DayOfWeek"は日曜(「7」)かそれ以外か
feature["DayOfWeek"] = feature["DayOfWeek"].apply(lambda x: 1 if x==7 else 0)
#"Store Type"は「b」かそれ以外か
feature["StoreType"] = feature["StoreType"].apply(lambda x: 1 if x=="b" else 0)
#"StateHoliday"は「"0"」かそれ以外か
feature["StateHoliday"] = feature["StateHoliday"].apply(lambda x: 0 if x=="0" else 1)
import collections
collections.Counter(train_store2["Assortment"])
#細かい変数を落とす
#"Week"は残す
drop_list2 = ["Store","Date","Year","Day","Season"]
feature2 = feature.drop(drop_list2, axis=1)
feature2.head()
#回帰用の変数を作成
y_feature2 = feature["Sales"]
x_feature2 = feature2.drop(["Sales","Customers"],axis=1)
x_feature2 = feature2.drop(["Sales","Customers"],axis=1)
x_feature2["Open_stHoli"] = feature["Open"]*x_feature2["StateHoliday"]
print(x_feature2.head())
print(x_feature2.shape)
#３．線形回帰する
from sklearn.linear_model import LinearRegression

#線形回帰実行（普通の重回帰）
model_0 = LinearRegression()
model_0.fit(x_feature2,y_feature2)
coeficient = pd.DataFrame({"変数":x_feature2.columns,"係数":model_0.coef_})
print(coeficient)
print({"定数項":model_0.intercept_})
pred = pd.DataFrame({"Sales_pred":model_0.predict(x_feature2)})
from sklearn.metrics import mean_squared_error
#各foldのスコアを保存するリスト
scores_RMSPE = []

#クロスバリデーションを行う
#学習データを4つに分割し、うち1つをバリデーションデータとすることをバリデーションデータを変えて繰り返す
kf = KFold(n_splits=4, shuffle =True, random_state=72)

for tr_idx, va_idx in kf.split(x_feature2):
    #学習データを学習データとバリデーションデータに分ける
    tr_x, va_x = x_feature2.iloc[tr_idx], x_feature2.iloc[va_idx]
    tr_y, va_y = y_feature2.iloc[tr_idx], y_feature2.iloc[va_idx]
    
    #モデルの学習を行う 
    model_cv = LinearRegression()
    model_cv.fit(tr_x,tr_y)
    
    #バリデーションデータの予測値を確率で出力する
    va_pred = pd.DataFrame({"Sales_pred":model_cv.predict(va_x)})
    
    #バリデーションデータでスコアを計算する
    RMSPE = gen_RMSPE(va_pred["Sales_pred"], va_y)
        
    #そのfoldスコアを保持する
    scores_RMSPE.append(RMSPE)
scores_RMSPE
#残差分析
#以下、コンペの趣旨に沿ってSalesが0の先は対象外とする
feature["pred_Sales"] = pred
feature_analysis = feature[feature["Sales"]>0]
feature_analysis["Residuals"] = np.power((feature_analysis["pred_Sales"] - feature_analysis["Sales"])/feature_analysis["Sales"], 2)
feature_analysis.head(30)
#残差の大きいレコードを抽出する
res_anal = feature_analysis.sort_values('Residuals',ascending=False)
res_anal.head(30)
