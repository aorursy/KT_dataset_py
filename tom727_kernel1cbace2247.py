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
# ライブラリの呼び出し



import warnings

warnings.filterwarnings('ignore')



import numpy as np 

import pandas as pd 

import seaborn as sns #시각화를 위한 라이브러리

import matplotlib.pyplot as plt

import calendar 

from datetime import datetime



import os

print(os.listdir("../input"))
# 訓練、テストデータセットの形とデータのカラムの属性と値の数を把握



train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')



train.head()
"""カラムの説明



datetime - hourly date + timestamp  

season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 

holiday - whether the day is considered a holiday

workingday - whether the day is neither a weekend nor holiday

weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy 

2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 

3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 

4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 

temp - temperature in Celsius

atemp - "feels like" temperature in Celsius

humidity - relative humidity

windspeed - wind speed

casual - number of non-registered user rentals initiated

registered - number of registered user rentals initiated

count - number of total rentals

"""



train.info()
# 中身の確認

test.head()
# データの前処理と可視化

# split関数を使用して年 - 月 - 日と時間を分離

train['tempDate'] = train.datetime.apply(lambda x:x.split())
# 分離したtempDateの年-月-日を利用してyear、month、dayとweekdayのカラムを抽出



train['year'] = train.tempDate.apply(lambda x:x[0].split('-')[0])

train['month'] = train.tempDate.apply(lambda x:x[0].split('-')[1])

train['day'] = train.tempDate.apply(lambda x:x[0].split('-')[2])



#weekdayはcalendarとdatetimeを利用

train['weekday'] = train.tempDate.apply(lambda x:calendar.day_name[datetime.strptime(x[0],"%Y-%m-%d").weekday()])



train['hour'] = train.tempDate.apply(lambda x:x[1].split(':')[0])
# 分離して抽出された属性が文字列であるため数値データに変換



train['year'] = pd.to_numeric(train.year,errors='coerce')

train['month'] = pd.to_numeric(train.month,errors='coerce')

train['day'] = pd.to_numeric(train.day,errors='coerce')

train['hour'] = pd.to_numeric(train.hour,errors='coerce')
# 変換をしたため確認



train.info()
# tempDateのカラムを除去



train = train.drop('tempDate',axis=1)
# 各カラムとcountの相関



#year - count

fig = plt.figure(figsize=[12,10])

ax1 = fig.add_subplot(2,2,1)

ax1 = sns.barplot(x='year',y='count',data=train.groupby('year')['count'].mean().reset_index())



#month - count

ax2 = fig.add_subplot(2,2,2)

ax2 = sns.barplot(x='month',y='count',data=train.groupby('month')['count'].mean().reset_index())



#day - count

ax3 = fig.add_subplot(2,2,3)

ax3 = sns.barplot(x='day',y='count',data=train.groupby('day')['count'].mean().reset_index())



#hour - count

ax4 = fig.add_subplot(2,2,4)

ax4 = sns.barplot(x='hour',y='count',data=train.groupby('hour')['count'].mean().reset_index())
#season - count

fig = plt.figure(figsize=[12,10])

ax1 = fig.add_subplot(2,2,1)

ax1 = sns.barplot(x='season',y='count',data=train.groupby('season')['count'].mean().reset_index())



#holiday - count

ax2 = fig.add_subplot(2,2,2)

ax2 = sns.barplot(x='holiday',y='count',data=train.groupby('holiday')['count'].mean().reset_index())



#workingday - count

ax3 = fig.add_subplot(2,2,3)

ax3 = sns.barplot(x='workingday',y='count',data=train.groupby('workingday')['count'].mean().reset_index())



#weather - count

ax4 = fig.add_subplot(2,2,4)

ax4 = sns.barplot(x='weather',y='count',data=train.groupby('weather')['count'].mean().reset_index())
def badToRight(month):

    if month in [12,1,2]:

        return 4

    elif month in [3,4,5]:

        return 1

    elif month in [6,7,8]:

        return 2

    elif month in [9,10,11]:

        return 3

    

train['season'] = train.month.apply(badToRight)
# 1つの列と結果の値を比較



#season - count

fig = plt.figure(figsize=[12,10])

ax1 = fig.add_subplot(2,2,1)

ax1 = sns.barplot(x='season',y='count',data=train.groupby('season')['count'].mean().reset_index())



#holiday - count

ax2 = fig.add_subplot(2,2,2)

ax2 = sns.barplot(x='holiday',y='count',data=train.groupby('holiday')['count'].mean().reset_index())



#woikingday - count

ax3 = fig.add_subplot(2,2,3)

ax3 = sns.barplot(x='workingday',y='count',data=train.groupby('workingday')['count'].mean().reset_index())



#weather - count

ax4 = fig.add_subplot(2,2,4)

ax4 = sns.barplot(x='weather',y='count',data=train.groupby('weather')['count'].mean().reset_index())
# 相関係数をheatmapを介して可視化



fig = plt.figure(figsize=[20,20])

ax = sns.heatmap(train.corr(),annot=True,square=True)
# heatmap相関を参照して二つの異なるカラムとcountを視覚化



#hour season - count

fig = plt.figure(figsize=[12,10])

ax1 = fig.add_subplot(2,2,1)

ax1 = sns.pointplot(x='hour',y='count',hue='season',data=train.groupby(['season','hour'])['count'].mean().reset_index())



#hour holiday - count

ax2 = fig.add_subplot(2,2,2)

ax2 = sns.pointplot(x='hour',y='count',hue='holiday',data=train.groupby(['holiday','hour'])['count'].mean().reset_index())



#hour weekday - count

ax3 = fig.add_subplot(2,2,3)

ax3 = sns.pointplot(x='hour',y='count',hue='weekday',hue_order=['Sunday','Monday','Tuesday','Wendnesday','Thursday','Friday','Saturday'],data=train.groupby(['weekday','hour'])['count'].mean().reset_index())



#hour weather - count

ax4 = fig.add_subplot(2,2,4)

ax4 = sns.pointplot(x='hour',y='count',hue='weather',data=train.groupby(['weather','hour'])['count'].mean().reset_index())
# 最後の可視化に異常値があるため確認



train[train.weather==4]
#month, weather - count 

fig = plt.figure(figsize=[12,10])

ax1 = fig.add_subplot(2,1,1)

ax1 = sns.pointplot(x='month',y='count',hue='weather',data=train.groupby(['weather','month'])['count'].mean().reset_index())



#month count

ax2 = fig.add_subplot(2,1,2)

ax2 = sns.barplot(x='month',y='count',data=train.groupby('month')['count'].mean().reset_index())
"""

Windspeed分布を表現したグラフでWindspeedが0である値が多かったが、

これは実際には0であったかor値を正しく測定できなくて0である二つの場合がある

後者の場合を考えて、windspeed値を付与する

"""



# 文字列をカテゴリー化し、それぞれに対応する数値に変換



train['weekday']= train.weekday.astype('category')

print(train['weekday'].cat.categories)
#0:Sunday --> 6:Saturday

train.weekday.cat.categories = ['5','1','6','0','4','2','3']
# ランダムフォレストを使ってwindspeedを付与



from sklearn.ensemble import RandomForestRegressor



# Windspeedが0のデータフレーム

windspeed_0 = train[train.windspeed == 0]

# Windspeedが0でないデータフレーム

windspeed_Not0 = train[train.windspeed != 0]



# Windspeedが0であるデータフレームから不要なカラムを除去

windspeed_0_df = windspeed_0.drop(['windspeed','casual','registered','count','datetime'],axis=1)



# Windspeedが0でないデータフレームはそのまま学習

windspeed_Not0_df = windspeed_Not0.drop(['windspeed','casual','registered','count','datetime'],axis=1)

windspeed_Not0_series = windspeed_Not0['windspeed'] 



# モデルに0以外のデータフレームと結果の値を学習させる

rf = RandomForestRegressor()

rf.fit(windspeed_Not0_df,windspeed_Not0_series)



# 学習したモデルからWindspeedが0であるデータフレームのWindspeedを導出

predicted_windspeed_0 = rf.predict(windspeed_0_df)



# 導出された値を元のデータフレームに挿入

windspeed_0['windspeed'] = predicted_windspeed_0
# 分割したデータフレームを元の形に復元

train = pd.concat([windspeed_0,windspeed_Not0],axis=0)
# 時間ごとのソートのためにstring typeのdatetimeを変換

train.datetime = pd.to_datetime(train.datetime,errors='coerce')
# 統合したデータをdatetime順に並べ替える

train = train.sort_values(by=['datetime'])
# windspeedを修正した後、再び相関係数を分析

fig = plt.figure(figsize=[20,20])

ax = sns.heatmap(train.corr(),annot=True,square=True)
fig = plt.figure(figsize=[5,5])

sns.distplot(train['windspeed'],bins=np.linspace(train['windspeed'].min(),train['windspeed'].max(),10))

plt.suptitle("Filled by Random Forest Regressor")

print("Min value of windspeed is {}".format(train['windspeed'].min()))
# 同様の前処理をtestとtrainの統合データで行う



train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
combine = pd.concat([train,test],axis=0)

combine.info()
combine['tempDate'] = combine.datetime.apply(lambda x:x.split())

combine['weekday'] = combine.tempDate.apply(lambda x: calendar.day_name[datetime.strptime(x[0],"%Y-%m-%d").weekday()])

combine['year'] = combine.tempDate.apply(lambda x: x[0].split('-')[0])

combine['month'] = combine.tempDate.apply(lambda x: x[0].split('-')[1])

combine['day'] = combine.tempDate.apply(lambda x: x[0].split('-')[2])

combine['hour'] = combine.tempDate.apply(lambda x: x[1].split(':')[0])
combine['year'] = pd.to_numeric(combine.year,errors='coerce')

combine['month'] = pd.to_numeric(combine.month,errors='coerce')

combine['day'] = pd.to_numeric(combine.day,errors='coerce')

combine['hour'] = pd.to_numeric(combine.hour,errors='coerce')
combine.info()
combine['season'] = combine.month.apply(badToRight)
combine.weekday = combine.weekday.astype('category')

combine.weekday.cat.categories = ['5','1','6','0','4','2','3']

dataWind0 = combine[combine['windspeed']==0]

dataWindNot0 = combine[combine['windspeed']!=0]



dataWind0.columns
dataWind0_df = dataWind0.drop(['windspeed','casual','registered','count','datetime','tempDate'],axis=1)



dataWindNot0_df = dataWindNot0.drop(['windspeed','casual','registered','count','datetime','tempDate'],axis=1)

dataWindNot0_series = dataWindNot0['windspeed']



dataWindNot0_df.head()
rf2 = RandomForestRegressor()

rf2.fit(dataWindNot0_df,dataWindNot0_series)

predicted = rf2.predict(dataWind0_df)

print(predicted)
dataWind0['windspeed'] = predicted

combine = pd.concat([dataWind0,dataWindNot0],axis=0)
# 不要なカラムの除去

categorizational_columns = ['holiday','humidity','season','weather','workingday','year','month','day','hour']

drop_columns = ['datetime','casual','registered','count','tempDate']
#数値に変換

for col in categorizational_columns:

    combine[col] = combine[col].astype('category')
# 統合したデータセットからcountの有無により訓練とテストセットを分離し、それぞれをdatetimeに並べ替え

train = combine[pd.notnull(combine['count'])].sort_values(by='datetime')

test = combine[~pd.notnull(combine['count'])].sort_values(by='datetime')



# データを訓練した結果

datetimecol = test['datetime']

yLabels = train['count'] #count

yLabelsRegistered = train['registered'] #登録者

yLabelsCasual = train['casual'] #一時的なユーザー
# 必要ないcolumnを除去した後のtrainとtest

train = train.drop(drop_columns,axis=1)

test = test.drop(drop_columns,axis=1)
"""

この問題では、RMSLE方式を利用して、適切に予測がされたか評価することになる。

RMSLEは、以下のリンクを参照して、利用。

https://programmers.co.kr/learn/courses/21/lessons/943#



RMSLE

過大評価された項目ではなく、過小評価された項目にペナルティを与える方式

誤差を二乗してヒョンギュンした値の平方根で値が小さくなるほど精度が高い

0に近い値が出てくるほど精度が高い

"""



# y is predict value y_ is actual value

def rmsle(y, y_,convertExp=True):

    if convertExp:

        y = np.exp(y), 

        y_ = np.exp(y_)

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
# 線形回帰モデル

# 線形回帰モデルは、触れるだけの内部attrがない

from sklearn.linear_model import LinearRegression,Ridge,Lasso





lr = LinearRegression()



"""

下のカーネルを参照してyLabelsをログ化しようとして、なぜnp.logではなく、np.log1pを活用するか？

np.log1pはnp.log（1+ x）と同じです。理由は、もしあれば、xの値が0であるが、これをlogになると、（ - ）無限大に収束するので、np.log1pを活用する。

"""

yLabelslog = np.log1p(yLabels)

#線形モデルに、私たちのデータを学習

lr.fit(train,yLabelslog)

#結果値導出

preds = lr.predict(train)

#rmsle関数のelementにnp.exp（）指数関数をとる理由は、私たちのpreds値に得られたのは、一度logをした値であるため、元のモデルには、logをしていない元の値を入れるウィハムイム。

print('RMSLE Value For Linear Regression: {}'.format(rmsle(np.exp(yLabelslog),np.exp(preds),False)))
"""

GridSearchCVを活用すれば、私たちが利用するようになるの各モデルごとに変更する必要がパラメータチューニング時にどのパラメータが最適値を出すのかなどを知ることができている。



GridSearchCV参照：

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

https://datascienceschool.net/view-notebook/ff4b5d491cc34f94aea04baca86fbef8/

"""

from sklearn.model_selection import GridSearchCV

from sklearn import metrics



#RidgeモデルはL2制約を持つ線形回帰モデルで改善されたモデルであり、そのモデルで有意深くチューニングする必要があるパラメータは、alpha値である。

ridge = Ridge()



#私たちは、チューニングしたいRidgeのパラメータのうちの特定のパラメータに配列の値に引き渡すと、テストした後どのようなパラメータが最適の値であるか知らせる

ridge_params = {'max_iter':[3000],'alpha':[0.001,0.01,0.1,1,10,100,1000]}

rmsle_scorer = metrics.make_scorer(rmsle,greater_is_better=False)

grid_ridge = GridSearchCV(ridge,ridge_params,scoring=rmsle_scorer,cv=5)



grid_ridge.fit(train,yLabelslog)

preds = grid_ridge.predict(train)

print(grid_ridge.best_params_)

print('RMSLE Value for Ridge Regression {}'.format(rmsle(np.exp(yLabelslog),np.exp(preds),False)))
#結果についてGridSearchCVの変数であるgrid_ridge変数にcv_result_を通じてalpha値の変化に応じて、平均値の変化を把握可能

df = pd.DataFrame(grid_ridge.cv_results_)
df.head()
#RidgeモデルはL1制約を持つ線形回帰モデルで改善されたモデルであり、そのモデルで有意深くチューニングする必要があるパラメータは、alpha値である。

lasso = Lasso()



lasso_params = {'max_iter':[3000],'alpha':[0.001,0.01,0.1,1,10,100,1000]}

grid_lasso = GridSearchCV(lasso,lasso_params,scoring=rmsle_scorer,cv=5)

grid_lasso.fit(train,yLabelslog)

preds = grid_lasso.predict(train)

print('RMSLE Value for Lasso Regression {}'.format(rmsle(np.exp(yLabelslog),np.exp(preds),False)))
rf = RandomForestRegressor()



rf_params = {'n_estimators':[1,10,100]}

grid_rf = GridSearchCV(rf,rf_params,scoring=rmsle_scorer,cv=5)

grid_rf.fit(train,yLabelslog)

preds = grid_rf.predict(train)

print('RMSLE Value for RandomForest {}'.format(rmsle(np.exp(yLabelslog),np.exp(preds),False)))
from sklearn.ensemble import GradientBoostingRegressor



gb = GradientBoostingRegressor()

gb_params={'max_depth':range(1,11,1),'n_estimators':[1,10,100]}

grid_gb=GridSearchCV(gb,gb_params,scoring=rmsle_scorer,cv=5)

grid_gb.fit(train,yLabelslog)

preds = grid_gb.predict(train)

print('RMSLE Value for GradientBoosting {}'.format(rmsle(np.exp(yLabelslog),np.exp(preds),False)))
predsTest = grid_gb.predict(test)

fig,(ax1,ax2)= plt.subplots(ncols=2)

fig.set_size_inches(12,5)

sns.distplot(yLabels,ax=ax1,bins=50)

sns.distplot(np.exp(predsTest),ax=ax2,bins=50)
submission = pd.DataFrame({

        "datetime": datetimecol,

        "count": [max(0, x) for x in np.exp(predsTest)]

    })

submission.to_csv('bike_predictions_gbm_separate_without_fe.csv', index=False)