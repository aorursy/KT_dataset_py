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
sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

sample_submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

train = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
train.sum(axis=1).sort_values(ascending=False)
train[8412:8413]
FOODS_3_090_CA_3 = sell_prices[sell_prices["store_id"]=="CA_3"][sell_prices["item_id"]=="FOODS_3_090"]
FOODS_3_090_CA_3
calendar = calendar[calendar["wm_yr_wk"]<=11621]

calendar
df = pd.merge(calendar, FOODS_3_090_CA_3, on='wm_yr_wk', how='outer')
sample_submission.head(1)
df = df[:1942]
drop_list = ['id','item_id','dept_id','cat_id','store_id','state_id']

train.drop(drop_list,axis = 1,inplace = True)
train[8412:8413]
value_count_per_day = train[8412:8413].values
value_count_per_day.shape
value_count_per_day = value_count_per_day.reshape(1913,1)
drop_list =['wm_yr_wk','weekday','month','year','d','store_id','item_id']

df.drop(drop_list,axis=1, inplace=True)
df.fillna(0,inplace=True)
import category_encoders as ce



list_cols = ['event_name_1','event_type_1','event_name_2','event_type_2']



ce_oe = ce.OrdinalEncoder(cols=list_cols,handle_unknown='impute')

df = ce_oe.fit_transform(df, inplace=True)
df
df.shape
date_index=pd.date_range('2011-01-29', freq='D', periods=1942) # 周期に日を指定

date_index
df = df.set_index(date_index)
target = df[0:1913]
target.shape
target
target['sold'] = value_count_per_day
num = target['sold']
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')
plt.figure(figsize=(20, 10))

num.plot()
import statsmodels.api as sm

sm.tsa.seasonal_decompose(num)
res = sm.tsa.seasonal_decompose(num)
original = num # オリジナルデータ

trend = res.trend # トレンドデータ

seasonal = res.seasonal # 季節性データ

residual = res.resid # 残差データ



plt.figure(figsize=(15, 12)) # グラフ描画枠作成、サイズ指定



# オリジナルデータのプロット

plt.subplot(411) # グラフ4行1列の1番目の位置（一番上）

plt.plot(original)

plt.ylabel('Original')



# trend データのプロット

plt.subplot(412) # グラフ4行1列の2番目の位置

plt.plot(trend)

plt.ylabel('Trend')



# seasonalデータ のプロット

plt.subplot(413) # グラフ4行1列の3番目の位置

plt.plot(seasonal)

plt.ylabel('Seasonality')



# residual データのプロット

plt.subplot(414) # グラフ4行1列の4番目の位置（一番下）

plt.plot(residual)

plt.ylabel('Residuals')



plt.tight_layout() # グラフの間隔を自動調整
original = num # オリジナルデータ

trend = res.trend # トレンドデータ

seasonal = res.seasonal # 季節性データ

residual = res.resid # 残差データ

sum_three_data = trend + seasonal + residual # トレンド + 季節性 + 残差



plt.figure(figsize=(12, 9)) # グラフ描画枠作成、サイズ指定

plt.plot(original, label='original')

plt.plot(sum_three_data, label='trend +season +resid', linestyle='--')

plt.legend(loc='best') # 凡例表示
num_month_mean = num.groupby(num.index.month).mean()

num_month_mean.plot(kind='bar')

num_acf = sm.tsa.stattools.acf(num, nlags=40)

num_acf
# 自己相関係数（Numpy利用）

LAG = 40 # 計算ラグ数

rk = np.empty(LAG+1) # 自己相関係数の計算結果を保持用

y = np.array(num) # 販売数データのndarray作成（計算用）

y_mean = np.mean(y) # 販売数の平均値



# ラグ0の自己相関係数：1.0

rk[0] = np.sum((y - y_mean)**2) / np.sum((y - y_mean)**2) 



# ラグ1〜40の自己相関係数：-1.0〜1.0

for k in np.arange(1, LAG+1): 

    rk[k] = np.sum((y[k:] - y_mean)*(y[:-k] - y_mean)) / np.sum((y - y_mean)**2)



print(rk) # -> acf()結果と同じ
# 自己相関(ACF)のグラフ自動作成

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(111)

sm.graphics.tsa.plot_acf(num, lags=40, ax=ax1) #販売数データ、ラグ40、グラフaxes

plt.show() 
fig = plt.figure(figsize=(12, 8))



# 自己相関(ACF)のグラフ

ax1 = fig.add_subplot(211)

sm.graphics.tsa.plot_acf(num, lags=40, ax=ax1) #ACF計算とグラフ自動作成



# 偏自己相関(PACF)のグラフ

ax2 = fig.add_subplot(212)

sm.graphics.tsa.plot_pacf(num, lags=40, ax=ax2) #PACF計算とグラフ自動作成



plt.tight_layout() # グラフ間スキマ調整
num_diff = num - num.shift() # 階差系列データの作成
num.head()
print(len(num_diff))

num_diff.head() # 階差系列(1次階差)
num_diff = num_diff.dropna() # 1個できるNaNデータは捨てる
fig = plt.figure(figsize=(12, 4))



# 原型列のグラフ

ax1 = fig.add_subplot(121)

plt.plot(num)



# 階差系列のグラフ

ax2 = fig.add_subplot(122)

plt.plot(num_diff)



plt.tight_layout()
# 階差系列データでコレログラム作成

fig = plt.figure(figsize=(12, 8))



# 自己相関(ACF)のグラフ

ax1 = fig.add_subplot(211)

sm.graphics.tsa.plot_acf(num_diff, lags=40, ax=ax1) #ACF計算とグラフ自動作成



# 偏自己相関(PACF)のグラフ

ax2 = fig.add_subplot(212)

sm.graphics.tsa.plot_pacf(num_diff, lags=40, ax=ax2) #PACF計算とグラフ自動作成



plt.tight_layout()
# ADF検定（原型列で確認だけ）

res_ctt = sm.tsa.stattools.adfuller(num, regression="ctt") # トレンド項あり（２次）、定数項あり

res_ct = sm.tsa.stattools.adfuller(num, regression="ct") # トレンド項あり（１次）、定数項あり

res_c = sm.tsa.stattools.adfuller(num, regression="c") # トレンド項なし、定数項あり

res_nc = sm.tsa.stattools.adfuller(num, regression="nc") # トレンド項なし、定数項なし

print(res_ctt)

print(res_ct)

print(res_c)

print(res_nc)
import warnings

warnings.filterwarnings('ignore') # 計算警告を非表示



# 自動ARMAパラメータ推定関数

res_selection = sm.tsa.arma_order_select_ic(num_diff, ic='aic', trend='nc')

res_selection
# SRIMAモデル作成その１

sarimax = sm.tsa.SARIMAX(num, 

                        order=(4, 1, 1),

                        seasonal_order=(1, 1, 1, 12),

                        enforce_stationarity = False,

                        enforce_invertibility = False

                        ).fit()

sarimax_resid = sarimax.resid # モデルの残差成分
fig = plt.figure(figsize=(12, 8))



# 自己相関(ACF)のグラフ

ax1 = fig.add_subplot(211)

sm.graphics.tsa.plot_acf(sarimax_resid, lags=40, ax=ax1) #ACF計算とグラフ自動作成



# 偏自己相関(PACF)のグラフ

ax2 = fig.add_subplot(212)

sm.graphics.tsa.plot_pacf(sarimax_resid, lags=40, ax=ax2) #PACF計算とグラフ自動作成



plt.tight_layout()
# SRIMAモデル 季節調整なし

sarimax_noseasonal = sm.tsa.SARIMAX(num, 

                        order=(4, 1, 1),

                        seasonal_order=(0, 0, 0, 0),

                        enforce_stationarity = False,

                        enforce_invertibility = False

                        ).fit()



sarimax_noseasonal_resid = sarimax_noseasonal.resid # 残差成分



fig = plt.figure(figsize=(12, 8))



# 自己相関(ACF)のグラフ

ax1 = fig.add_subplot(211)

sm.graphics.tsa.plot_acf(sarimax_noseasonal_resid, lags=40, ax=ax1) #ACF計算とグラフ自動作成



# 偏自己相関(PACF)のグラフ

ax2 = fig.add_subplot(212)

sm.graphics.tsa.plot_pacf(sarimax_noseasonal_resid, lags=40, ax=ax2) #PACF計算とグラフ自動作成



plt.tight_layout()
print(sarimax.aic) # 季節調整あり

print(sarimax_noseasonal.aic) # 季節調整なし
sample_submission.shape
target_day=pd.date_range('2016-04-25', freq='D', periods=29) # 周期に日を指定

target_day
sarimax_pred = sarimax.predict('2016-03-01', '2016-05-23') 

plt.figure(figsize=(16, 4))

plt.plot(sarimax_pred, c="b")

sarimax_pred = sarimax.predict('2016-03-01', '2016-05-23') 



plt.figure(figsize=(20, 4))



plt.plot(num, label="original")

plt.plot(sarimax_pred, c="b", label="model-pred", alpha=0.7)

plt.legend(loc='best')
plt.figure(figsize=(8, 4))



plt.plot(num[1800:], label="actual") # 正解

plt.plot(sarimax_pred, c="b", label="predict", alpha=0.7) # 予測

plt.legend(loc='best')
predict_dy = sarimax.get_prediction(start ='2016-03-01', end='2016-05-23') # 未来予測

predict_dy_ci = predict_dy.conf_int() # 信頼区間取得



#　グラフ表示

plt.figure(figsize=(12, 4))

plt.plot(num[1800:], label="actual") # 実データプロット

plt.plot(predict_dy.predicted_mean, c="b", linestyle='--', label="model-pred", alpha=0.7) # 予測プロット



# 予測の95%信頼区間プロット（帯状）

plt.fill_between(predict_dy_ci.index, predict_dy_ci.iloc[:, 0], predict_dy_ci.iloc[:, 1], color='g', alpha=0.2)



plt.legend(loc='upper left')
sarimax_pred = sarimax.predict('2016-03-01', '2016-04-24') 
true = num['2016-03-01':'2016-04-24']
from sklearn.metrics import mean_squared_error



train_rmse = np.sqrt(mean_squared_error(true, sarimax_pred))



print('RMSE(train): {:.5}'.format(train_rmse))
#SARIMAパラメター最適化（総当たりチェック）

import warnings

warnings.filterwarnings('ignore') # 警告非表示（収束：ConvergenceWarning）



# パラメータ範囲

# order(p, d, q)

min_p = 1; max_p = 3 # min_pは1以上を指定しないとエラー

min_d = 0; max_d = 1

min_q = 0; max_q = 3 



# seasonal_order(sp, sd, sq)

min_sp = 0; max_sp = 1

min_sd = 0; max_sd = 1

min_sq = 0; max_sq = 1



test_pattern = (max_p - min_p +1)*(max_q - min_q + 1)*(max_d - min_d + 1)*(max_sp - min_sp + 1)*(max_sq - min_sq + 1)*(max_sd - min_sd + 1)

print("pattern:", test_pattern)



sfq = 12 # seasonal_order周期パラメータ

ts = num # 時系列データ



test_results = pd.DataFrame(index=range(test_pattern), columns=["model_parameters", "aic"])

num = 0

for p in range(min_p, max_p + 1):

    for d in range(min_d, max_d + 1):

        for q in range(min_q, max_q + 1):

            for sp in range(min_sp, max_sp + 1):

                for sd in range(min_sd, max_sd + 1):

                    for sq in range(min_sq, max_sq + 1):

                        sarima = sm.tsa.SARIMAX(

                            ts, order=(p, d, q), 

                            seasonal_order=(sp, sd, sq, sfq), 

                            enforce_stationarity = False, 

                            enforce_invertibility = False

                        ).fit()

                        test_results.iloc[num]["model_parameters"] = "order=(" + str(p) + ","+ str(d) + ","+ str(q) + "), seasonal_order=("+ str(sp) + ","+ str(sd) + "," + str(sq) + ")"

                        test_results.iloc[num]["aic"] = sarima.aic

                        print(num,'/', test_pattern-1, test_results.iloc[num]["model_parameters"],  test_results.iloc[num]["aic"] )

                        num = num + 1



# 結果（最小AiC）

print("best[aic] parameter ********")

print(test_results[test_results.aic == min(test_results.aic)])
num = target['sold'] # モデル作成用データ（訓練）

exog = target['sell_price']

exog_forecast =df['2016-04-25':]['sell_price']
sarimax_optimization = sm.tsa.SARIMAX(num,

                        exog=exog,

                        order=(3, 1, 3),

                        seasonal_order=(0, 0, 1, 12),

                        enforce_stationarity = False,

                        enforce_invertibility = False

                        ).fit()



sarimax_optimization_resid = sarimax_optimization.resid # 残差成分



fig = plt.figure(figsize=(8, 8))



# 自己相関(ACF)のグラフ

ax1 = fig.add_subplot(211)

sm.graphics.tsa.plot_acf(sarimax_optimization_resid, lags=40, ax=ax1) #ACF計算とグラフ自動作成



# 偏自己相関(PACF)のグラフ

ax2 = fig.add_subplot(212)

sm.graphics.tsa.plot_pacf(sarimax_optimization_resid, lags=40, ax=ax2) #PACF計算とグラフ自動作成



plt.tight_layout() # グラフ間スキマ調整
sarimax_pred = sarimax_optimization.predict(start = '2016-04-25',end = '2016-05-23', exog=exog_forecast, dynamic= True) 
plt.figure(figsize=(20, 4))



plt.plot(num, label="original")

plt.plot(sarimax_pred, c="b", label="model-pred", alpha=0.7)

plt.legend(loc='best')

plt.figure(figsize=(8, 4))



plt.plot(num[1800:], label="actual") # 正解

plt.plot(sarimax_pred, c="b", label="predict", alpha=0.7) # 予測

plt.legend(loc='best')
predict_dy = sarimax_optimization.get_prediction(start = '2016-04-25',end = '2016-05-23', exog=exog_forecast, dynamic= True) 

predict_dy_ci = predict_dy.conf_int() # 信頼区間取得



#　グラフ表示

plt.figure(figsize=(12, 4))

plt.plot(num[1813:], label="actual") # 実データプロット

plt.plot(predict_dy.predicted_mean, c="b", linestyle='--', label="model-pred", alpha=0.7) # 予測プロット



# 予測の95%信頼区間プロット（帯状）

plt.fill_between(predict_dy_ci.index, predict_dy_ci.iloc[:, 0], predict_dy_ci.iloc[:, 1], color='g', alpha=0.2)



plt.legend(loc='upper left')

exog_forecast =df['2016-03-01':]['sell_price']
sarimax_optimization = sm.tsa.SARIMAX(num,

                        exog=exog,

                        order=(3, 1, 3),

                        seasonal_order=(0, 0, 1, 12),

                        enforce_stationarity = False,

                        enforce_invertibility = False

                        ).fit()



sarimax_optimization_resid = sarimax_optimization.resid # 残差成分
sarimax_pred = sarimax_optimization.predict(start = '2016-03-01',end = '2016-04-24', exog=exog_forecast, dynamic= True) 
from sklearn.metrics import mean_squared_error



train_rmse = np.sqrt(mean_squared_error(true, sarimax_pred))



print('RMSE(train): {:.5}'.format(train_rmse))

# exog =