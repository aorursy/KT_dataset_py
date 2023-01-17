# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile
import os
import glob, re
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# ZIPで保存されているファイルを一通り読み込み
path_list = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path_list.append(os.path.join(dirname, filename))
        
path_list.sort()
print(path_list)
air_reserve = pd.read_csv(path_list[0])
air_store_info = pd.read_csv(path_list[1])
air_visit_data = pd.read_csv(path_list[2])
date_info = pd.read_csv(path_list[3])
hpg_reserve = pd.read_csv(path_list[4])
hpg_store_info = pd.read_csv(path_list[5])
sample_submission = pd.read_csv(path_list[6])
store_id_relation = pd.read_csv(path_list[7])
air_reserve
air_store_info
air_visit_data
date_info
hpg_reserve
hpg_store_info
sample_submission
store_id_relation
# sample_submittionの中身を確認
# air_ストアID_日付 の情報から、来客数を予測して欲しい
# 頭から20件
print(sample_submission.head(20))

# (行数、列数)
sample_submission.shape
# 元々のものは残しておく
test_data = sample_submission

# 元々のIDからstore_idとvisit_dateを切り出す
test_data['store_id'] = test_data['id'].str[:20]
test_data['visit_date'] = test_data['id'].str[21:]
# 現時点では visitors に意味がないので切り出し
test_data.drop(['visitors'], axis=1, inplace=True)
# 日付の型をobjectからdatetimeに変換
test_data['visit_date'] = pd.to_datetime(test_data['visit_date'])
# 念のためカラム情報の確認
test_data.info()
# ヘッダー情報もみておく
test_data.head()
# air_visit_data の前処理
# Airレジの各レストランの日付と実客数のデータ
# -> 予測すべきレストランの過去の客数データ  (シミュレータの過去のvimpsと似てる？)

# visit_dateを日付方にする
air_visit_data['visit_date'] = pd.to_datetime(air_visit_data['visit_date'])
air_visit_data.head()
# (行数、列数)
air_visit_data.info()
# あるレストラン（air_00a91d42b08b08d9）の実客数を確認
## 条件に合うものを選択
store_sample = air_visit_data[air_visit_data['air_store_id'] == 'air_00a91d42b08b08d9']

## 基本統計量の確認
## 平均26.08人、最大99人などの情報
store_sample.describe()
# visit_dateの確認
# 被りなしそう（解説ではfirst, lastもあるがこちらでは表示されていない） -> date型にしていないだけでした
# 2016年7月1日〜2017年4月22日のデータを元に、2017年4月最終週〜2017年5月末日までの各日の来店数を予測
store_sample.visit_date.describe()
# 狙いを定める
target_df = air_visit_data
target_df.info()
date_info['calendar_date'] = pd.to_datetime(date_info['calendar_date'])
date_info.info()
target_df =  pd.merge(target_df, date_info, how='left', left_on='visit_date', right_on='calendar_date').drop(columns='calendar_date')
# 日付から曜日を算出 (0-6, 0:mondayとして数値に変換される)
# one_hot = pd.get_dummies(target_df['day_of_week'])
# target_df = target_df.join(one_hot).drop(columns='day_of_week')
target_df['visit_date'] = pd.to_datetime(target_df['visit_date'])
target_df
target_df
import pandas.tseries.offsets as offsets

# 翌日が休日かフラグと、土日含めて休みかフラグに変更
# 激烈に遅いのでなんとかできそう
for index, row in target_df.iterrows():
    tommorow = row['visit_date'] + offsets.Day()
    # 土日はそもそも休日
    if row['day_of_week'] == 'Saturday' or row['day_of_week'] == 'Sunday':
        target_df.loc[index, 'holiday_flg'] = 0
    if index + 1 < len(target_df):
        row_next = target_df.loc[index + 1]
        if row_next['air_store_id'] == row['air_store_id'] and (row_next['holiday_flg'] == 1 or row['day_of_week'] == 'Saturday'):
            target_df.loc[index, 'tomorrow_holiday_flg'] = 1
            continue
    target_df.loc[index, 'tomorrow_holiday_flg'] = 0
target_df.head(30)
test_data.info()
test_data =  pd.merge(test_data, date_info, how='left', left_on='visit_date', right_on='calendar_date').drop(columns='calendar_date')
# 日付から曜日を算出 (0-6, 0:mondayとして数値に変換される)
# one_hot = pd.get_dummies(test_data['day_of_week'])
# test_data = test_data.join(one_hot).drop(columns='day_of_week')
test_data['visit_date'] = pd.to_datetime(test_data['visit_date'])
test_data
# 翌日が休日かフラグと、土日含めて休みかフラグに変更
# 激烈に遅いのでなんとかできそう
for index, row in test_data.iterrows():
    tommorow = row['visit_date'] + offsets.Day()
    # 土日だったらフラグを消す
    if row['day_of_week'] == 'Saturday' or row['day_of_week'] == 'Sunday':
        test_data.loc[index, 'holiday_flg'] = 0
    if index + 1 < len(test_data):
        row_next = test_data.loc[index + 1]
        if row_next['store_id'] == row['store_id'] and (row_next['holiday_flg'] == 1 or row['day_of_week'] == 'Saturday'):
            test_data.loc[index, 'tomorrow_holiday_flg'] = 1
            continue
    test_data.loc[index, 'tomorrow_holiday_flg'] = 0
test_data
train_data = target_df.copy()
train_data
# 一旦保存
train_data.to_csv('/kaggle/working/train_data.csv')
test_data.to_csv('/kaggle/working/test_data.csv')

# 読み直し
train_data = pd.read_csv('/kaggle/working/train_data.csv')
test_data = pd.read_csv('/kaggle/working/test_data.csv')
train_data['visit_date'] = pd.to_datetime(train_data['visit_date'])
test_data['visit_date'] = pd.to_datetime(test_data['visit_date'])
mean_data = train_data.groupby(['air_store_id', 'day_of_week', 'holiday_flg', 'tomorrow_holiday_flg']).agg({'visitors':'median'}).reset_index()
mean_data.columns = ['air_store_id', 'day_of_week', 'holiday_flg', 'tomorrow_holiday_flg','median_visitors_all']
mean_data

# # train_data(学習用データ)からair_store_idとdowをグルーピングしてvisitorsの中央値（median）を算出
# agg_data = train_data.groupby(['air_store_id', 'dow']).agg({'visitors':'median'}).reset_index()

# # agg_dataのカラム名をつける
# agg_data.columns = ['air_store_id', 'dow', 'visitors']
# agg_data['visitors']= agg_data['visitors']
 
# # agg_dataを確認
# agg_data.head(12)

print(date_info)

# data_infoの祝日フラグが1（オン）のデータを確認
date_info[date_info['holiday_flg'] == 1].head(10)

# 土日でも祝日であれば祝日フラグが立っているが、土日は基本的に休日なので祝日にする必要はない
# 土日の場合はフラグを0にするという処理を行う (土日かつ祝日の日付を取得し、フラグを0にする)
weekend_hdays = date_info.apply((lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1), axis=1)
date_info.loc[weekend_hdays, 'holiday_flg'] = 0
# 新しい日付により大きい重みを与える
# date_info.indexの値が小さい＝より昔のデータ
date_info['weight'] = (date_info.index + 1) / len(date_info) 
 
#ヘッダーとテイルの情報を出して確認
print(date_info.head())
date_info.tail()
# air_visit_dataと重みを追加したdate_infoをマージさせてvisit_dataを作成
# visit_dataから不必要なcalendar_dateを落とす
visit_data = train_data.merge(date_info, left_on=['visit_date'], right_on='calendar_date', how='left', suffixes=['','_y'])
visit_data.drop('calendar_date', axis=1, inplace=True)
visit_data.drop('day_of_week_y', axis=1, inplace=True)
visit_data.drop('holiday_flg_y', axis=1, inplace=True)
# visit_dataの実客数にnp.log1pの対数関数を使って処理
# なぜ？
visit_data['visitors'] = visit_data.visitors.map(pd.np.log1p)
# visit_dataの確認
visit_data
# wmean（重み付き平均）の式を格納
wmean = lambda x:( (x.weight * x.visitors).sum() / x.weight.sum() )
# グルーピングして重み付き平均を算出
visitors = visit_data.groupby(
['air_store_id', 'day_of_week', 'holiday_flg', 'tomorrow_holiday_flg']).apply(wmean).reset_index()
visitors.rename(columns={0:'weighted_visitors_all'}, inplace=True) 
# データを確認
visitors

# air_00a91d42b08b08d9 Monday
# 祝日フラグあり、なしで重みが違うため、微妙に違う計算結果になっている
agg_data = pd.merge(mean_data, visitors, on=['air_store_id', 'day_of_week', 'holiday_flg', 'tomorrow_holiday_flg'])
agg_data['median_visitors_all'] = agg_data.median_visitors_all.map(pd.np.log1p)
agg_data
train_data
agg_data
merged_df = pd.merge(train_data, agg_data, on=['air_store_id','day_of_week','holiday_flg','tomorrow_holiday_flg'], how='left')
merged_df = merged_df.drop('Unnamed: 0', axis=1)
merged_df
test_data
test_data = test_data.drop('Unnamed: 0', axis=1)

test_data.rename(columns={'store_id':'air_store_id'}, inplace=True)
merged_test_data = pd.merge(test_data, agg_data, on=['air_store_id','day_of_week','holiday_flg','tomorrow_holiday_flg'], how='left').drop('id',axis=1)
merged_test_data
train = merged_df.copy()
train
test = merged_test_data.copy()
test

# 日付から曜日を算出 (0-6, 0:mondayとして数値に変換される)
one_hot = pd.get_dummies(train['day_of_week'])
train = train.join(one_hot)
train = train.drop('day_of_week',axis=1)
train
one_hot = pd.get_dummies(test['day_of_week'])
test = test.join(one_hot)
test = test.drop('day_of_week',axis=1)
train.describe()
test.describe()
# 欠損値を平均値で埋める
test['median_visitors_all'] = test['median_visitors_all'].fillna(2.772849)
test['weighted_visitors_all'] = test['weighted_visitors_all'].fillna(2.727645)
test.describe()
train
test
train['date_year'] = train['visit_date'].dt.year
train['date_month'] = train['visit_date'].dt.month
train['date_day'] = train['visit_date'].dt.day
train
test['date_year'] = test['visit_date'].dt.year
test['date_month'] = test['visit_date'].dt.month
test['date_day'] = test['visit_date'].dt.day
test
train['date_year'] = train['date_year'] - 2016.5
train
test['date_year'] = test['date_year'] - 2016.5
test
train['date_month'] = train.date_month.map(pd.np.log1p)
train['date_day'] = train.date_day.map(pd.np.log1p)
train['visitors'] = train.visitors.map(pd.np.log1p)
test['date_month'] = test.date_month.map(pd.np.log1p)
test['date_day'] = test.date_day.map(pd.np.log1p)
train
test
train_df = train[train['visit_date'] <= '2017-01-28'].reset_index().drop('index',axis=1).drop('visit_date',axis=1)
valid = train[train['visit_date'] > '2017-01-28'].reset_index().drop('index',axis=1).drop('visit_date',axis=1)
print(train_df)
train_df
train_df_y = train_df['visitors']
valid_y = valid['visitors']
train_df_y
train_df_X = train_df.copy().drop(['visitors','air_store_id'], axis=1)
valid_X = valid.copy().drop(['visitors','air_store_id'], axis=1)
test_df = test.copy().drop(['air_store_id', 'visit_date'], axis=1)
print(test_df)
valid_X

import lightgbm as lgb

lgb_train = lgb.Dataset(train_df_X, train_df_y)
lgb_eval = lgb.Dataset(valid_X, valid_y)
params = {'metric': 'rmse','max_depth' : -1}
gbm = lgb.train(params,
               lgb_train,
               valid_sets=(lgb_train, lgb_eval),
               num_boost_round=10000,
               early_stopping_rounds=100,
               verbose_eval=50)
lgb.plot_importance(gbm, height=0.5, figsize=(8,16))

valid_y_pred = gbm.predict(valid_X)

y_pred = gbm.predict(test_df)
y_pred
len(y_pred)
submission = sample_submission.drop(['store_id','visit_date'],axis=1)
submission = pd.concat([submission, pd.Series(y_pred)],axis=1)
submission = submission.rename(columns={0:'visitors'})
submission['visitors'] = submission.visitors.map(pd.np.expm1)
submission
 
# # 提出フォーマットの規定に合うように処理してsub_fileへ格納
# submission = submission[['id', 'visitors']]
# final['visitors'][final['visitors'] ==0] = submission['visitors'][final['visitors'] ==0]
# sub_file = final.copy()
 
# # データの確認
# sub_file.head()
submission.to_csv('/kaggle/working/submission.csv', index=False)
import xgboost as xgb

fit_params = {
    'eval_metric': 'rmse',
    'eval_set': [[train_df_X,train_df_y]]
    }
 
#グリッドサーチの範囲
params = {
    'learning_rate': list(np.arange(0.05, 0.41, 0.05)),
    'max_depth': list(np.arange(3, 11, 1))
}

from sklearn.model_selection import GridSearchCV

def GSfit(params):
    regressor = xgb.XGBRegressor(n_estimators=100)
    grid = GridSearchCV(regressor, params, cv=3, scoring='neg_mean_squared_error',verbose=2)
    grid.fit(train_df_X,train_df_y)
    return grid
grid = GSfit(params)
grid_best_params = grid.best_params_
grid_scores_df = pd.DataFrame(grid.cv_results_)
#関数の処理で必要なライブラリ
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#データ可視化ライブラリ
import matplotlib.pyplot as plt
import seaborn as sns

#予測値と正解値を描写する関数
def True_Pred_map(pred_df):
    RMSE = np.sqrt(mean_squared_error(pred_df['true'], pred_df['pred']))
    R2 = r2_score(pred_df['true'], pred_df['pred']) 
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111)
    ax.scatter('true', 'pred', data=pred_df)
    ax.set_xlabel('True Value', fontsize=15)
    ax.set_ylabel('Pred Value', fontsize=15)
    ax.set_xlim(pred_df.min().min()-0.1 , pred_df.max().max()+0.1)
    ax.set_ylim(pred_df.min().min()-0.1 , pred_df.max().max()+0.1)
    x = np.linspace(pred_df.min().min()-0.1, pred_df.max().max()+0.1, 2)
    y = x
    ax.plot(x,y,'r-')
    plt.text(0.1, 0.9, 'RMSE = {}'.format(str(round(RMSE, 5))), transform=ax.transAxes, fontsize=15)
    plt.text(0.1, 0.8, 'R^2 = {}'.format(str(round(R2, 5))), transform=ax.transAxes, fontsize=15)
pred_df = pd.concat([valid_y.reset_index(drop=True), pd.Series(valid_y_pred)], axis=1)
pred_df.columns = ['true', 'pred']

True_Pred_map(pred_df)
print(grid_best_params)
model = grid.best_estimator_ 
valid_y_pred = model.predict(valid_X)
y_pred = model.predict(test_df)

pred_df = pd.concat([valid_y.reset_index(drop=True), pd.Series(valid_y_pred)], axis=1)
pred_df.columns = ['true', 'pred']

True_Pred_map(pred_df)
xgb_submission = sample_submission.drop(['store_id','visit_date'],axis=1)
xgb_submission = pd.concat([xgb_submission, pd.Series(y_pred)],axis=1)
xgb_submission = xgb_submission.rename(columns={0:'visitors'})
xgb_submission['visitors'] = xgb_submission.visitors.map(pd.np.expm1)
xgb_submission
 
# # 提出フォーマットの規定に合うように処理してsub_fileへ格納
# submission = submission[['id', 'visitors']]
# final['visitors'][final['visitors'] ==0] = submission['visitors'][final['visitors'] ==0]
# sub_file = final.copy()
 
# # データの確認
# sub_file.head()
submission
submission_merge = submission.copy()
submission_merge = pd.merge(submission_merge, xgb_submission,on='id',suffixes=['','_y'])
submission_merge['mean_visitors'] = round((submission_merge['visitors'] + submission_merge['visitors_y']) / 2)
submission_merge = submission_merge.drop(['visitors', 'visitors_y'], axis=1)
submission_merge = submission_merge.rename(columns={'mean_visitors':'visitors'})
submission_merge
submission_merge.to_csv('/kaggle/working/submission_merge.csv', index=False)


# sample_submissionのIDをレストランIDや日付に分ける
sample_submission['air_store_id'] = sample_submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))
sample_submission['calendar_date'] = sample_submission.id.map(lambda x: x.split('_')[2])
 
# 重み付き平均で予測したvisitorsとsample_submissionをマージする
# 祝日データを同時にマージし、祝日フラグで紐付ける
sample_submission.drop('visitors', axis=1, inplace=True)
sample_submission = sample_submission.merge(date_info, on='calendar_date', how='left')
sample_submission = sample_submission.merge(
    visitors, on=['air_store_id', 'day_of_week', 'holiday_flg'], how='left')
 
# データセットを確認
sample_submission.head()
# 「air_store_id」と「day_of_week」のみで欠損データに重み平均を入れる
missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[visitors.holiday_flg==0], on=('air_store_id', 'day_of_week'), how='left')['visitors_y'].values
 
# 改めて欠損データの確認
missing_values_table(sample_submission)

# まだ余っているので、air_store_idだけで紐付ける
# 「air_store_id」のみの重み付き平均を計算して欠損データへ入れる
missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[['air_store_id', 'visitors']].groupby('air_store_id').mean().reset_index(), 
    on='air_store_id', how='left')['visitors_y'].values
 
# 欠損データを確認
missing_values_table(sample_submission)

# 全部埋まっている
# visitorsをnp.expm1で処理して実客数へ戻す
sample_submission['visitors'] = sample_submission.visitors.map(pd.np.expm1)
 
# 提出フォーマットの規定に合うように処理してsub_fileへ格納
sample_submission = sample_submission[['id', 'visitors']]
final['visitors'][final['visitors'] ==0] = sample_submission['visitors'][final['visitors'] ==0]
sub_file = final.copy()
 
# データの確認
sub_file.head()






# train(学習用データ)からair_store_idとdowをグルーピングしてvisitorsの中央値（median）を算出
agg_data = train.groupby(['air_store_id', 'dow']).agg({'visitors':'median'}).reset_index()

# agg_dataのカラム名をつける
agg_data.columns = ['air_store_id', 'dow', 'visitors']
agg_data['visitors']= agg_data['visitors']
 
# agg_dataを確認
agg_data.head(12)
# testデータのair_store_idを保存し、後でtrainに適応させる
store_id = []

for i in test_data['store_id'].unique():
  store_id += [i]

print(store_id)
# IDを処理する(test)
for index, name in enumerate(store_id):
  test_data.replace(name, index, inplace = True)
test_data['date_year'] = test_data['visit_date'].dt.year
test_data['date_month'] = test_data['visit_date'].dt.month
test_data['date_day'] = test_data['visit_date'].dt.day
test_data = test_data.drop('visit_date',axis=1)
test_data
train_data
train_data = pd.merge(train_data, pd.DataFrame(store_id, columns=["air_store_id"]), 
                                    how='inner', on='air_store_id')
train_data
# IDを処理する(train)
for index, name in enumerate(store_id):
  train_data.replace(name, index, inplace = True)
train_data['visit_date'] = pd.to_datetime(train_data['visit_date'])
train_data
train_data['date_year'] = train_data['visit_date'].dt.year
train_data['date_month'] = train_data['visit_date'].dt.month
train_data['date_day'] = train_data['visit_date'].dt.day
train_data = train_data.drop('visit_date',axis=1)
train_data
test_data = test_data.drop(columns='id')
# 型を直す
train_data = train_data.astype('int64')
test_data = test_data.astype('int64')
train_data
test_data
train_data.to_csv('train_data.csv')
test_data.to_csv('test_data.csv')
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')
# 目的変数
y = train_df['visitors']
# 説明変数
X = train_df.drop(columns=['visitors', 'Unnamed: 0'])

# いらない列を消す
test_df = test_df.drop(columns=['Unnamed: 0'])
test_df = test_df.rename(columns={'store_id': 'air_store_id'})
test_df
X
import xgboost

estimator = xgboost.XGBRegressor()
estimator.fit(X,y)
y_test = estimator.predict(test_df)
print(y_test)
sample_submission = pd.read_csv(path_list[6])
sample_submission
submission = sample_submission.copy()
submission = submission.drop('visitors', axis=1)
submission
submission["visitors"] = y_test
submission
submission.to_csv('submission.csv', index=None)
# 一旦提出データを作成
# 各曜日の来客数の中央値を予測値として提出する

# test_dfとagg_dataのstoreid_id、dowをすり合わせmergeさせる
merged = pd.merge(submission_data, agg_data, how='left', left_on=['store_id', 'dow'], right_on=['air_store_id', 'dow'])
 
# idとvisitorsだけをfinalへ格納
final = merged[['id', 'visitors']]
 
# finalのヘッダー情報
final.head()
# NaNを探してテーブルにする関数
def missing_values_table(df): 
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum()/len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        return mis_val_table_ren_columns 
 
# finalのNaNを確認 (3.5%程度が欠損値)
missing_values_table(final)
# とりあえず欠損値には0を入れる
final.fillna(0, inplace=True)

# 念のため確認
missing_values_table(final)
path_list = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path_list.append(os.path.join(dirname, filename))
        
path_list.sort()
print(path_list)
air_reserve = pd.read_csv(path_list[0])
air_store_info = pd.read_csv(path_list[1])
air_visit_data = pd.read_csv(path_list[2])
date_info = pd.read_csv(path_list[3])
hpg_reserve = pd.read_csv(path_list[4])
hpg_store_info = pd.read_csv(path_list[5])
sample_submission = pd.read_csv(path_list[6])
store_id_relation = pd.read_csv(path_list[7])

print(date_info)
print(date_info)

# data_infoの祝日フラグが1（オン）のデータを確認
date_info[date_info['holiday_flg'] == 1].head(10)

# 土日でも祝日であれば祝日フラグが立っているが、土日は基本的に休日なので祝日にする必要はない
# 土日の場合はフラグを0にするという処理を行う (土日かつ祝日の日付を取得し、フラグを0にする)
weekend_hdays = date_info.apply((lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1), axis=1)
date_info.loc[weekend_hdays, 'holiday_flg'] = 0
# 新しい日付により大きい重みを与える
# date_info.indexの値が小さい＝より昔のデータ
date_info['weight'] = (date_info.index + 1) / len(date_info) 
 
#ヘッダーとテイルの情報を出して確認
print(date_info.head())
date_info.tail()
# air_visit_dataと重みを追加したdate_infoをマージさせてvisit_dataを作成
# visit_dataから不必要なcalendar_dateを落とす
visit_data = air_visit_data.merge(date_info, left_on='visit_date', right_on='calendar_date', how='left')
visit_data.drop('calendar_date', axis=1, inplace=True)
# visit_dataの実客数にnp.log1pの対数関数を使って処理
# なぜ？
visit_data['visitors'] = visit_data.visitors.map(pd.np.log1p)
# visit_dataの確認
visit_data.head(10)
# wmean（重み付き平均）の式を格納
wmean = lambda x:( (x.weight * x.visitors).sum() / x.weight.sum() )
# グルーピングして重み付き平均を算出
visitors = visit_data.groupby(
['air_store_id', 'day_of_week', 'holiday_flg']).apply(wmean).reset_index()
visitors.rename(columns={0:'visitors'}, inplace=True) 
# データを確認
visitors.head(10)

# air_00a91d42b08b08d9 Monday
# 祝日フラグあり、なしで重みが違うため、微妙に違う計算結果になっている
# sample_submissionのIDをレストランIDや日付に分ける
sample_submission['air_store_id'] = sample_submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))
sample_submission['calendar_date'] = sample_submission.id.map(lambda x: x.split('_')[2])
 
# 重み付き平均で予測したvisitorsとsample_submissionをマージする
# 祝日データを同時にマージし、祝日フラグで紐付ける
sample_submission.drop('visitors', axis=1, inplace=True)
sample_submission = sample_submission.merge(date_info, on='calendar_date', how='left')
sample_submission = sample_submission.merge(
    visitors, on=['air_store_id', 'day_of_week', 'holiday_flg'], how='left')
 
# データセットを確認
sample_submission.head()
# sample_submissionの欠損データを確認
missing_values_table(sample_submission)

# 祝日フラグで紐づかなかったものが余っている
# 「air_store_id」と「day_of_week」のみで欠損データに重み平均を入れる
missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[visitors.holiday_flg==0], on=('air_store_id', 'day_of_week'), how='left')['visitors_y'].values
 
# 改めて欠損データの確認
missing_values_table(sample_submission)

# まだ余っているので、air_store_idだけで紐付ける
# 「air_store_id」のみの重み付き平均を計算して欠損データへ入れる
missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[['air_store_id', 'visitors']].groupby('air_store_id').mean().reset_index(), 
    on='air_store_id', how='left')['visitors_y'].values
 
# 欠損データを確認
missing_values_table(sample_submission)

# 全部埋まっている
# 内容を確認
sample_submission
# visitorsをnp.expm1で処理して実客数へ戻す
sample_submission['visitors'] = sample_submission.visitors.map(pd.np.expm1)
 
# 提出フォーマットの規定に合うように処理してsub_fileへ格納
sample_submission = sample_submission[['id', 'visitors']]
final['visitors'][final['visitors'] ==0] = sample_submission['visitors'][final['visitors'] ==0]
sub_file = final.copy()
 
# データの確認
sub_file.head()
# 算術平均をnp.meanで算出
sub_file['visitors'] = np.mean([final['visitors'], sample_submission['visitors']], axis = 0)
sub_file.to_csv('sub_math_mean_1.csv', index=False)
 
# 相乗平均を算出
sub_file['visitors'] = (final['visitors'] * sample_submission['visitors']) ** (1/2)
sub_file.to_csv('sub_geo_mean_1.csv', index=False)
 
# 調和平均を算出
sub_file['visitors'] = 2/(1/final['visitors'] + 1/sample_submission['visitors'])
sub_file.to_csv('sub_hrm_mean_1.csv', index=False)

