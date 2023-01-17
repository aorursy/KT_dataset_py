# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.float_format',lambda x: '%.2f' %x)
pd.set_option('display.max_columns',200)
pd.set_option('display.max_rows',1000)
import sklearn

import scipy.sparse 
import matplotlib.pyplot as plt
%matplotlib inline 
import lightgbm as lgb
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import catboost
from catboost import Pool
from catboost import CatBoostRegressor

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
base_path = "/kaggle/input/competitive-data-science-predict-future-sales/"
shops = pd.read_csv(base_path+"shops.csv");
test = pd.read_csv(base_path+"test.csv");
sample_submission = pd.read_csv(base_path+"sample_submission.csv")
sales_train = pd.read_csv(base_path+"sales_train.csv")
categories = pd.read_csv(base_path+"item_categories.csv")
items = pd.read_csv(base_path+"items.csv")

base_path_trans = "/kaggle/input/predict-future-sales-translated-dataset/"
shops_en = pd.read_csv(base_path_trans+"shops_en.csv");
categories_en = pd.read_csv(base_path_trans+"item_categories_en.csv")
items_en = pd.read_csv(base_path_trans+"items_en.csv")

# Print column names and columns type
print("** Items **\n",items.dtypes)
print("\n** Categories **\n",categories.dtypes)
print("\n** Shops **\n",shops.dtypes)
print("\n** Sales **\n",sales_train.dtypes)
# Join data frames
sales_train = sales_train.join(items, on='item_id', rsuffix='_').join(categories, on='item_category_id', rsuffix='_').drop(['item_id_', 'item_category_id_'], axis=1)

# Describe dataframe
sales_train.describe().T
sales_train.head(3).T
categories_en
shops_en.sort_values('shop_name')
test[test["shop_id"].isin([0,57,1,58,10,11,39,40])].groupby(["shop_id"])["item_id"].agg("count")

print("Missing values\n", sales_train.isnull().sum()) 
print("\nInvalid Values")
print("item_price: ",sales_train[sales_train.item_price<0].item_price.size)
# Remove data with negative price
sales_train.drop(sales_train[sales_train['item_price'] < 0].index, inplace = True)

# Convert date column format to format that may be better for analysis splitting in diferent features
sales_train['date'] = pd.to_datetime(sales_train['date'], format="%d.%m.%Y")
# Split date
sales_train['year'] = pd.DatetimeIndex(sales_train['date']).year
sales_train['month'] = pd.DatetimeIndex(sales_train['date']).month
sales_train['day'] = pd.DatetimeIndex(sales_train['date']).day

# Combining data
sales_train.loc[sales_train['shop_id']==11,'shop_id'] = 10
sales_train.loc[sales_train['shop_id']==0,'shop_id'] = 57
sales_train.loc[sales_train['shop_id']==1,'shop_id'] = 58
sales_train.loc[sales_train['shop_id']==40,'shop_id'] = 39

sales_train.describe().T
plt.hist(sales_train["item_id"],bins=50)
plt.xlabel("item_id")
plt.ylabel("count")
plt.show()
plt.hist(sales_train["shop_id"], bins=60)
plt.xlabel("shop_id")
plt.ylabel("count")
plt.show()

plt.hist(sales_train["item_category_id"], bins=84)
plt.xlabel("item_category_id")
plt.ylabel("count")
plt.figure(figsize=(3,4))
plt.show()
shop_per_cat = sales_train.groupby(['item_category_id'])['item_cnt_day'].agg("sum")
shop_per_cat.columns=["item_category_id","cnt_per_cate"]
shop_per_cat.plot()
plt.title("Shops per category")
plt.show()
# Daily Shops
daily_shops= sales_train.groupby(['date'])['item_cnt_day'].agg("sum")
daily_shops.columns=["date","cnt_day"]
daily_shops.plot()
plt.title("Shops per day")
plt.ylabel("sum_cnt")
plt.show()
# Monthly shops
monthly_shops= sales_train.groupby(['month'])['item_cnt_day'].agg("sum")
monthly_shops.columns=["date","cnt_day"]
monthly_shops.plot()
plt.title("Shops per month")
plt.show()
# Shops per block num
shops_per_block = sales_train.groupby(['date_block_num'], as_index=False)['item_cnt_day'].agg("sum")
#shops_per_block.columns=["date_block_num","item_cnt"]
shops_per_block['item_cnt_day'].plot()
plt.xticks (shops_per_block['date_block_num'])
plt.title("Sales per block num")
plt.show()
# Sales per week day
sales_train["week_day"] = pd.DatetimeIndex(sales_train['date']).dayofweek
sales_per_wd = sales_train.groupby(['week_day'])['item_cnt_day'].agg("sum")
sales_per_wd.plot()
plt.show()
def days_of_month_rate(year,month):
    import calendar
    import datetime
    days = calendar.monthrange(year,month)[1]
    date =  datetime.date(year, month, 1)
    days_of_month = 0   
   
    for x in range(1,8):
        if(date.weekday()<4):
            days_of_month = days_of_month + 1
        else:
            days_of_month = days_of_month + 1
        date = date + datetime.timedelta(1)
    
    days_of_month = days_of_month * 4    
    if(days<=29):
        return days_of_month;
    date =  datetime.date(year, month, 29)
    
    for x in range(29, days+1):        
        if(date.weekday()<4):
            days_of_month = days_of_month + 1
        else:
            days_of_month = days_of_month + 1 
        date = date + datetime.timedelta(1)
    return days_of_month

    return days_of_month

plt.scatter(sales_train['item_price'], sales_train['item_cnt_day'])
plt.ylabel("item_cnt_day")
plt.xlabel("item_price")
plt.show()
sales_train = sales_train[(sales_train['item_price']<40000)&(sales_train['item_cnt_day']<=100)]
sales_train.loc[sales_train['item_cnt_day']<0, 'item_cnt_day'] = 0
plt.scatter(sales_train['item_price'], sales_train['item_cnt_day'])
plt.ylabel("item_cnt_day")
plt.xlabel("item_price")
plt.show()
test_shop_ids = test['shop_id'].unique()
test_item_ids = test['item_id'].unique()

# Only shops and items that exist in test set.
work_data = sales_train[sales_train['shop_id'].isin(test_shop_ids)]
work_data = work_data[work_data['item_id'].isin(test_item_ids)]

print('Data set size before :', sales_train.shape[0])
print('Data set size after:', work_data.shape[0])
work_data.describe().T
# Group data and generate feautures from price and sales per day
train_monthly = work_data.groupby(
    ["date_block_num","shop_id","item_id","item_category_id","year","month"], as_index=False).agg({
    'item_price':['sum', 'mean'], 'item_cnt_day':['sum', 'mean','count']#, "week_day" : ['count', 'mean']
});

# Rename features.
train_monthly.columns = ['date_block_num', 'shop_id', 'item_id','item_category_id', 'year', 'month','sum_item_price', 'mean_item_price', 'item_cnt_month','mean_item_cnt', 'sales']


# Add custom feature: month_rate
train_monthly['month_rate'] = train_monthly.apply(lambda x: days_of_month_rate(x['year'].astype(int),x['month'].astype(int)),axis=1)


train_monthly.describe().T
# Build a empty data set with possible combinations for date_block_num, shop_id and tem_id
shop_ids = train_monthly['shop_id'].unique()
item_ids = train_monthly['item_id'].unique()
empty_df = []
for i in range(34):
    for shop in shop_ids:
        for item in item_ids:
            empty_df.append([i, shop, item])
    
empty_df = pd.DataFrame(empty_df, columns=['date_block_num','shop_id','item_id'])
train_monthly = pd.merge(empty_df, train_monthly, on=['date_block_num','shop_id','item_id'], how='left')
train_monthly.fillna(0, inplace=True)
train_monthly.describe().T
train_monthly['item_cnt_next_month'] = train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(-1)
# Features based on item prices
#hist_item_price_ft = work_data.groupby(['item_id'], as_index=False).agg({'item_price':['min', 'max']})
#hist_item_price_ft.columns = ['item_id',"hist_min_price","hist_max_price"]
#hist_item_price_ft
# Aggregation functions
getMin = lambda x: x.rolling(window=3, min_periods=1).min()
getMax = lambda x: x.rolling(window=3, min_periods=1).max()
getMean = lambda x: x.rolling(window=3, min_periods=1).mean()
getStd = lambda x: x.rolling(window=3, min_periods=1).std()

fn_lst = [getMin, getMax, getMean, getStd]
fn_name = ['min', 'max', 'mean', 'std']

for i in range(len(fn_lst)):
    train_monthly[('item_cnt_hist_%s' % fn_name[i])] = train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_category_id', 'item_id'])['item_cnt_month'].apply(fn_lst[i])

# Fill the empty std features with 0
train_monthly['item_cnt_hist_std'].fillna(0, inplace=True)
train_monthly.describe().T
train_monthly["item_cnt_shifted_1"] = train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_category_id', 'item_id'])['item_cnt_month'].shift(1)
train_monthly["item_cnt_shifted_1"].fillna(0, inplace=True)
train_monthly["item_cnt_shifted_2"] = train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_category_id', 'item_id'])['item_cnt_month'].shift(2)
train_monthly["item_cnt_shifted_2"].fillna(0, inplace=True)
train_monthly["item_cnt_shifted_3"] = train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_category_id', 'item_id'])['item_cnt_month'].shift(3)
train_monthly["item_cnt_shifted_3"].fillna(0, inplace=True)
train_monthly['item_trend'] = train_monthly['item_cnt_month']
train_monthly['item_trend'] -= (train_monthly["item_cnt_shifted_1"]+train_monthly["item_cnt_shifted_2"]+train_monthly["item_cnt_shifted_3"])

train_monthly['item_trend'] /= 4

train_monthly['date_block_num'].values.max()
# Take months from 3 to 28. 3 first months where take for aggregation functions
train_data = train_monthly[(train_monthly['date_block_num']<=28) & (train_monthly['date_block_num']>=3)]
# Take months from 29 to 32
val_data = train_monthly[(train_monthly['date_block_num']<=32) & (train_monthly['date_block_num']>=29)]
# Last block is for test
test_data = train_monthly[train_monthly['date_block_num']==33]
TotalData = train_monthly.shape[0]
print('number of validation samples: %d (%f%%)'%(val_data.shape[0],(val_data.shape[0]/TotalData*100)))
print("number of training samples: %d (%f%%)"%(train_data.shape[0],(train_data.shape[0]/TotalData*100)))
print("number of test samples: %d (%f%%)"%(test_data.shape[0],(test_data.shape[0]/TotalData*100)))

# Item mean
hist_item_mean = train_data.groupby(['item_id']).agg({'item_cnt_next_month': ['mean']})
hist_item_mean.columns = ['hist_item_mean']
hist_item_mean.reset_index(inplace=True)
# Shop mean
hist_shop_mean = train_data.groupby(['shop_id']).agg({'item_cnt_next_month': ['mean']})
hist_shop_mean.columns = ['hist_shop_mean']
hist_shop_mean.reset_index(inplace=True)

#Item-shop mean
hist_item_shop_mean = train_data.groupby(['shop_id', 'item_id']).agg({'item_cnt_next_month': ['mean']})
hist_item_shop_mean.columns = ['hist_shop_item_mean']
hist_item_shop_mean.reset_index(inplace=True)
# Year mean
hist_year_mean = train_data.groupby(['year']).agg({'item_cnt_next_month': ['mean']})
hist_year_mean.columns = ['hist_year_mean']
hist_year_mean.reset_index(inplace=True)
# Month mean
hist_month_mean = train_data.groupby(['month']).agg({'item_cnt_next_month': ['mean']})
hist_month_mean.columns = ['hist_month_mean']
hist_month_mean.reset_index(inplace=True)

# Add mean encoding features to train set.
train_data = pd.merge(train_data, hist_item_mean, on=['item_id'], how='left')
train_data = pd.merge(train_data, hist_shop_mean, on=['shop_id'], how='left')
train_data = pd.merge(train_data, hist_item_shop_mean, on=['shop_id', 'item_id'], how='left')
train_data = pd.merge(train_data, hist_year_mean, on=['year'], how='left')
train_data = pd.merge(train_data, hist_month_mean, on=['month'], how='left')
# Add meand encoding features to validation set.
val_data = pd.merge(val_data, hist_item_mean, on=['item_id'], how='left')
val_data = pd.merge(val_data, hist_shop_mean, on=['shop_id'], how='left')
val_data = pd.merge(val_data, hist_item_shop_mean, on=['shop_id', 'item_id'], how='left')
val_data = pd.merge(val_data, hist_year_mean, on=['year'], how='left')
val_data = pd.merge(val_data, hist_month_mean, on=['month'], how='left')
train_data.describe().T
#train_data.columns
target_feature = ['item_cnt_next_month']
all_features = ['date_block_num', 'shop_id','item_id','item_category_id','year','month',
                'sum_item_price','mean_item_price','item_cnt_month','mean_item_cnt',
                'sales','month_rate','item_cnt_hist_min','item_cnt_hist_max',
                'item_cnt_hist_mean','item_cnt_hist_std','hist_item_mean','hist_shop_mean',
                'hist_shop_item_mean','hist_year_mean','hist_month_mean',
               'item_cnt_shifted_1','item_cnt_shifted_2','item_cnt_shifted_3','item_trend']

x_train = train_data[all_features]
y_train = train_data[target_feature]

x_val = val_data[all_features]
y_val = val_data[target_feature]
test = pd.read_csv(base_path+"test.csv");
latest_records = pd.concat([train_data, val_data]).drop_duplicates(subset=['shop_id', 'item_id'], keep='last')
x_test = pd.merge(test, latest_records, on=['shop_id', 'item_id'], how='left', suffixes=['', '_'])
x_test['year'] = 2015
x_test['month'] = 9
x_test['month_rate'] = days_of_month_rate(2015, 11)
#x_test[int_features] = X_test[int_features].astype('int32')
x_test = x_test[x_train.columns]
datasets = [x_train,x_val, x_test]

          
for dataset in datasets:
    for shop_id in dataset['shop_id'].unique():
        for column in dataset.columns:
            shop_median = dataset[(dataset['shop_id'] == shop_id)][column].median()
            dataset.loc[(dataset[column].isnull()) & (dataset['shop_id'] == shop_id), column] = shop_median
            
# Fill remaining missing values on test set with mean.
x_test.fillna(x_test.mean(), inplace=True)
# x_train.columns
#lre_model_features = ['shop_id','item_id','month', 
#                      'hist_year_mean', 'hist_month_mean','item_cnt_hist_mean',
#                      'item_cnt_hist_std','item_cnt_hist_max','mean_item_price','month_rate'] #0.6663

#lre_model_features = ['item_cnt_month', 'item_cnt_shifted_3', 'item_trend', 
#                      'mean_item_cnt', 'hist_shop_mean','mean_item_price', 'item_cnt_hist_std'] #0.71459

lre_model_features = ['item_cnt_month', 'item_cnt_shifted_3', 'item_trend', 'month', 'month_rate','item_cnt_hist_max'
                      'mean_item_cnt', 'hist_shop_mean','mean_item_price', 'item_cnt_hist_std']

lre_x_train = x_train[lre_model_features]
lre_x_val = x_val[lre_model_features]

# Normalization
scaler = sklearn.preprocessing.MinMaxScaler()
scaler.fit(lre_x_train)
lre_x_train = scaler.transform(lre_x_train)
lre_x_val = scaler.transform(lre_x_val)

# Modeling
from sklearn.linear_model import LinearRegression
lre_model = LinearRegression()
lre_model.fit(lre_x_train, y_train)
lre_model.score(lre_x_val, y_val)
lre_val_pred = lre_model.predict(lre_x_val)
lre_val_pred
#print("x_train", x_train.shape)
#print("lre_x_train", lre_x_train.shape)
#print("y_train", y_train.shape)
#print("lre_x_val", lre_x_val.shape)
#print("y_val", lre_x_train.shape)

# x_train.columns
# Features to use with random forest
#rfr_model_features = ['shop_id','item_id','month', 
#                      'hist_year_mean', 'hist_month_mean','item_cnt_hist_mean',
#                      'item_cnt_hist_std','item_cnt_hist_max','mean_item_price']
# Score 0.7008
# 
rfr_model_features = ['shop_id', 'item_id', 'item_cnt_month', 'sales', 'year',
                      'item_cnt_hist_mean', 'item_cnt_hist_std', 'item_cnt_shifted_1',
                      'hist_shop_mean', 'item_trend', 'mean_item_cnt'] # Score: 0.7249/100 est

rfr_x_train = x_train[rfr_model_features]
rfr_x_val = x_val[rfr_model_features]

rfr_model = RandomForestRegressor(n_estimators=500, max_depth=7, random_state=0, n_jobs=-1)
rfr_model.fit(rfr_x_train, y_train)
rfr_model.score(rfr_x_val, y_val)
# Features to use with XBOST
#print("x_train", x_train.columns)
#xgb_model_features = ['item_cnt_month','item_cnt_hist_min', 'item_cnt_hist_std',
#                      'hist_shop_mean', 'hist_shop_item_mean'] # validation_1-rmse:1.83021

xgb_model_features = ['item_cnt_month','item_cnt_hist_min', 'item_cnt_hist_std',
                      'item_cnt_shifted_2', 'item_cnt_shifted_3',#'item_cnt_shifted_1',
                      'hist_shop_mean', 'hist_shop_item_mean', 'item_trend', 'mean_item_cnt'] # 100 - Est --> validation_1-rmse:1.70364

xgb_x_train = x_train[xgb_model_features]
xgb_x_val = x_val[xgb_model_features]

xgb_model = XGBRegressor(max_depth=8, n_estimators=500, min_child_weight=1000,
                         colsample_bytree=0.7, subsample=0.7, eta=0.3, seed=12345)

xgb_model.fit(xgb_x_train, y_train, 
              eval_set=[(xgb_x_train, y_train), (xgb_x_val, y_val)],
              eval_metric='rmse',               
              verbose=True, 
              early_stopping_rounds=20)

xgb_model.score
#x_train.isna().sum()
#cat_features = [0, 1,2,3]
#cbt_model_features = ['item_cnt_month','item_cnt_hist_mean', 'item_cnt_hist_std',
#                      'item_cnt_shifted_2', 'item_cnt_shifted_3',
#                      'hist_shop_mean', 'hist_shop_item_mean', 'item_trend', 'mean_item_cnt'] 

cbt_model_features = ['month','year','shop_id','item_id']

cbt_x_train = x_train[cbt_model_features]
cbt_x_val = x_val[cbt_model_features]

cbt_x_train['month'] = cbt_x_train['month'].astype(np.int) 
cbt_x_val['month'] = cbt_x_val['month'].astype(np.int) 
cbt_x_train['year'] = cbt_x_train['year'].astype(np.int) 
cbt_x_val['year'] = cbt_x_val['year'].astype(np.int) 
cbt_x_train['shop_id'] = cbt_x_train['shop_id'].astype(np.int) 
cbt_x_val['shop_id'] = cbt_x_val['shop_id'].astype(np.int) 
cbt_x_train['item_id'] = cbt_x_train['item_id'].astype(np.int) 
cbt_x_val['item_id'] = cbt_x_val['item_id'].astype(np.int)

catboost_model = CatBoostRegressor(
    iterations=100,
    max_ctr_complexity=4,
    random_seed=0,
    od_type='Iter',
    od_wait=25,
    verbose=50,
    depth=4
)

catboost_model.fit(
    cbt_x_train, y_train,
    cat_features=cbt_model_features,
    eval_set=(cbt_x_val, y_val)
)

catboost_model.score
#Prediction
y_pred=rfr_model.predict(x_val)
# a data frame with actual and predicted values of y
evaluate = pd.DataFrame({'Actual': y_val.values.flatten(), 'Predicted': y_pred.flatten()})
evaluate.head(10).T
y_test = x_test
y_test[target_feature]=0
y_test = y_test[target_feature]
rfr_x_test = x_test[rfr_model_features]

print("y_test", y_test.shape)
print("rfr_x_test", rfr_x_test.shape)
print("x_test", x_test.shape)


y_test = x_test
y_test[target_feature]=0
y_test = y_test[target_feature]

rfr_x_test = x_test[rfr_model_features]
lre_x_test = x_test[lre_model_features]
lre_x_test = scaler.transform(lre_x_test) # Normalized
xgb_x_test = x_test[xgb_model_features]

rfr_val_preds = rfr_model.predict(rfr_x_val)
lre_val_preds = lre_model.predict(lre_x_val)
xgb_val_preds = xgb_model.predict(xgb_x_val)

rfr_test_preds = rfr_model.predict(rfr_x_test)
lre_test_preds = lre_model.predict(lre_x_test)
xgb_test_preds = xgb_model.predict(xgb_x_test)


stacked_val_predictions = np.column_stack((rfr_val_preds,lre_val_preds, xgb_val_preds))
stacked_test_predictions = np.column_stack((rfr_test_preds,lre_test_preds, xgb_test_preds))

metaModel = LinearRegression()
metaModel.fit(stacked_val_predictions, y_val)
final_predictions = metaModel.predict(stacked_test_predictions)
#final_predictions
#rfr_test_preds
#lre_test_preds
#xgb_test_preds
#lre_x_test
test2 = pd.read_csv(base_path+"test.csv");
prediction_df = pd.DataFrame(test2['ID'], columns=['ID'])
prediction_df['item_cnt_month'] = final_predictions.clip(0., 20.)
prediction_df.to_csv('submission_06.csv', index=False)
prediction_df.head(10)
df = pd.DataFrame({"Col1": [10, 20, 15, 30, 45],
                   "Col2": [13, 23, 18, 33, 48],
                   "Col3": [17, 27, 22, 37, 52]},
                  index=pd.date_range("2020-01-01", "2020-01-05"))


df
# Prueba para enviar a la plataforma
test = pd.read_csv(base_path+"test.csv");
copy = sales_train
copy = copy[copy["item_price"]<=40000]
copy = copy[copy["item_cnt_day"]<=50]
copy = copy[copy["item_cnt_day"]>0]
sum_by_month = copy.groupby(["shop_id","item_id","date_block_num"] , as_index=False)["item_cnt_day"].agg("sum")
mean_expected = sum_by_month.groupby(["shop_id","item_id"] , as_index=False)["item_cnt_day"].agg("mean")
print(mean_expected.shape)
print(test.shape)
result = pd.merge(test, mean_expected, on=['shop_id','item_id',], how='left')

test[test['shop_id']==10]