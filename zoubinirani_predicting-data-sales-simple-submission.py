import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.externals import joblib
from sklearn.datasets import load_digits
item_categories = pd.read_csv("../input/item_categories.csv")
items = pd.read_csv("../input/items.csv")
sales = pd.read_csv('../input/sales_train.csv.gz',compression='gzip')
shops = pd.read_csv("../input/shops.csv")
data_to_predict = pd.read_csv('../input/test.csv.gz',compression='gzip')
sales = sales.sort_values(by = ['date_block_num'])
item_categories.head()
items.head()
sales.head()
shops.head()
data_to_predict.head()
item_categories.describe()
items.describe()
sales.describe()
shops.describe()
data_to_predict.describe()
#several different types of shops and items (many different combinations)

#past data submissions saying mean submission should be ~0.33
sales.drop(['date'], axis=1, inplace=True)
sales = sales[sales["date_block_num"] >= 5] #for recency
#source all required data
unique_dates = sales['date_block_num'].unique()
new_sales = pd.DataFrame()
for date in unique_dates:
    unique_shops = sales[sales['date_block_num'] == date]['shop_id'].unique()
    unique_items = sales[sales['date_block_num'] == date]['item_id'].unique()
    for shop in unique_shops:
        date_criteria = sales['date_block_num'] == date
        shop_criteria = sales['shop_id'] == shop
        unique_items_in_shop = sales[date_criteria & shop_criteria]['item_id'].unique()
        missing_items = np.setdiff1d(unique_items, unique_items_in_shop)
        date_array = pd.Series([date] * len(missing_items))
        shop_array = pd.Series([shop] * len(missing_items))
        item_array = pd.Series(missing_items)
        item_price_array = pd.Series([999999] * len(missing_items))    
        item_cnt_day_array = pd.Series([0] * len(missing_items))
        new_data_to_append = pd.concat([date_array, shop_array,
                                        item_array, item_price_array, item_cnt_day_array], axis=1)
        
        new_sales = pd.concat([new_sales, new_data_to_append], axis=0)
new_sales.rename(columns={0: 'date_block_num',1: 'shop_id',
                          2: 'item_id',3: 'item_price', 4: 'item_cnt_day'},
                          inplace=True)

sales = pd.concat([sales, new_sales], axis=0)

del new_sales
del new_data_to_append
group_lists = [['date_block_num', 'shop_id', 'item_id'],
               ['date_block_num', 'shop_id'],
               ['date_block_num', 'item_id']]

group_name = ['shop_item', 'shop', 'item']

for i in range(len(group_name)):
    
    group = group_lists[i]
    name = group_name[i]
    
    print(i)

    sales_sum = sales.groupby(group).item_cnt_day.sum().reset_index()
    sales_count = sales.groupby(group).item_cnt_day.count().reset_index()

    sales_sum_name = ''.join([name, "_sum"])
    sales_count_name = ''.join([name, "_count"])

    sales_sum = sales_sum.rename(columns={'item_cnt_day': sales_sum_name})
    sales_count = sales_count.rename(columns={'item_cnt_day': sales_count_name})
    
    sales_sum.drop_duplicates(inplace=True) #drop duplicates
    sales_count.drop_duplicates(inplace=True) #drop duplicates

    sales = pd.merge(sales, sales_sum, how = "left")
    sales = pd.merge(sales, sales_count, how = "left")

del sales_sum
del sales_count
group_lists = [['date_block_num', 'shop_id', 'item_id'],
               ['date_block_num', 'shop_id'],
               ['date_block_num', 'item_id']]

group_name = ['shop_item', 'shop', 'item']

for i in range(len(group_name)):
    
    group = group_lists[i]
    name = group_name[i]
    to_avg = ''.join([group_name[i], "_sum"])
    
    print(i)

    sales_avg = sales.groupby(group)[to_avg].median().reset_index()
    sales_avg_name = ''.join([name, "_avg"])
    sales_avg = sales_avg.rename(columns={to_avg: sales_avg_name})
        
    sales_avg.drop_duplicates(inplace=True) #drop duplicates
    
    sales = pd.merge(sales, sales_avg, how = "left")

del sales_avg
sales.drop(['item_cnt_day'], axis=1, inplace=True)
sales.drop_duplicates(['date_block_num', 'shop_id', 'item_id'], inplace=True) #drop duplicates
#set constants for join
sales['is_train'] = 1
sales['ID'] = np.nan

data_to_predict['is_train'] = 0
data_to_predict['date_block_num'] = 34

#join where applicable
sales = pd.concat([sales, data_to_predict], axis=0)
#done for target, using same structure as lag below

columns_to_lag = ['shop_item_sum']
index_columns = ['shop_id', 'item_id', 'date_block_num']
lag_range = [1]

for month in lag_range: #used structure from Assignment 4
    print (month)
    train_shift = sales[columns_to_lag + index_columns].copy()
    train_shift['date_block_num'] = train_shift['date_block_num'] - month
    foo = lambda x: '{}_lag_{}'.format(x, month) if x in columns_to_lag else x
    train_shift = train_shift.rename(columns=foo)
    sales = pd.merge(sales, train_shift, on=index_columns, how='left').fillna(0)

sales.rename(columns={"shop_item_sum_lag_1": 'target'}, inplace=True)

del train_shift

#target distribution is huge; needs to be truncated to lead to best training
# avg target in board is 0.33; range to submit is 0-20
# monthly sales can be > 1000; must be truncated before training
columns_to_lag = ['shop_sum', 'shop_item_sum', 'shop_item_count', 'shop_count',
                 'item_sum', 'item_price', 'item_count', "item_avg", "shop_item_avg","shop_avg"]

index_columns = ['shop_id', 'item_id', 'date_block_num']

lag_range = [1,2,3,4,5,6,7,8,9,10,11,12]

for month in lag_range: #used structure in Assignment 4
    print (month)
    train_shift = sales[index_columns + columns_to_lag].copy()
    train_shift['date_block_num'] = train_shift['date_block_num'] + month
    foo = lambda x: '{}_lag_{}'.format(x, month) if x in columns_to_lag else x
    train_shift = train_shift.rename(columns=foo)
    sales = pd.merge(sales, train_shift, on=index_columns, how='left').fillna(0)
    del train_shift
sales = sales[sales["date_block_num"] >= 16] #due to lag

#start here
sales = pd.merge(sales, shops, how = "left")
sales = pd.merge(sales, items, how = "left")
sales = pd.merge(sales, item_categories, how = "left")
shop_name_expand = sales["shop_name"].str.split(pat=" ", expand = True)
shop_name_expand = shop_name_expand.loc[:,[0,1]] #first two columns
shop_name_expand.rename(columns={0: 'shop_0',1:'shop_1'},inplace=True)
item_category_name_expand = sales["item_category_name"].str.split(pat=" ", expand = True)
item_category_name_expand = item_category_name_expand.loc[:,[0,1]] #first two columns
item_category_name_expand.rename(columns={0: 'category_0',1:'category_1'},inplace=True)
#cbind by column
sales = pd.concat([sales, shop_name_expand, item_category_name_expand], axis=1)
shop_0_encoder = preprocessing.LabelEncoder()
shop_1_encoder = preprocessing.LabelEncoder()
category_0_encoder = preprocessing.LabelEncoder()
category_1_encoder = preprocessing.LabelEncoder()

sales['shop_0'] = shop_0_encoder.fit_transform(sales['shop_0'].astype(str))
sales['shop_1'] = shop_1_encoder.fit_transform(sales['shop_1'].astype(str))
sales['category_0'] = category_0_encoder.fit_transform(sales['category_0'].astype(str))
sales['category_1'] = category_1_encoder.fit_transform(sales['category_1'].astype(str))
splits = KFold(n_splits=5, shuffle=False)
splits.get_n_splits(sales)

sales_mean_encoded = pd.DataFrame(None)

for split_index, validation_index in splits.split(sales):
    all_data_train = sales.iloc[split_index, :]
    all_data_test = sales.iloc[validation_index, :]
    
    category_id_target_mean = all_data_train.groupby('category_0').target.mean()
    all_data_train['category_target_enc'] = all_data_train['category_0'].map(category_id_target_mean)
    category_id_target_test = all_data_train.groupby('category_0').category_target_enc.mean()
    all_data_test['category_target_enc'] = all_data_test['category_0'].map(category_id_target_test)
    
    shop_id_target_mean = all_data_train.groupby('shop_id').target.mean()
    all_data_train['shop_target_enc'] = all_data_train['shop_id'].map(shop_id_target_mean)
    shop_id_target_test = all_data_train.groupby('shop_id').shop_target_enc.mean()
    all_data_test['shop_target_enc'] = all_data_test['shop_id'].map(shop_id_target_test)
    
    temp = [sales_mean_encoded, all_data_test]
    sales_mean_encoded = pd.concat(temp)
sales_mean_encoded.category_target_enc.fillna(0.3343, inplace=True)
sales_mean_encoded.shop_target_enc.fillna(0.3343, inplace=True)
sales = sales_mean_encoded
#drop for now maybe reverse later
sales.drop(['shop_name', 'item_name', 'item_category_name'], axis=1, inplace = True)
#drop since not needed due to lag
sales.drop(['item_count', 'item_sum', 'item_price', 'shop_count',
            "shop_item_count", 'shop_item_sum', 'shop_sum',
            'item_avg', 'shop_avg', 'shop_item_avg'], axis=1, inplace = True)
added_back_columns = ["ID", "is_train", "target", "date_block_num"]

categorical_columns = ["shop_id", 'item_id', 'item_category_id', 'shop_0', 'shop_1', 'category_0', 'category_1']
        
numerical_columns = ["date_block_num",
                    
       'shop_sum_lag_1', 'shop_item_sum_lag_1','shop_item_count_lag_1', 'shop_count_lag_1', 'item_sum_lag_1',
       'item_price_lag_1', 'item_count_lag_1','shop_avg_lag_1', 'shop_item_avg_lag_1', 'shop_item_avg_lag_1',
                    
       'shop_sum_lag_2', 'shop_item_sum_lag_2','shop_item_count_lag_2', 'shop_count_lag_2', 'item_sum_lag_2',
       'item_price_lag_2', 'item_count_lag_2','shop_avg_lag_2', 'shop_item_avg_lag_2', 'shop_item_avg_lag_2',

       'shop_sum_lag_3', 'shop_item_sum_lag_3','shop_item_count_lag_3', 'shop_count_lag_3', 'item_sum_lag_3',
       'item_price_lag_3', 'item_count_lag_3','shop_avg_lag_3', 'shop_item_avg_lag_3', 'shop_item_avg_lag_3',
            
       'shop_sum_lag_4', 'shop_item_sum_lag_4','shop_item_count_lag_4', 'shop_count_lag_4', 'item_sum_lag_4',
       'item_price_lag_4', 'item_count_lag_4','shop_avg_lag_4', 'shop_item_avg_lag_4', 'shop_item_avg_lag_4',
             
       'shop_sum_lag_5', 'shop_item_sum_lag_5','shop_item_count_lag_5', 'shop_count_lag_5', 'item_sum_lag_5',
       'item_price_lag_5', 'item_count_lag_5','shop_avg_lag_5', 'shop_item_avg_lag_5', 'shop_item_avg_lag_5',
             
       'shop_sum_lag_6', 'shop_item_sum_lag_6','shop_item_count_lag_6', 'shop_count_lag_6', 'item_sum_lag_6',
       'item_price_lag_6', 'item_count_lag_6','shop_avg_lag_6', 'shop_item_avg_lag_6', 'shop_item_avg_lag_6',
            
       'shop_sum_lag_7', 'shop_item_sum_lag_7','shop_item_count_lag_7', 'shop_count_lag_7', 'item_sum_lag_7',
       'item_price_lag_7', 'item_count_lag_7','shop_avg_lag_7', 'shop_item_avg_lag_7', 'shop_item_avg_lag_7',

       'shop_sum_lag_8', 'shop_item_sum_lag_8','shop_item_count_lag_8', 'shop_count_lag_8', 'item_sum_lag_8',
       'item_price_lag_8', 'item_count_lag_8','shop_avg_lag_8', 'shop_item_avg_lag_8', 'shop_item_avg_lag_8',
                    
       'shop_sum_lag_9', 'shop_item_sum_lag_9','shop_item_count_lag_9', 'shop_count_lag_9', 'item_sum_lag_9',
       'item_price_lag_9', 'item_count_lag_9','shop_avg_lag_9', 'shop_item_avg_lag_9', 'shop_item_avg_lag_9',
             
       'shop_sum_lag_10', 'shop_item_sum_lag_10','shop_item_count_lag_10', 'shop_count_lag_10', 'item_sum_lag_10',
       'item_price_lag_10', 'item_count_lag_10','shop_avg_lag_10', 'shop_item_avg_lag_10', 'shop_item_avg_lag_10',

       'shop_sum_lag_11', 'shop_item_sum_lag_11','shop_item_count_lag_11', 'shop_count_lag_11', 'item_sum_lag_11',
       'item_price_lag_11', 'item_count_lag_11','shop_avg_lag_11', 'shop_item_avg_lag_11', 'shop_item_avg_lag_11',
                     
       'shop_sum_lag_12', 'shop_item_sum_lag_12','shop_item_count_lag_12', 'shop_count_lag_12', 'item_sum_lag_12',
       'item_price_lag_12', 'item_count_lag_12','shop_avg_lag_12', 'shop_item_avg_lag_12', 'shop_item_avg_lag_12']
sales_added = sales[added_back_columns]
sales_categorical = sales[categorical_columns]
sales_numerical = sales[numerical_columns]
sales_categorical.fillna(0, inplace=True)
sales_categorical = sales_categorical.applymap(int) #convert to int since obj
sales_pca = sales_numerical
#for preprocessing
x = sales_pca.values 
x = np.nan_to_num(x)
normalization = preprocessing.MinMaxScaler()
x_normalized = normalization.fit_transform(x)
sales_pca = pd.DataFrame(x_normalized)
sales_pca = pd.DataFrame(x)
pca = PCA(n_components=10)
pca_to_append = pca.fit_transform(sales_pca)
pca_to_append = pd.DataFrame(pca_to_append)
pca_to_append.rename(columns={0: 'pca_0',
                              1: 'pca_1',
                              2: 'pca_2',
                              3: 'pca_3',
                              4: 'pca_4',
                              5: 'pca_5',
                              6: 'pca_6',
                              7: 'pca_7',
                              8: 'pca_8',
                              9: 'pca_9'}, inplace=True)
sales = pd.concat([sales_added, sales_categorical, sales_pca, pca_to_append], axis=1) #join back
#seperate to train/test set
data_to_predict = sales[sales['is_train'] == 0]
sales = sales[sales['is_train'] == 1]
#only use date_block_num 18 to 32 since have valid data here (can't use 33 since no target information)
sales = sales[sales['date_block_num']>= 18] #use semi-recent data (hyper tuned)
sales = sales[sales['date_block_num']<= 32]
#clip where target_sales is less than or equal to 50 and greater or equal to 0 (clip last)
sales = sales[sales['target']<= 80] 
sales = sales[sales['target']>= 0]
target = sales['target']
sales.drop(['target', 'is_train', 'ID'], axis = 1, inplace = True)
#0.2 is good (hyper tuned)
temp_X, hyperparameter_X, temp_y, hyperparameter_y = train_test_split(sales, target, test_size=0.25)
train_X, test_X, train_y, test_y = train_test_split(temp_X, temp_y, test_size=0.25)
#LGB Model (Version 1)
lgbm_train = lgb.Dataset(train_X, train_y)

lgbm_eval = lgb.Dataset(test_X, test_y, reference=lgbm_train)
num_rounds = 2000
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'rmse'},
    'num_leaves': 100,
    'min_data_in_leaf': 1,
    'max_depth': 20,
    'max_bin': 250,
    'learning_rate': 0.3,
    'num_threads': 2,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'verbose': 100,
}

lgbm_evaluate = lgb.train(params, lgbm_train, num_rounds, valid_sets=[lgbm_train, lgbm_eval],
                      early_stopping_rounds=30)
#LGBM Round 2
lgbm_train = lgb.Dataset(train_X, train_y)

lgbm_eval = lgb.Dataset(test_X, test_y, reference=lgbm_train)

num_rounds = 2000

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'rmse'},
    'num_leaves': 5000,
    'min_data_in_leaf': 2,
    'max_depth': 100,
    'max_bin': 5000,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 4,
    'verbose': 1
}

lgbm_2_evaluate = lgb.train(params, lgbm_train, num_rounds, valid_sets=[lgbm_train, lgbm_eval],
                      early_stopping_rounds=30)
#XGRegressor_round 1
baseline_model = xgb.XGBRegressor(max_depth = 8, njobs=-1, n_estimators = 100,
                                  colsample_bytree=0.9, subsample=0.5,
                                  eval_metric = "rmse", eta = 0.45)

XGB_fit = baseline_model.fit(train_X, train_y, eval_set=[(train_X, train_y), (test_X, test_y)],
                   early_stopping_rounds = 20, verbose=True)

#Random Forest
Random_Forest = RandomForestRegressor(max_depth=300, random_state=0, n_jobs=-1, verbose=100)
Random_Forest.fit(train_X, train_y)
GBM_Pred1 = lgbm_evaluate.predict(hyperparameter_X) #GBM
GBM_Pred2 = lgbm_2_evaluate.predict(hyperparameter_X) #GBM
XGR_Pred1 = baseline_model.predict(hyperparameter_X) # XGRegressor
RF_Pred1 = Random_Forest.predict(hyperparameter_X)


GBM_Pred1[GBM_Pred1 > 80] = 80
GBM_Pred2[GBM_Pred2 > 80] = 80
XGR_Pred1[XGR_Pred1 > 80] = 80
RF_Pred1[RF_Pred1 > 80] = 80


GBM_Pred1[GBM_Pred1 < 0] = 0
GBM_Pred2[GBM_Pred2 < 0] = 0
XGR_Pred1[XGR_Pred1 < 0] = 0
RF_Pred1[RF_Pred1 < 0] = 0


ensemble_X = pd.concat([pd.Series(GBM_Pred1),
                        pd.Series(GBM_Pred2),
                        pd.Series(XGR_Pred1),
                        pd.Series(RF_Pred1)], axis=1)
hyper_train_X, hyper_test_X, hyper_train_y, hyper_test_y = train_test_split(ensemble_X, hyperparameter_y, test_size=0.20)


model_for_ensemble = xgb.XGBRegressor(max_depth = 6, njobs=-1, n_estimators = 200,
                                  colsample_bytree=0.9, subsample=0.5,
                                  eval_metric = "rmse", eta = 0.03)

ensemble_model_eval = model_for_ensemble.fit(hyper_train_X, hyper_train_y, eval_set=[(hyper_train_X, hyper_train_y), (hyper_test_X, hyper_test_y)],
                   early_stopping_rounds = 20, verbose=True)

#saving models
# baseline_model.save_binary('XGB_20180205.model')
# lgbm_evaluate.save_model('LGBM_20180205.txt')

ID = data_to_predict["ID"]
data_to_predict.drop(['target', 'is_train', 'ID'], axis = 1, inplace = True)
sGBM_Pred1 = lgbm_evaluate.predict(data_to_predict) #GBM
sGBM_Pred2 = lgbm_2_evaluate.predict(data_to_predict) #GBM
sXGR_Pred1 = baseline_model.predict(data_to_predict) # XGRegressor
sRF_Pred1 = Random_Forest.predict(data_to_predict)


sGBM_Pred1[sGBM_Pred1 > 80] = 80
sGBM_Pred2[sGBM_Pred2 > 80] = 80
sXGR_Pred1[sXGR_Pred1 > 80] = 80
sRF_Pred1[sRF_Pred1 > 80] = 80

sGBM_Pred1[sGBM_Pred1 < 0] = 0
sGBM_Pred2[sGBM_Pred2 < 0] = 0
sXGR_Pred1[sXGR_Pred1 < 0] = 0
sRF_Pred1[sRF_Pred1 < 0] = 0


entry_predictions = pd.concat([pd.Series(sGBM_Pred1),
                        pd.Series(sGBM_Pred2),
                        pd.Series(sXGR_Pred1),
                        pd.Series(sRF_Pred1)], axis=1)

submission_predictions = ensemble_model_eval.predict(entry_predictions)
submission_predictions[submission_predictions > 20] = 20 #predictions > 20 should equal 20
submission_predictions[submission_predictions < 0] = 0 #predictions < 0 should equal 0
submission = pd.concat([pd.Series(np.array(ID)), pd.Series(submission_predictions)], axis=1)
submission.rename(columns={0: 'ID',1:'item_cnt_month'},inplace=True)
submission['ID'] = submission['ID'].astype(int)
#Save models
# lgbm_evaluate.save_model('LGBM_Model1.txt')
# lgbm_2_evaluate.save_model('LGBM_Model2.txt')
# joblib.dump(baseline_model,'XGR_Model.pkl')
# joblib.dump(Random_Forest,'RFR_Model.pkl')

# joblib.dump(ensemble_model_eval,'Ensemble_Model.pkl')
#save Data To Predict
# data_to_predict.to_csv("data_to_predict_with_categories.csv", index=False)
#Save prediction
# submission.to_csv("submission_20180218_v3.csv", index=False)