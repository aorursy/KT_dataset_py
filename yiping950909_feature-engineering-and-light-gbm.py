# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import datetime
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
%matplotlib inline
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
h_transaction = pd.read_csv('../input/historical_transactions.csv')
ms = h_transaction.isnull().sum()/len(h_transaction)
print(ms)
msno.bar(h_transaction)
new_train = train.merge(h_transaction, how = 'inner')
print(new_train.info())
print(train.info())
print(new_train.head(100))
b_engin = new_train.loc[:,['card_id', 'purchase_amount','month_lag']]
engin = b_engin.groupby(['card_id'])['purchase_amount','month_lag'].mean()
print(engin.info())
new_train['authorized_flag'] = str(new_train['authorized_flag'])
print(new_train.info())
new_train.first_active_month = pd.to_datetime(new_train.first_active_month)
new_train.purchase_date = pd.to_datetime(new_train.purchase_date)
new_train['purchase_time_length'] = (new_train.purchase_date - new_train.first_active_month)
activity_month = new_train.loc[:, ['card_id','purchase_time_length']]
activity_month['trans30'] = activity_month['purchase_time_length'] <= datetime.timedelta(days=30)
activity_month['trans30'][activity_month['trans30'] == True] = 1
activity_month['trans30'][activity_month['trans30'] == False] = 0
activity_month_1 = activity_month.groupby(['card_id'])['trans30'].sum()
print(activity_month_1.head())
activity_month_1 = activity_month_1.to_frame()
activity_month2 = new_train.loc[:, ['card_id','purchase_time_length']]
activity_month2['trans180'] = activity_month2['purchase_time_length'] <= datetime.timedelta(days=180)
activity_month2['trans180'][activity_month2['trans180'] == True] = 1
activity_month2['trans180'][activity_month2['trans180'] == False] = 0
activity_month_6 = activity_month2.groupby(['card_id'])['trans180'].sum()
print(activity_month_6.head())
activity_month_6 = activity_month_6.to_frame()
activity_month3 = new_train.loc[:, ['card_id','purchase_time_length']]
activity_month3['trans90'] = activity_month3['purchase_time_length'] <= datetime.timedelta(days=90)
activity_month3['trans90'][activity_month3['trans90'] == True] = 1
activity_month3['trans90'][activity_month3['trans90'] == False] = 0
activity_month_3 = activity_month3.groupby(['card_id'])['trans90'].sum()
print(activity_month_3.head())
activity_month_3 = activity_month_3.to_frame()
print(new_train.info())
# further data engineering on transaction history
c_engin = new_train.loc[:,['card_id', 'purchase_amount','month_lag','merchant_category_id','state_id','city_id']]
engin2_max = c_engin.loc[:,['card_id','purchase_amount','month_lag']].groupby('card_id')['purchase_amount','month_lag'].max()
#change the columns name
engin2_max.columns = ['purchase_amount_max','month_lag_max']
#create columns of min values of 'purchase_amount','month_lag'
engin2_min = c_engin.loc[:,['card_id','purchase_amount','month_lag']].groupby('card_id')['purchase_amount','month_lag'].min()
#change the columns name
engin2_min.columns = ['purchase_amount_min','month_lag_min']
#create columns of median values of 'purchase_amount','month_lag'
engin2_median = c_engin.loc[:,['card_id','purchase_amount','month_lag']].groupby('card_id')['purchase_amount','month_lag'].median()
#change the columns name
engin2_median.columns = ['purchase_amount_median','month_lag_median']
#create columns of std values of 'purchase_amount','month_lag'
engin2_std = c_engin.loc[:,['card_id','purchase_amount','month_lag']].groupby('card_id')['purchase_amount','month_lag'].std()
#change the columns name
engin2_std.columns = ['purchase_amount_std','month_lag_std']
c_engin['merchant_category_id'] = c_engin['merchant_category_id'].apply(int)
c_engin['state_id'] =  c_engin['state_id'].apply(int)
c_engin['city_id'] =  c_engin['city_id'].apply(int)
#create columns of unique count value of state id, city and category
engin2_u_count = c_engin.loc[:,['card_id','merchant_category_id','state_id','city_id']].groupby('card_id')['merchant_category_id','state_id','city_id'].agg('nunique')
engin2_u_count.columns = ['merchant_category_id_u_count','state_id_u_count','city_id_u_count']
c_engin.loc[:,['card_id','merchant_category_id','state_id','city_id']].head()
#join! every! things!
train1 = train.merge(engin,on = 'card_id', how = 'inner')
train2 = train1.merge(activity_month_1,on = 'card_id', how = 'inner')
train3 = train2.merge(activity_month_3,on = 'card_id', how = 'inner')
train4 = train3.merge(activity_month_6,on = 'card_id', how = 'inner')
train4 = train4.merge(engin2_max,on = 'card_id', how = 'inner')
train4 = train4.merge(engin2_min,on = 'card_id', how = 'inner')
train4 = train4.merge(engin2_median,on = 'card_id', how = 'inner')
train4 = train4.merge(engin2_std,on = 'card_id', how = 'inner')
train4 = train4.merge(engin2_u_count,on = 'card_id', how = 'inner')
train4['trans30_n'] = (train4['trans30'] - min(train4['trans30']))/(max(train4['trans30']) - min(train4['trans30']))
train4['trans90_n'] = (train4['trans90'] - min(train4['trans90']))/(max(train4['trans90']) - min(train4['trans90']))
train4['trans180_n'] = (train4['trans180'] - min(train4['trans180']))/(max(train4['trans180']) - min(train4['trans180']))
train4['purchase_amount_n'] = (train4['purchase_amount'] - min(train4['purchase_amount']))/(max(train4['purchase_amount']) - min(train4['purchase_amount']))
train4['month_lag_n'] = (train4['month_lag'] - min(train4['month_lag']))/(max(train4['month_lag']) - min(train4['month_lag']))
train_relation = train4.iloc[:, [1,2,3,5,6,7,11,12,13,14]]
train4['purchase_amount_n'][train4['purchase_amount_n'] > 0.5] = np.average(train4['purchase_amount_n'])
train4.first_active_month = pd.to_datetime(train4.first_active_month)
print(train4.head())
train4['year_active'], train4['month_active'], train4['day_active'] = train4.first_active_month.dt.year, train4.first_active_month.dt.month,train4.first_active_month.dt.day 
print(train4.info())
sns.pairplot(train_relation)
sns.violinplot(y = "trans30", x="feature_1", data = train4);
sns.barplot(y = "trans90", x="feature_1",hue = 'feature_2', data = train4, palette = "RdBu");
sns.barplot(y = "trans90", x="feature_1",hue = 'feature_3', data = train4);
fig = plt.figure(figsize = (14,14))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)
sns.violinplot(x="feature_1", y="target",
            data = train4, ax =ax1, palette="Set2");
sns.violinplot(x="feature_1", y="trans30",
            data = train4, ax =ax2, palette="Set2");
sns.violinplot(x="feature_1", y="month_lag",
            data = train4, ax =ax3, palette="Set2");
sns.violinplot(x="feature_1", y="purchase_amount",
            data = train4, ax =ax4, palette="Set2");
fig = plt.figure(figsize = (14,14))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)
sns.violinplot(x="feature_2", y="target",
            data = train4, ax =ax1);
sns.violinplot(x="feature_2", y="trans30",
            data = train4, ax =ax2);
sns.violinplot(x="feature_2", y="month_lag",
            data = train4, ax =ax3);
sns.violinplot(x="feature_2", y="purchase_amount",
            data = train4, ax =ax4);
train4.corr()
from sklearn.ensemble import RandomForestRegressor
train_new = train4.drop(['card_id'], axis=1)
train_independent = train_new.drop(['target','first_active_month'], axis = 1)
print(train_new.columns)
model = RandomForestRegressor()
train_independent = pd.get_dummies(train_independent)
model.fit(train_independent,train_new.target)

importances = model.feature_importances_
print(importances)
#经过分析筛选前十的variable是important的
import matplotlib.pyplot as plt
plt.style.use('ggplot')
features = train_independent.columns
print(features)
importance = model.feature_importances_
find = np.argsort(importance[0:30])
print(find)  # top 20 features
plt.title('selected features')
plt.barh(range(len(find)), importance[find], color='orange', align='center')
plt.yticks(range(len(find)), [features[i] for i in find])
plt.xlabel('Importance')
plt.show()
train_m = train4.loc[:,['feature_1', 'feature_2', 'feature_3', 'purchase_amount', 'month_lag',
        'trans30_n', 'trans90_n',
       'trans180_n', 'purchase_amount_n', 'month_lag_n', 'year_active',
       'month_active','purchase_amount_max_x',
       'month_lag_max_x',
       'purchase_amount_min', 'month_lag_min', 'purchase_amount_median',
       'month_lag_median', 'purchase_amount_std', 'month_lag_std',
       'merchant_category_id_u_count', 'state_id_u_count', 'city_id_u_count']]
target = train4.loc[:,['target']]
print(train_m.head())
from sklearn.model_selection import train_test_split
train_X,test_X, train_y, test_y = train_test_split(train_m,target,test_size = 0.3,random_state = 0)
#LIGHT GBM方法预测
lgb_train = lgb.Dataset(train_X, train_y) # 将数据保存到LightGBM二进制文件将使加载更快
lgb_eval = lgb.Dataset(test_X, test_y, reference=lgb_train) 


params = {
    'task': 'train',  
    'objective': 'regression', 
    'metric': {'l2', 'rmse'},  
    'num_leaves': 12,  
    'learning_rate': 0.23,  
    'feature_fraction': 0.9, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 1, 
    'verbose': 1,
    "reg_alpha": 1.5,
    "reg_lambda": 1,
    "max_depth": 7,
    "min_child_samples": 8
}

print('Start training...')

gbm = lgb.train(params,lgb_train,num_boost_round=20,valid_sets=lgb_eval,early_stopping_rounds=5)
 
print('Save model...') 
 
gbm.save_model('model.txt') \
 
print('Start predicting...')

y_pred = gbm.predict(test_X, num_iteration=gbm.best_iteration) 
print(y_pred)
print('The rmse of prediction is:', mean_squared_error(test_y, y_pred) ** 0.5) 