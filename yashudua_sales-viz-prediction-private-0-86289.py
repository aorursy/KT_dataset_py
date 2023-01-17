import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
from sklearn import metrics


df_sales_train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
df_sales_test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
df_sales_shop = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
df_sales_item = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
df_sales_item_cat = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
df_sales_train.head()
sns.boxplot(x=df_sales_train.item_price)
plt.title('Outliers By Item Price');
sns.boxplot(x=df_sales_train.item_cnt_day)
plt.title('Outliers By Item Day');
df_sales_train = df_sales_train[df_sales_train.item_price<100000]
df_sales_train = df_sales_train[df_sales_train.item_cnt_day<1001]
median = df_sales_train[(df_sales_train.shop_id==32)&(df_sales_train.item_id==2973)&(df_sales_train.date_block_num==4)&(df_sales_train.item_price>0)].item_price.median()
df_sales_train.loc[df_sales_train.item_price<0, 'item_price'] = median
# Якутск Орджоникидзе, 56
df_sales_train.loc[df_sales_train.shop_id == 0, 'shop_id'] = 57
df_sales_test.loc[df_sales_test.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
df_sales_train.loc[df_sales_train.shop_id == 1, 'shop_id'] = 58
df_sales_test.loc[df_sales_test.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
df_sales_train.loc[df_sales_train.shop_id == 10, 'shop_id'] = 11
df_sales_test.loc[df_sales_test.shop_id == 10, 'shop_id'] = 11
df_sales_train_grouped = df_sales_train.groupby(['shop_id','item_id']).agg({'item_cnt_day':'sum', 'item_price':'mean'}).reset_index()
df_sales_train_grouped_negative = df_sales_train_grouped.groupby('shop_id').sum().reset_index().sort_values(by='item_cnt_day').head(10)
df_sales_shop_labels = df_sales_shop.loc[df_sales_shop['shop_id'].isin(df_sales_train_grouped_negative['shop_id']),'shop_name'].reset_index()
legend_list  = df_sales_shop_labels.apply((lambda x : str(x['index']) + ' --> ' + x['shop_name']),axis=1)
plt.figure(figsize=(8,6))
sns.barplot(y='shop_id',x='item_cnt_day',data=df_sales_train_grouped_negative,
            order=df_sales_train_grouped_negative['shop_id'],orient='h')
leg = plt.legend(legend_list,loc='best', bbox_to_anchor=(1.3, 0.515, 0.5, 0.5))
plt.xlabel('Sales')

for item in leg.legendHandles:
    item.set_visible(False)
plt.title('Bottom 10 sales by shop');
df_sales_train_grouped_positive = df_sales_train_grouped.groupby('shop_id').sum().reset_index().sort_values(by='item_cnt_day',ascending=False).head(10)
df_sales_shop_labels = df_sales_shop.loc[df_sales_shop['shop_id'].isin(df_sales_train_grouped_positive['shop_id']),'shop_name'].reset_index()
legend_list  = df_sales_shop_labels.apply((lambda x : str(x['index']) + ' --> ' + x['shop_name']),axis=1)
plt.figure(figsize=(8,6))
sns.barplot(y='shop_id',x='item_cnt_day',data=df_sales_train_grouped_positive,
            order=df_sales_train_grouped_positive['shop_id'],orient='h')
leg = plt.legend(legend_list,loc='best', bbox_to_anchor=(1., 0.515, 0.5, 0.5))
plt.xlabel('Sales')

for item in leg.legendHandles:
    item.set_visible(False)
plt.title('Top 10 sales by shop');
df_sales_train_grouped_negative = df_sales_train_grouped.groupby('item_id').sum().reset_index().sort_values(by='item_cnt_day').head(10)
df_sales_item_labels = df_sales_item.loc[df_sales_item['item_id'].isin(df_sales_train_grouped_negative['item_id']),'item_name'].reset_index()
legend_list  = df_sales_item_labels.apply((lambda x : str(x['index']) + ' --> ' + x['item_name']),axis=1)
plt.figure(figsize=(8,6))
sns.barplot(x='item_id',y='item_cnt_day',data=df_sales_train_grouped_negative,
            order=df_sales_train_grouped_negative['item_id'],orient='v')
leg = plt.legend(legend_list,loc='best', bbox_to_anchor=(1.0, 0.515, 0.5, 0.5))
plt.xlabel('Sales')


for item in leg.legendHandles:
    item.set_visible(False)
plt.title('Bottom 10 sales by Item');
df_sales_train_grouped_positive = df_sales_train_grouped.groupby('item_id').sum().reset_index().sort_values(by='item_cnt_day',ascending=False).head(10)
df_sales_item_labels = df_sales_item.loc[df_sales_item['item_id'].isin(df_sales_train_grouped_positive['item_id']),'item_name'].reset_index()
legend_list  = df_sales_item_labels.apply((lambda x : str(x['index']) + ' --> ' + x['item_name']),axis=1)
plt.figure(figsize=(8,6))
sns.barplot(y='item_id',x='item_cnt_day',data=df_sales_train_grouped_positive,
            order=df_sales_train_grouped_positive['item_id'],orient='h')
leg = plt.legend(legend_list,loc='best', bbox_to_anchor=(1.0, 0.515, 0.5, 0.5))
plt.xlabel('Sales')

for item in leg.legendHandles:
    item.set_visible(False)
plt.title('Top 10 sales by Item');
df_sales_train_grouped_negative = df_sales_train_grouped.sort_values(by='item_cnt_day').head(10)
df_sales_train_grouped_negative['shop & item'] = df_sales_train_grouped_negative.apply((lambda x: str(x['shop_id']) + ' & '+ str(x['item_id'])),axis=1)
df_sales_item_labels = df_sales_item.loc[df_sales_item['item_id'].isin(df_sales_train_grouped_negative['item_id']),'item_name'].reset_index()
legend_list_item  = df_sales_item_labels.apply((lambda x : str(x['index']) + ' --> ' + x['item_name']),axis=1)
df_sales_shop_labels = df_sales_shop.loc[df_sales_shop['shop_id'].isin(df_sales_train_grouped_negative['shop_id']),'shop_name'].reset_index()
legend_list_shop  = df_sales_shop_labels.apply((lambda x : str(x['index']) + ' --> ' + x['shop_name']),axis=1)

legend_list = legend_list_item.append(legend_list_shop)
plt.figure(figsize=(8,6))
sns.barplot(y='shop & item',x='item_cnt_day',data=df_sales_train_grouped_negative,
            order=df_sales_train_grouped_negative['shop & item'],orient='h')
leg = plt.legend(legend_list,loc='best', bbox_to_anchor=(1.0, 0.515, 0.5, 0.5))
plt.xlabel('Sales')

for item in leg.legendHandles:
    item.set_visible(False)
plt.title('Bottom 10 Sales by Shop and item Combination');
df_sales_train_grouped_positive = df_sales_train_grouped.sort_values(by='item_cnt_day',ascending=False).head(10)
df_sales_train_grouped_positive['shop & item'] = df_sales_train_grouped_positive.apply((lambda x: str(x['shop_id']) + ' & '+ str(x['item_id'])),axis=1)
df_sales_item_labels = df_sales_item.loc[df_sales_item['item_id'].isin(df_sales_train_grouped_positive['item_id']),'item_name'].reset_index()
legend_list_item  = df_sales_item_labels.apply((lambda x : str(x['index']) + ' --> ' + x['item_name']),axis=1)
df_sales_shop_labels = df_sales_shop.loc[df_sales_shop['shop_id'].isin(df_sales_train_grouped_positive['shop_id']),'shop_name'].reset_index()
legend_list_shop  = df_sales_shop_labels.apply((lambda x : str(x['index']) + ' --> ' + x['shop_name']),axis=1)

legend_list = legend_list_item.append(legend_list_shop)
plt.figure(figsize=(8,6))
sns.barplot(y='shop & item',x='item_cnt_day',data=df_sales_train_grouped_positive,
            order=df_sales_train_grouped_positive['shop & item'],orient='h')
leg = plt.legend(legend_list,loc='best', bbox_to_anchor=(1.0, 0.515, 0.5, 0.5))
plt.xlabel('Sales')
for item in leg.legendHandles:
    item.set_visible(False)
plt.title('Top 10 Sales by Shop and item Combination');
train = df_sales_train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day':'sum', 'item_price':'mean'}).reset_index()
def create_lag_feature(df,lags,col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for lag in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id',col+'_lag_'+str(lag)]
        shifted['date_block_num'] += lag
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'],how = 'left')
    
    return df
train = create_lag_feature(train,range(1,13),'item_cnt_day')
train.fillna(value=0,inplace=True)

train = create_lag_feature(train,range(1,13),'item_price')
train.fillna(value=0,inplace=True)
train_shop = df_sales_train.groupby(['date_block_num','shop_id']).agg({'item_cnt_day':'sum', 'item_price':'mean'}).reset_index()
def create_lag_feature(df,lags,col):
    tmp = df[['date_block_num','shop_id',col]]
    for lag in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id',col+'_shop_lag_'+str(lag)]
        shifted['date_block_num'] += lag
        df = pd.merge(df, shifted, on=['date_block_num','shop_id'],how = 'left')
    
    return df
train_shop = create_lag_feature(train_shop,range(1,13),'item_cnt_day')
train_shop.fillna(value=0,inplace=True)

train_shop = create_lag_feature(train_shop,range(1,13),'item_price')
train_shop.fillna(value=0,inplace=True)


train_shop.rename(columns = {'item_cnt_day':'item_cnt_day_shop','item_price':'item_price_shop'},inplace=True)
train_item = df_sales_train.groupby(['date_block_num','item_id']).agg({'item_cnt_day':'sum', 'item_price':'mean'}).reset_index()
def create_lag_feature(df,lags,col):
    tmp = df[['date_block_num','item_id',col]]
    for lag in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','item_id',col+'_item_lag_'+str(lag)]
        shifted['date_block_num'] += lag
        df = pd.merge(df, shifted, on=['date_block_num','item_id'],how = 'left')
    
    return df
train_item = create_lag_feature(train_item,range(1,13),'item_cnt_day')
train_item.fillna(value=0,inplace=True)

train_item = create_lag_feature(train_item,range(1,13),'item_price')
train_item.fillna(value=0,inplace=True)

train_item.rename(columns = {'item_cnt_day':'item_cnt_day_item','item_price':'item_price_item'},inplace=True)
train = pd.merge(train,train_item,on=['date_block_num','item_id'],how = 'left')
train = pd.merge(train,train_shop,on=['date_block_num','shop_id'],how = 'left')
train['month'] = train['date_block_num'] % 12
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
train['days'] = train['month'].map(days).astype(np.int8)
train['Year'] = (train['date_block_num'] // 12 ) + 2013
sns.pointplot(x='month', y='item_cnt_day', hue='Year', data=train,estimator=np.sum)
df_sales_shop['city'] = df_sales_shop['shop_name'].str.split(' ').map(lambda x: x[0])
df_sales_shop.loc[df_sales_shop.city == '!Якутск', 'city'] = 'Якутск'
df_sales_shop['city_code'] = LabelEncoder().fit_transform(df_sales_shop['city'])

coords = dict()
coords['Якутск'] = (62.028098, 129.732555, 4)
coords['Адыгея'] = (44.609764, 40.100516, 3)
coords['Балашиха'] = (55.8094500, 37.9580600, 1)
coords['Волжский'] = (53.4305800, 50.1190000, 3)
coords['Вологда'] = (59.2239000, 39.8839800, 2)
coords['Воронеж'] = (51.6720400, 39.1843000, 3)
coords['Выездная'] = (0, 0, 0)
coords['Жуковский'] = (55.5952800, 38.1202800, 1)
coords['Интернет-магазин'] = (0, 0, 0)
coords['Казань'] = (55.7887400, 49.1221400, 4)
coords['Калуга'] = (54.5293000, 36.2754200, 4)
coords['Коломна'] = (55.0794400, 38.7783300, 4)
coords['Красноярск'] = (56.0183900, 92.8671700, 4)
coords['Курск'] = (51.7373300, 36.1873500, 3)
coords['Москва'] = (55.7522200, 37.6155600, 1)
coords['Мытищи'] = (55.9116300, 37.7307600, 1)
coords['Н.Новгород'] = (56.3286700, 44.0020500, 4)
coords['Новосибирск'] = (55.0415000, 82.9346000, 4)
coords['Омск'] = (54.9924400, 73.3685900, 4)
coords['РостовНаДону'] = (47.2313500, 39.7232800, 3)
coords['СПб'] = (59.9386300, 30.3141300, 2)
coords['Самара'] = (53.2000700, 50.1500000, 4)
coords['Сергиев'] = (56.3000000, 38.1333300, 4)
coords['Сургут'] = (61.2500000, 73.4166700, 4)
coords['Томск'] = (56.4977100, 84.9743700, 4)
coords['Тюмень'] = (57.1522200, 65.5272200, 4)
coords['Уфа'] = (54.7430600, 55.9677900, 4)
coords['Химки'] = (55.8970400, 37.4296900, 1)
coords['Цифровой'] = (0, 0, 0)
coords['Чехов'] = (55.1477000, 37.4772800, 4)
coords['Ярославль'] = (57.6298700, 39.8736800, 2) 

df_sales_shop['city_coord_l'] = df_sales_shop['city'].apply(lambda x: coords[x][0])
df_sales_shop['city_coord_lt'] = df_sales_shop['city'].apply(lambda x: coords[x][1])
df_sales_shop['country_part'] = df_sales_shop['city'].apply(lambda x: coords[x][2])
df_sales_shop = df_sales_shop[['shop_id','city_code', 'city_coord_l', 'city_coord_lt', 'country_part']]


df_sales_item_cat['split'] = df_sales_item_cat['item_category_name'].str.split('-')
df_sales_item_cat['type'] = df_sales_item_cat['split'].map(lambda x: x[0].strip())
df_sales_item_cat['type_code'] = LabelEncoder().fit_transform(df_sales_item_cat['type'])
# if subtype is nan then type
df_sales_item_cat['subtype'] = df_sales_item_cat['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
df_sales_item_cat['subtype_code'] = LabelEncoder().fit_transform(df_sales_item_cat['subtype'])
df_sales_item_cat = df_sales_item_cat[['item_category_id','type_code', 'subtype_code']]

train = pd.merge(train,df_sales_item[['item_id','item_category_id']],on='item_id',how='inner')
train = pd.merge(train, df_sales_shop, on='shop_id',how = 'left')
train = pd.merge(train, df_sales_item_cat, on='item_category_id',how = 'left')
def revenues(df):
    df['revenue_lag_1'] = df['item_cnt_day_lag_1']*df['item_price_lag_1']
    df['revenue_lag_item_1'] = df['item_cnt_day_item_lag_1']*df['item_price_item_lag_1']
    df['revenue_lag_shop_1'] = df['item_cnt_day_shop_lag_1']*df['item_price_shop_lag_1']
    
    return df
    
train = revenues(train)
train.to_pickle('train.pkl')
del df_sales_shop
del df_sales_item
del df_sales_item_cat
del df_sales_train
del df_sales_train_grouped
del df_sales_train_grouped_negative
del df_sales_train_grouped_positive
del legend_list
del train
train = pd.read_pickle('train.pkl')
X_train = train.drop(labels=['item_cnt_day','item_cnt_day_item','item_cnt_day_shop',
                             'item_price','item_price_item','item_price_shop'],axis=1)
y_train = train['item_cnt_day'].clip(0,20)
X_test = train[(train['date_block_num']>32)].drop(labels=['item_cnt_day','item_cnt_day_item','item_cnt_day_shop',
                             'item_price','item_price_item','item_price_shop'],axis=1)
y_test = train[(train['date_block_num']>32)]['item_cnt_day'].clip(0,20)


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
xgb = XGBRegressor(objective='reg:squarederror',
    n_estimators=1000,
    max_depth=10,
    reg_alpha=0.1,
    reg_lambda=2,
    eta=0.2,
    tree_method='gpu_hist')
xgb.fit(X_train,y_train,eval_metric="rmse",
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
        early_stopping_rounds = 10)
predxg = xgb.predict(X_test)
print(metrics.mean_squared_error(y_test,predxg))
coeff_df = pd.DataFrame(xgb.feature_importances_,X_train.columns,columns=['Coefficient'])

fig, ax = plt.subplots(1,1,figsize=(12,8))
coeff_df.sort_values(by='Coefficient',ascending=False).head(20).plot(kind='bar',ax=ax)
plt.xlabel('Features')
plt.title('Top 20 Features');
df_sales_test['date_block_num'] = 34
train_updated_columns = list()
for i in train.columns:
    train_updated_column = ''
    if i[-1].isnumeric():
        train_updated_column = '_'.join(i.split('_')[:-1]) +'_' + str(int(i.split('_')[-1])+1)
    elif i.find('item_cnt_day') > -1:
        train_updated_column = i + "_lag_1"
    elif i.find('item_price') > -1:
        train_updated_column = i + "_lag_1"
    else:
        train_updated_column = i
        
    
        
    if i[-1].isnumeric():
        if int(i.split('_')[-1]) < 12:
            train_updated_columns.append(train_updated_column)
    else:
        train_updated_columns.append(train_updated_column)
        
train['date_block_num'] +=1
train = train[train['date_block_num']==34]
train.drop(labels=['item_cnt_day_lag_12','item_price_lag_12','item_cnt_day_item_lag_12','item_price_item_lag_12',
                  'item_cnt_day_shop_lag_12','item_price_shop_lag_12'],axis=1,inplace=True)

train.columns = train_updated_columns
test = pd.DataFrame()
prev = 0
for i in range(0,df_sales_test.shape[0],10000):
    test = pd.concat([test,pd.merge(df_sales_test[prev:i], train, on=['date_block_num','shop_id','item_id'],how = 'left')],ignore_index=True)
    prev=i
test = pd.concat([test,pd.merge(df_sales_test[prev:], train, on=['date_block_num','shop_id','item_id'],how = 'left')],ignore_index=True)
Id = test['ID']
test['month'] = test['date_block_num'] % 12
test['days'] = test['month'].map(days).astype(np.int8)
test['Year'] = (test['date_block_num'] // 12 ) + 2013
test.fillna(value=0,inplace=True)
test.drop('ID',axis=1,inplace=True)
test = revenues(test)
pred_xg = xgb.predict(test[X_train.columns])
pred_xg = pred_xg.reshape(pred_xg.shape[0],).clip(0,20)
submission = pd.DataFrame({
        "ID": Id,
        "item_cnt_month": pred_xg
    })
submission.to_csv('submission.csv', index=False)