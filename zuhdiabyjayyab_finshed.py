import numpy as np 

import pandas as pd 

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

from xgboost import plot_importance

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import os

print(os.listdir("../input/competitive-data-science-predict-future-sales"))
start = '../input/competitive-data-science-predict-future-sales'

items  = pd.read_csv(start+'/items.csv')

train = pd.read_csv(start+'/sales_train.csv')

test = pd.read_csv(start+'/test.csv')

item_cat = pd.read_csv(start+'/item_categories.csv')

shops = pd.read_csv(start+'/shops.csv')

sample = pd.read_csv(start+'/sample_submission.csv')
def disc(data):

    '''

    discribe data with showing data's head, info, shape, duplicated data

    '''

    print("----------head----------")

    print(data.head(7))

    print("-----------info-----------")

    print(data.info())

    print("----------Missing value-----------")

    print(data.isna().sum())

    print("----------shape----------")

    print(data.shape)

    print('----------Number of duplicates----------')

    print(len(data[data.duplicated()]))

    

def graph_insight(data):

    '''

    Drow the columns belong to that typies: 'float64', 'int64', 'int8','int16', 'float16'

    '''

    print(set(data.dtypes.tolist()))

    df_num = data.select_dtypes(include = ['float64', 'int64', 'int8','int16', 'float16'])

    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);

    

def drop_dupl(data, subset):

    '''

    remove douplecated data

    '''

    print('Before drop shape:', data.shape)

    before = data.shape[0]

    data.drop_duplicates(subset,keep='first', inplace=True) 

    #subset is list where you have to put all column for duplicate check

    data.reset_index(drop=True, inplace=True)

    print('After drop shape:', data.shape)

    after = data.shape[0]

    print('Total Duplicate:', before-after)

    

def disc_ds(data):

    '''

    use duscrebtion function and graph function togather

    '''

    disc(data)

    graph_insight(data)

    

def pre(data):

    '''

    downcast data types and lable incoder

    '''

    for _ in [1,2] :

        for col in data.keys(): 

            if data[col].dtype in ['int64'] :

                data[col] = data[col].astype('int16')

            elif data[col].dtype in ['float64'] :

                data[col] = data[col].astype('float16')

            elif data[col].dtype in ['object'] :

                data[col] = LabelEncoder().fit_transform(data[col])
disc_ds(train)
train.tail()
plt.figure(figsize=(10,4))

plt.xlim(-100, 3000)

sns.boxplot(x=train.item_cnt_day)



plt.figure(figsize=(10,4))

plt.xlim(train.item_price.min(), train.item_price.max()*1.1)

sns.boxplot(x=train.item_price)
# -1 and 307980 looks like outliers, let's delete them

print('before train shape:', train.shape)

train = train[(train.item_price > 0) & (train.item_price < 300000)]

drop_dupl(train, train.keys())

print('after train shape:', train.shape)
trained = train.groupby(['shop_id', 'item_id','date_block_num']).item_cnt_day.count().reset_index(name='items_month')

trained = trained.sort_values(by=['date_block_num','item_id'])
disc_ds(test)
tested = test.copy().drop("ID" ,axis=1)

tested['date_block_num'] = 34 #for test month

tested['items_month'] = 0
con = [tested, trained]

train_test = pd.concat(con)
disc(shops)
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops['shop_center'] = shops['shop_name'].str.split(' ').map(lambda x: x[1])

shops = shops[['shop_id','city','shop_center']]

pre(shops)
disc(items)
items_icat = pd.merge(items,  item_cat, on=['item_category_id'], how='left')
disc(item_cat)
item_cat['type'] = item_cat['item_category_name'].str.split('-').map(lambda x: x[0].strip())

item_cat['subtype'] = item_cat['item_category_name'].str.split('-').map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

item_cat = item_cat[['item_category_id','type','subtype']]

pre(item_cat)

pre(items)
tr_te_sh = pd.merge(shops,train_test, on=['shop_id'], how='right', sort=False)
tr_te_sh_it_icat = pd.merge(tr_te_sh, items_icat, on=['item_id'], how='left', sort=False)
final = tr_te_sh_it_icat.copy()
pre(final)
X_train = final[final.date_block_num < 33].drop(['items_month'], axis=1)

Y_train = final[final.date_block_num < 33]['items_month']

X_valid = final[final.date_block_num == 33].drop(['items_month'], axis=1)

Y_valid = final[final.date_block_num == 33]['items_month']

X_test  = final[final.date_block_num == 34].drop(['items_month'], axis=1)
model = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)



model.fit(

    X_train, 

    Y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    verbose=True, 

    early_stopping_rounds = 10)
Y_pred = model.predict(X_valid).clip(0, 20)

Y_test = model.predict(X_test).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": Y_test

})

submission.to_csv('xgb_sub.csv', index=False)