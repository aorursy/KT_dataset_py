# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
items=pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
item_categories=pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
shops=pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

test=pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
sample_submission=pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
sales_train=pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
items.info()
items.head()
items.describe()
item_categories.info()
item_categories.head()
item_categories.describe()
shops.info()
shops.head()
shops.describe()
test.info()
test.head()
test.describe()
sample_submission.info()
sample_submission.head()
sample_submission.describe()
sales_train.info()
sales_train.head()
sales_train.describe()
sales_train.item_price.hist()
sales_train.item_price.value_counts()
sns.boxplot(sales_train.item_price, data=sales_train)
sales_train.item_price.max() #на графике видно, что максимаьной значение значительно отличается от остальных, похоже на выброс 
sales_train[sales_train.item_price==sales_train.item_price.max()]
#узнаем, что за item с такой большой ценой
items[items['item_id']==6066]
#Radmin 3 - 522 лиц это софт, не думаю, что он такой дорогой, поэтому проверим были ли события с другой ценой 
sales_train[sales_train.item_id==6066]
sales_train[sales_train['item_price']<60000].item_price.hist()
sales_train[sales_train['item_price']<10000].item_price.hist()
sales_train[sales_train['item_price']<5000].item_price.hist()
sales_train[sales_train['item_price']<1000].item_price.hist()
item_categories["meta_category"]=item_categories.item_category_name.apply(lambda x:x.split(" - ")[0])
print(item_categories.meta_category.head(5))
print(item_categories["meta_category"].unique())
print(len(item_categories["meta_category"].unique())) #т.е. все товары можно разделить на 20 категорий
shops.head() #можно заметить, что первое слово в названии это город в котором располагается магазин
shops['town'] = shops['shop_name'].replace('[^а-яА-Я0-9]', ' ', regex = True)
shops["town"] = shops["town"].apply(lambda x:x.split()[0])
print(shops["town"].unique())
print(len(shops["town"].unique())) #т.е. все магазины можно разделить на 31 категорию
shops['town'][shops["town"]=='Н'] = 'НижнийНовгород'
shops['town'][shops["town"]=='Выездная'] = 'ВыезднаяТорговля'
print(shops["town"].unique())
print(len(shops["town"].unique())) #т.е. все магазины можно разделить на 31 категорию
shops.to_csv('final_shops.csv',index=False) #cохраняем изменения в новый файл
sales_train['day'] = pd.to_datetime(sales_train['date'], format = '%d.%m.%Y').dt.day
sales_train['month'] = pd.to_datetime(sales_train['date'], format = '%d.%m.%Y').dt.month
sales_train['year'] = pd.to_datetime(sales_train['date'], format = '%d.%m.%Y').dt.year
sales_train['weekday'] = pd.to_datetime(sales_train['date'], format = '%d.%m.%Y').dt.dayofweek
print(sales_train['weekday'].unique())
sales_train=sales_train.merge(items, how='left')
sales_train=sales_train.merge(item_categories,how="left")
sales_train=sales_train.merge(shops,how="left")
sales_train.drop("item_name",axis=1,inplace=True)
sales_train.drop("shop_name",axis=1,inplace=True)
sales_train.drop("item_category_name",axis=1,inplace=True)
sales_train.head()
sales_train["revenue"]=sales_train.item_cnt_day * sales_train.item_price #выручка от продажи товара
sales_train.groupby("date_block_num").sum()["revenue"].plot() #выручка в зависимости от месяца, хорошо видны два новогодних пика
sales_train.groupby("date_block_num").sum()["item_cnt_day"].plot(kind='bar') #колличество проданного товара в месяц
sales_train.groupby("weekday").sum()["revenue"].plot() 
sales_train.groupby("weekday").sum()["item_cnt_day"].plot(kind='bar') 
fig, axes = plt.subplots(1, 1, figsize=(10, 10), sharey=True)
axes = sales_train.groupby("shop_id").sum()["revenue"].plot(kind='bar')
print(shops[shops['shop_id']==31])
print(shops[shops['shop_id']==25])
print(shops[shops['town']=='Интернет']) #самые прибыльные магазины + интернет магазин
# колличество товаров проданных в интернет магазине
sales_train[sales_train["town"]=='Интернет'].groupby("date_block_num").sum()["item_cnt_day"].plot(kind='bar')
sales_train.groupby("date_block_num").mean().item_price.plot() #средняя цена проданных товаров в месяц (растет)
sales_train.groupby("date_block_num").mean().revenue.plot() #средняя выручка от каждого проданного товара в месяц (растет)
sales_train.groupby(["date_block_num","meta_category"]).sum()["revenue"].unstack().plot(figsize=(10,10))
sales_train.groupby(["date_block_num","town"]).sum()["revenue"].unstack().plot(figsize=(10,10)) # наибольшая прибыль приходится на Москву
shop_life = pd.DataFrame(columns=["shop_id","Start", "Stop"])
shop_life['shop_id'] = np.arange(60) 
shop_life['Start'] = sales_train.groupby("shop_id")["date_block_num"].min()
shop_life['Stop'] = sales_train.groupby("shop_id")["date_block_num"].max()
shop_life
print(shops[shops['shop_id']==10])
print(shops[shops['shop_id']==11])
sales_train[(sales_train['shop_id']==10) & (sales_train['date_block_num']==25)] 
sales_train['shop_id'][(sales_train['shop_id']==11)]=10
shops.drop(shops[shops['shop_id']==11].index,axis=0, inplace=True)
print(shops[shops['shop_id']==39])
print(shops[shops['shop_id']==40])
close_shop_id = shop_life['shop_id'][(shop_life['Stop']<33) & (shop_life['shop_id']!=11)] 
print(close_shop_id.unique())
print(len(close_shop_id.unique())) #список id 16-ти закрытых магазинов 
print(test['shop_id'].unique())
print(len(test['shop_id'].unique())) #список 42 магазинов из тестовых данных 
print (len(set(test['shop_id']) | set(close_shop_id))) # колличество закрытых магазинов + магазины из тестового набора
print (len(set(test['shop_id']) & set(close_shop_id))) # проверка есть ли закрытые магазины в тестовом наборе (таких нет)
print(set(shops['shop_id']) - (set(test['shop_id']) | set(close_shop_id))) #список открытых магазинов, которых нет в тестовых данных
shops[(shops['shop_id']==9) | (shops['shop_id']==20)] 
sales_train[(sales_train["shop_id"]==9) | (sales_train["shop_id"]==20)].groupby(["shop_id","date_block_num"]).sum()["revenue"]
print(sales_train['month'][(sales_train['date_block_num']==9) | (sales_train['date_block_num']==21) | (sales_train['date_block_num']==27)].unique())
print(sales_train['month'][(sales_train['date_block_num']==21) | (sales_train['date_block_num']==33)].unique())
print(len(sales_train[sales_train.date_block_num==33]["item_id"].unique())) #число items в продаже в последний месяц
print(len(sales_train["item_id"].unique())) #общее число items
print(len(test["item_id"].unique())) #число items для которых нужно сделать предсказание
print(len(set(sales_train[sales_train.date_block_num==33]["item_id"]) & set(test["item_id"]))) #число items которые продаются в последний месяц и есть в тесте
print(len(set(test["item_id"]) - set(sales_train[sales_train.date_block_num==33]["item_id"]))) #число Item котоые есть в тесте, но не продатся в 33 месяц
print(len(set(sales_train["item_id"]) & set(test["item_id"]))) # число items которые есть в traine и для которых надо сделать прогноз
print(len(set(test["item_id"]) - set(sales_train["item_id"]))) # число items, которых нет в traine, но есть в тесте
new_item_list = list(set(test["item_id"]) - set(sales_train["item_id"]))
shop_new_item = list()
cat_new_item = list()
for id_item in new_item_list:
    shop_new_item.append(test[test['item_id']==id_item]['shop_id'].values)
    cat_new_item.append(items[items['item_id']==id_item]['item_category_id'].values)
cat_new_item = list(set(np.array(cat_new_item).reshape(363)))
shop_new_item = list(set(np.array(shop_new_item).reshape(15246)))
print(len(shop_new_item)) #магазины в которых встречаются новые товары
print(len(cat_new_item)) # id категории новых товаров
cat_name_new_item = list()
for id_cat in cat_new_item:
    cat_name_new_item.append(item_categories[item_categories['item_category_id']==id_cat]['meta_category'].values)
cat_name_new_item = list(set(np.array(cat_name_new_item).reshape(39)))
print(cat_name_new_item)
print(len(cat_name_new_item)) #мета категории новых товаров
plt.scatter(sales_train['item_category_id'],sales_train['item_id'], s=0.1)
sales_train.columns
sales_train['item_cnt_day'].hist()
sns.boxplot(sales_train.item_cnt_day, data=sales_train)
sales_train['item_cnt_day'].value_counts()
print(sales_train['item_cnt_day'].max())
print(sales_train['item_cnt_day'][sales_train['item_cnt_day']<2169].max())
print(sales_train[sales_train['item_cnt_day']==2169]) # 2169 раз доставили товар за день? Сомнительно
print(items[items['item_id']==11373])
print(shops[shops['shop_id']==12])
print(sales_train[sales_train['item_cnt_day']==1000])
print(items[items['item_id']==20949])
print(shops[shops['shop_id']==12])
interim = sales_train[sales_train["date_block_num"]==33].groupby(["shop_id", "item_id"],as_index=False).sum()[["shop_id","item_id","item_cnt_day"]]
interim["item_cnt_day"].clip(0,20,inplace=True)
print(interim['item_cnt_day'].min())
print(interim['item_cnt_day'].max())
interim.columns
print(test.columns)
print(test.shape)
interim_pred = pd.merge(test, interim, how='left', left_on=["shop_id","item_id"], right_on = ["shop_id","item_id"])
print(interim_pred.shape)
interim_pred=interim_pred[["ID","item_cnt_day"]]
interim_pred.columns=["ID","item_cnt_month"]
interim_pred.fillna(0,inplace=True)
interim_pred['item_cnt_month'].value_counts() #очень много нулей
interim_pred.to_csv('submission2.csv',index=False) #score 1.16
sales_train.head(2)
df=pd.DataFrame()
df=sales_train.groupby(["date_block_num","shop_id","item_id","month","item_price","item_category_id","meta_category","town"],as_index=False).sum()[["date_block_num","shop_id","item_id","month","item_price","item_category_id","meta_category","town","item_cnt_day"]]
df["item_cnt_day"].clip(0,20,inplace=True)
df=df.rename(columns = {'item_cnt_day':'item_cnt_month'})
df.head()
df["price_category"]=np.nan
df["price_category"][(df["item_price"]>=0)&(df["item_price"]<=100)]=0
df["price_category"][(df["item_price"]>100)&(df["item_price"]<=200)]=1
df["price_category"][(df["item_price"]>200)&(df["item_price"]<=400)]=2
df["price_category"][(df["item_price"]>400)&(df["item_price"]<=750)]=3
df["price_category"][(df["item_price"]>750)&(df["item_price"]<=1000)]=4
df["price_category"][(df["item_price"]>1000)&(df["item_price"]<=2000)]=5
df["price_category"][(df["item_price"]>2000)&(df["item_price"]<=3000)]=6
df["price_category"][df["item_price"]>3000]=7
sns.countplot(df.price_category)
from sklearn import preprocessing
le_met_cat = preprocessing.LabelEncoder()
le_met_cat.fit(df.meta_category)
df["meta_category"]=le_met_cat.transform(df.meta_category)

le_town = preprocessing.LabelEncoder()
le_town.fit(df.town)
df["town"]=le_town.transform(df.town)

df.head()
X_train=df.drop("item_cnt_month", axis=1)
y_train=df["item_cnt_month"]
X_train.fillna(0, inplace=True)
from sklearn.linear_model import LinearRegression

linmodel=LinearRegression()
linmodel.fit(X_train,y_train)
predictions=linmodel.predict(X_train)
from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(y_train,predictions)))
print(linmodel.score(X_train, y_train))
linmodel.coef_
X_train.columns
key_met_cat = np.array(range(0,20))
for i, j in zip(key_met_cat,le_met_cat.classes_):
    print(i, j)
key_met_cat_dict ={}
for i, j in zip(key_met_cat,le_met_cat.classes_):
    key_met_cat_dict[j]=i
for cat in cat_name_new_item:
    print(key_met_cat_dict[cat], cat)
plt.scatter(df['meta_category'], df['price_category'])
fig, axes = plt.subplots(5,4, figsize=(40, 40), sharey=True)
for i in range(0,20):
    sns.countplot(x=df[df['meta_category']==i]['price_category'], data=df, ax=axes[i // 4, i % 4])
    axes[i // 4, i % 4].set_title('categ №'+ str(i))
meta_price = {0:0,1:6,2:5,3:2,4:7,5:5,6:1,7:3,8:2,9:5,10:3,11:1,12:2,13:2,14:3,15:5,16:5,17:1,18:0,19:1} #meta price for mew items
for cat in cat_name_new_item:
    print('CAT: ',key_met_cat_dict[cat],  cat, '   PRICE: ', meta_price[key_met_cat_dict[cat]])
df2 = df[['town','meta_category','item_price']].groupby(['town','meta_category']).mean()
df2["price_category"]=np.nan
df2["price_category"][(df2["item_price"]>=0)&(df2["item_price"]<=100)]=0
df2["price_category"][(df2["item_price"]>100)&(df2["item_price"]<=200)]=1
df2["price_category"][(df2["item_price"]>200)&(df2["item_price"]<=400)]=2
df2["price_category"][(df2["item_price"]>400)&(df2["item_price"]<=750)]=3
df2["price_category"][(df2["item_price"]>750)&(df2["item_price"]<=1000)]=4
df2["price_category"][(df2["item_price"]>1000)&(df2["item_price"]<=2000)]=5
df2["price_category"][(df2["item_price"]>2000)&(df2["item_price"]<=3000)]=6
df2["price_category"][df2["item_price"]>3000]=7

df2 = df2.drop('item_price',axis=1).unstack()
df2.head()
fig, axes = plt.subplots(5, 4, figsize=(40, 40), sharey=True)
for i in range(0,20):
    ax = axes[i // 4, i % 4]
    ax.scatter(np.array(range(0,31)),df2[df2.columns[i]])
    ax.set_title('categ №'+ str(i))
test2 = pd.merge(test,items[['item_id','item_category_id']], on='item_id', how='left')
test3 = pd.merge(test2,shops[['shop_id','town']], on='shop_id', how='left')
test4 = pd.merge(test3,item_categories[['item_category_id','meta_category']], on='item_category_id', how='left')
test4['month'] = pd.Series([11]*214200)
test4['date_block_num'] = pd.Series([34]*214200)
test4["meta_category"]=le_met_cat.transform(test4.meta_category)
test4["town"]=le_town.transform(test4.town)
test4.head()
test_part_list = list() #test file with item_id & shop_id in df
test_temp = test4
test_temp['price_category'] = test_temp['shop_id']
sum_id = 0
for month in range(33,0,-1):
    print('month:', month)
    test_all = pd.merge(test_temp.drop('price_category', axis=1), df[df['date_block_num']==month][['shop_id','item_id','item_price']].groupby(['shop_id','item_id']).mean(), how='left', on=['item_id','shop_id'])
    
    test_all["price_category"]=np.nan
    test_all["price_category"][(test_all["item_price"]>=0)&(test_all["item_price"]<=100)]=0
    test_all["price_category"][(test_all["item_price"]>100)&(test_all["item_price"]<=200)]=1
    test_all["price_category"][(test_all["item_price"]>200)&(test_all["item_price"]<=400)]=2
    test_all["price_category"][(test_all["item_price"]>400)&(test_all["item_price"]<=750)]=3
    test_all["price_category"][(test_all["item_price"]>750)&(test_all["item_price"]<=1000)]=4
    test_all["price_category"][(test_all["item_price"]>1000)&(test_all["item_price"]<=2000)]=5
    test_all["price_category"][(test_all["item_price"]>2000)&(test_all["item_price"]<=3000)]=6
    test_all["price_category"][test_all["item_price"]>3000]=7
    test_all.drop('item_price',axis=1, inplace=True)
    
    s_temp = test_all['price_category'].count()
    sum_id += s_temp
    print('count_id_item:', s_temp)
    test_part_list.append(test_all[test_all['price_category'].notnull()])
    test_temp = test_all[test_all['price_category'].isnull()]
print('sum_determ:', sum_id)
print('sum_not_determ:', test_temp.shape[0])
print('sum:', sum_id + test_temp.shape[0])
test_temp_part1 = test_temp
test_part_list_2 = list() #test file part2 with item_id in df
test_temp = test_temp_part1
sum_id = 0
for month in range(33,0,-1):
    print('month:', month)
    test_all = pd.merge(test_temp.drop('price_category', axis=1), df[df['date_block_num']==month][['item_id','price_category','town']].groupby(["item_id",'town']).mean(), how='left', on=['item_id','town'])
    s_temp = test_all['price_category'].count()
    sum_id += s_temp
    print('count_id_item:', s_temp)
    test_part_list_2.append(test_all[test_all['price_category'].notnull()])
    test_temp = test_all[test_all['price_category'].isnull()]
print('sum_determ:', sum_id)
print('sum_not_determ:', test_temp.shape[0])
print('sum:', sum_id + test_temp.shape[0])
test_temp_part2 = test_temp
test_part_list_3 = list() #test file part2 with item_id in df
test_temp = test_temp_part2
sum_id = 0
for month in range(33,0,-1):
    print('month:', month)
    test_all = pd.merge(test_temp.drop('price_category', axis=1), df[df['date_block_num']==month][['item_id','price_category']].groupby(["item_id"]).mean(), how='left', on=['item_id'])
    s_temp = test_all['price_category'].count()
    sum_id += s_temp
    print('count_id_item:', s_temp)
    test_part_list_3.append(test_all[test_all['price_category'].notnull()])
    test_temp = test_all[test_all['price_category'].isnull()]
print('sum_determ:', sum_id)
print('sum_not_determ:', test_temp.shape[0])
print('sum:', sum_id + test_temp.shape[0])
test_temp_part3 = test_temp
test_temp_part3.info()
print(len(test_temp_part3['item_id'].unique()))
test_part_list_4 = list() #test file part2 with item_id in df
test_temp = test_temp_part3
sum_id = 0
for month in range(33,1,-1):
    print('month:', month)
    test_all = pd.merge(test_temp.drop('price_category', axis=1), df[df['date_block_num']==month][['meta_category','item_price']].groupby(["meta_category"]).mean(), how='left', on=["meta_category"])
    
    test_all["price_category"]=np.nan
    test_all["price_category"][(test_all["item_price"]>=0)&(test_all["item_price"]<=100)]=0
    test_all["price_category"][(test_all["item_price"]>100)&(test_all["item_price"]<=200)]=1
    test_all["price_category"][(test_all["item_price"]>200)&(test_all["item_price"]<=400)]=2
    test_all["price_category"][(test_all["item_price"]>400)&(test_all["item_price"]<=750)]=3
    test_all["price_category"][(test_all["item_price"]>750)&(test_all["item_price"]<=1000)]=4
    test_all["price_category"][(test_all["item_price"]>1000)&(test_all["item_price"]<=2000)]=5
    test_all["price_category"][(test_all["item_price"]>2000)&(test_all["item_price"]<=3000)]=6
    test_all["price_category"][test_all["item_price"]>3000]=7
    test_all.drop('item_price',axis=1, inplace=True)
    s_temp = test_all['price_category'].count()
    sum_id += s_temp
    print('count_id_item:', s_temp)
    test_part_list_4.append(test_all[test_all['price_category'].notnull()])
    test_temp = test_all[test_all['price_category'].isnull()]
print('sum_determ:', sum_id)
print('sum_not_determ:', test_temp.shape[0])
print('sum:', sum_id + test_temp.shape[0])
test_temp_part4 = test_temp
test_temp_part4.info()
print(len(test_temp_part4['item_id'].unique()))
test_part_list_final = pd.concat([t for t in test_part_list])
test_part_list_2_final = pd.concat([t for t in test_part_list_2])
test_part_list_3_final = pd.concat([t for t in test_part_list_3])
test_part_list_4_final = pd.concat([t for t in test_part_list_4])
print(test_part_list_final.shape[0])
print(test_part_list_2_final.shape[0])
print(test_part_list_3_final.shape[0])
print(test_part_list_4_final.shape[0])
test_final = pd.concat([test_part_list_final,test_part_list_2_final,test_part_list_3_final,test_part_list_4_final])
test_final.info()
test_final = test_final.sort_values(by=['ID'])
test_final.index = test_final['ID']
test_final.head()
df.loc[df.index==290279, df.columns =='price_category'] = 5 #устраним аномальное значение цены
test_df = df[df['date_block_num']==33].drop('item_cnt_month',axis=1)
y_test_df = df[1705824:]['item_cnt_month']
train_df = df[:1705824].drop('item_cnt_month',axis=1)
y_train_df = df[:1705824]['item_cnt_month']
print(train_df.shape)
print(y_train_df.shape)
print(test_df.shape)
print(y_test_df.shape)
train_df.head()
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(df.corr(), square=True, annot=True, cbar=False)
train_df.drop('item_category_id', axis=1, inplace=True)
test_df.drop('item_category_id', axis=1, inplace=True)

train_df.drop('shop_id', axis=1, inplace=True)
test_df.drop('shop_id', axis=1, inplace=True)

train_df.drop('item_id', axis=1, inplace=True)
test_df.drop('item_id', axis=1, inplace=True)

train_df.drop('item_price', axis=1, inplace=True)
test_df.drop('item_price', axis=1, inplace=True)
train_df.columns
from sklearn.preprocessing import OneHotEncoder
enc_month = OneHotEncoder(handle_unknown='ignore')
enc_month.fit(df['month'].values.reshape(-1, 1))
month_categ_train = enc_month.transform(train_df['month'].values.reshape(-1, 1))
month_categ_test = enc_month.transform(test_df['month'].values.reshape(-1, 1))

enc_town = OneHotEncoder(handle_unknown='ignore')
enc_town.fit(df['town'].values.reshape(-1, 1))
town_categ_train = enc_town.fit_transform(train_df['town'].values.reshape(-1, 1))
town_categ_test = enc_town.transform(test_df['town'].values.reshape(-1, 1))

enc_meta_category = OneHotEncoder(handle_unknown='ignore')
enc_meta_category.fit(df['meta_category'].values.reshape(-1, 1))
meta_category_categ_train = enc_meta_category.fit_transform(train_df['meta_category'].values.reshape(-1, 1))
meta_category_categ_test = enc_meta_category.transform(test_df['meta_category'].values.reshape(-1, 1))
from scipy.sparse import hstack
X_train = hstack([month_categ_train, town_categ_train, meta_category_categ_train, train_df['date_block_num'].values.reshape(-1, 1), 
                  train_df['price_category'].values.reshape(-1, 1)])
X_test = hstack([month_categ_test, town_categ_test, meta_category_categ_test, test_df['date_block_num'].values.reshape(-1, 1), 
                  test_df['price_category'].values.reshape(-1, 1)])
print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LinearRegression
model= LinearRegression(fit_intercept=False)
model.fit(X=X_train, y=y_train_df)
y_pred = model.predict(X_test)
y_pred_r=[round(v) for v in y_pred]
print(set(y_pred_r))
len(set(y_pred_r))
from sklearn.metrics import confusion_matrix
fig, ax = plt.subplots(figsize=(20,20))
mat = confusion_matrix(y_pred_r, y_test_df)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
print(np.sqrt(mean_squared_error(y_test_df,y_pred)))
print(model.score(X_train, y_train_df))
print(model.coef_)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, class_weight='balanced',verbose = True)
#model.fit(X=X_train.toarray(), y=y_train_df)
#y_pred_rf = model.predict(X_test.toarray())
print(set(y_pred_rf))
print(len(set(y_pred_rf)))
print(np.sqrt(mean_squared_error(y_test_df,y_pred_rf)))
from sklearn.metrics import confusion_matrix
fig, ax = plt.subplots(figsize=(20,20))
mat = confusion_matrix(y_pred_rf,y_test_df)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
model.feature_importances_
train = df.pivot_table(index=['shop_id','item_id'], columns='date_block_num', values='item_cnt_month',aggfunc='sum').fillna(0.0)
train.head()
train=train.reset_index()
train.head()
final_train=train.merge(df[["shop_id","item_id","item_price","item_category_id","meta_category","town","price_category"]])
final_train.head()
model= LinearRegression(fit_intercept=False)
model.fit(X=final_train.drop(33,axis=1), y=final_train[33])
y_pred = model.predict(final_train.drop(33,axis=1))
y_pred_r=[round(v) for v in y_pred]
y_pred_r = pd.Series(y_pred_r)
y_pred_r.clip(0,20,inplace=True)
print(set(y_pred_r.values))
len(set(y_pred_r.values))
ft = final_train[33].clip(0,20)
from sklearn.metrics import confusion_matrix
fig, ax = plt.subplots(figsize=(20,20))
mat = confusion_matrix(y_pred_r, ft)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
print(np.sqrt(mean_squared_error(final_train[33],predictions)))
print(model.score(final_train.drop(33,axis=1),final_train[33]))
test_final.head()
df['ID'] = -1
test_final_temp = pd.concat([df,test_final])
test_final_temp_2 = test_final_temp.pivot_table(index=['shop_id','item_id'], columns='date_block_num', values='item_cnt_month',aggfunc='sum').fillna(0.0)
test_final_temp_2=test_final_temp_2.reset_index()
test_final_temp_3=test_final_temp_2.merge(test_final_temp[["shop_id","item_id","item_price","item_category_id","meta_category","town","price_category",'ID']])
test_final_temp_3 = test_final_temp_3[test_final_temp_3['ID']!=-1]
test_final_temp_3 = test_final_temp_3.sort_values(by=['ID'])
test_final_temp_3.drop('item_price',axis=1, inplace=True)
test_final_temp_3.index = test_final_temp_3['ID']

print(test_final_temp_3.shape)
print(test_final.shape)
test_final_all = test_final_temp_3.drop(34,axis=1)
test_final_all.head()
test_final_all.drop(['item_category_id','town','price_category','ID'],axis=1, inplace=True)
model= LinearRegression()
model.fit(X=test_final_all.drop(33,axis=1), y=test_final_all[33])
y_pred = model.predict(test_final_all.drop(33,axis=1))
y_pred_r=[round(v) for v in y_pred]
y_pred_r = pd.Series(y_pred_r)
y_pred_r.clip(0,20,inplace=True)
ft = test_final_all[33].clip(0,20)
print(set(y_pred_r.values))
len(set(y_pred_r.values))
from sklearn.metrics import confusion_matrix
fig, ax = plt.subplots(figsize=(20,20))
mat = confusion_matrix(y_pred_r, ft)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
print(np.sqrt(mean_squared_error(test_final_all[33],y_pred)))
print(model.score(test_final_all.drop(33,axis=1),test_final_all[33]))
model = RandomForestClassifier(n_estimators=100, verbose = True)
model.fit(X=test_final_all.drop(33,axis=1), y=test_final_all[33])
y_pred = model.predict(test_final_all.drop(33,axis=1))
y_pred_r=[round(v) for v in y_pred]
y_pred_r = pd.Series(y_pred_r)
y_pred_r.clip(0,20,inplace=True)
ft = test_final_all[33].clip(0,20)
print(set(y_pred_r.values))
len(set(y_pred_r.values))
from sklearn.metrics import confusion_matrix
fig, ax = plt.subplots(figsize=(20,20))
mat = confusion_matrix(y_pred_r, ft)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
print(np.sqrt(mean_squared_error(test_final_all[33],y_pred)))
print(model.feature_importances_)
y = pd.Series(y_pred)
y.clip(0,20,inplace=True)
sample_submission['item_cnt_month'] = y
sample_submission['item_cnt_month'].value_counts() #очень много нулей
sample_submission.to_csv('submission2.csv',index=False) #