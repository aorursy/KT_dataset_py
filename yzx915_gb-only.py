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
# read data and import packages

sales = pd.read_csv("../input/sales_train.csv")

item_cat = pd.read_csv("../input/item_categories.csv")

item = pd.read_csv("../input/items.csv")

shops = pd.read_csv("../input/shops.csv")

test = pd.read_csv("../input/test.csv")



from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA

from sklearn.linear_model import Ridge

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt

import seaborn as sns
sales.date = pd.to_datetime(sales.date, format='%d.%m.%Y')



group = sales.groupby(['shop_id','item_id','date_block_num']).item_cnt_day.sum()



train = group.reset_index()

train = train.rename(columns={'item_cnt_day':'item_cnt_month'}).drop('date_block_num',axis=1)



test.item_id = test.item_id.fillna(test.item_id.median())
# number of items for month per shop

x = train.groupby('shop_id').item_cnt_month.sum().reset_index()



plt.figure(figsize=(30,4))

ax = sns.barplot(x.shop_id, x.item_cnt_month)

plt.title("Items per shop")

plt.ylabel('number of items for month', fontsize=12)

plt.xlabel('shop id', fontsize=12)

plt.show()
shops['city'] = shops['shop_name'].str.split().map(lambda x: x[0])

shops['city_id'] = LabelEncoder().fit_transform(shops['city'])



# add shops city_id feature

sci = pd.Series(data=shops.city_id,index=shops.shop_id)

train['city_id'] = train.shop_id.map(sci)

test['city_id'] = test.shop_id.map(sci)





# add item_category_id feature

ici = pd.Series(data=item.item_category_id,index=item.item_id)

train['item_category_id'] = train.item_id.map(ici)

test['item_category_id'] = test.item_id.map(ici)

train.head()
# expanding mean encoding

for col in ['shop_id','item_id','item_category_id','city_id']:

    cumsum = train.groupby(col).item_cnt_month.cumsum() - train.item_cnt_month

    cumcnt = train.groupby(col).cumcount()

    means = cumsum/cumcnt

    global_mean = train.item_cnt_month.mean()

    train[col+'_mean'] = means.fillna(global_mean)

    test[col+'_mean'] = test[col].map(means).fillna(global_mean)

train.head()
# prepare train and test

X_train = train.drop('item_cnt_month',axis=1)

y_train = train.item_cnt_month

X_test = test.drop('ID',axis=1)
gb = GradientBoostingRegressor(learning_rate=0.01,n_estimators=2000,min_samples_split=30)

gb.fit(X_train,y_train)

gb_result = gb.predict(X_test)
result = (gb_result).clip(0, 20)
submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": result

})

submission.to_csv('submission.csv', index=False)