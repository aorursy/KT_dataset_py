import numpy as np

import pandas as pd



from sklearn import *

import nltk, datetime

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

item_cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')



train.head()

train.shape
test.shape
# date sutununu, datetime formatina cevir.

# daha sonra date sutunundan month alinacak.

train['date'] = pd.to_datetime(train['date'], format = '%d.%m.%Y')

# date time'den aylari al.

train['month'] = train['date'].dt.month

# date time'den yillari al.

train['year'] = train['date'].dt.year



# aylik tahmin yapilacagi icin, ilgili df'lere parse edildi. daha sonra bu sutun, drop edilir.

train = train.drop(['date', 'item_price'], axis = 1)
# her bir item'a ait gunluk satislari gruplayarak aylık satis sayilarini elde ediyoruz.

train = train.groupby([column for column in train.columns if column not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()

# gunluk satisi yapilan item sayaci, aya gore gruplandigindan sutun ismi degistirildi.

train = train.rename(columns={'item_cnt_day' : 'item_count_monthly'})

train.head(10)
# itemlerin aylik satis adetlerine gore ortalamasinin alinmasi

shop_item_ort = train[['shop_id', 'item_id', 'item_count_monthly']].groupby(['shop_id', 'item_id'], as_index=False)[['item_count_monthly']].mean()

#aylik satıs sayaci sutunu, aylik ortalama satis adedi olacak sekilde degisir (sutun adi)

shop_item_ort = shop_item_ort.rename(columns={'item_count_monthly' : 'item_count_monthly_mean'})
# Elde ettigimiz ortalama ozelligini (shop_item_ort), train verimize ekliyoruz.

train = pd.merge(train, shop_item_ort, how='left', on=['shop_id', 'item_id'])
# goruldugu uzere, olusturdugumuz item_count_monthly_mean sutunu, train verimize eklendi.

train.head(9)
# veri setindeki en son ay icerisinde satisi yapilan urunleri aliyoruz. shop_son_ay = shop_prev_month

# 2013-2015 arasi satis bilgileri mevcuttur. 2013 ocak'ta date_block_num -> 0'dır. en son 2015 Ekim icin ilgili deger 33'tur.

shop_son_ay = train[train['date_block_num'] == 33][['shop_id', 'item_id', 'item_count_monthly']]

shop_son_ay = shop_son_ay.rename(columns={'item_count_monthly' : 'item_count_son_ay'})

shop_son_ay.head()
# elde edilen son ay satis adetlerini de train verimize ekliyoruz.

train = pd.merge(train, shop_son_ay, how='left', on=['shop_id', 'item_id']).fillna(0.)

# train verisine item ozelliklerini ekliyoruz.

train = pd.merge(train, items, how='left', on='item_id')

# train verisine item kategorilerini ekliyoruz.

train = pd.merge(train, item_cats, how='left', on='item_category_id')

# train verisine dukkanlari (shop) ekliyoruz.

train = pd.merge(train, shops, how='left', on='shop_id')



# random forest ile predict yapilacagindan, train verisine tum featureler eklenmektedir.

# RF, tum bu featurelerden random secim yaparak egitimi gerceklestirecek

train.head()
# Test verisinin ayarlanmasi

# Uygulama icerisinde, 2015 Kasım'daki aylik satislari predict etmek istiyoruz.

# 2015 kasim bilgilerini ekliyoruz.

# train data'da 2015 ekimin date_block_number sayısı 33 oldugundan, 2015 kasim icin date_block_number'a 34 set ettik.

test['month'] = 11

test['year'] = 2015

test['date_block_num'] = 34

# ayrica RF ile train ve prediction islemleri yapilacagindan

# train verisine ekledigimiz tum ozellikleri test veri setine de ekleyecegiz (aylik ortalama satis, shop ve item bilgileri).



# Elde edilen ortalama ozelligini (shop_item_ort), test verimize ekliyoruz.

test = pd.merge(test, shop_item_ort, how='left', on=['shop_id', 'item_id']).fillna(0.)

# elde edilen son ay satis adetlerini de train verimize ekliyoruz.

test = pd.merge(test, shop_son_ay, how='left', on=['shop_id', 'item_id']).fillna(0.)

# test verisine item ozelliklerini ekliyoruz.

test = pd.merge(test, items, how='left', on='item_id')

# test verisine item kategorilerini ekliyoruz.

test = pd.merge(test, item_cats, how='left', on='item_category_id')

# test verisine dukkanlari (shop) ekliyoruz.

test = pd.merge(test, shops, how='left', on='shop_id')

# predict edilecek aylik satis sayisi sutunu degerlerini 0 olarak init. ediyoruz.

test['item_count_monthly'] = 0.



test.head()
# RF Modeli kurulmadan once label encoding islemi

for column in ['shop_name', 'item_name', 'item_category_name']:

    label = preprocessing.LabelEncoder()

    label.fit(list(train[column].unique()) + list(test[column].unique()))

    train[column] = label.transform(train[column].astype(str))

    test[column] = label.transform(test[column].astype(str))

    print(column)
# Random Forest Alg. (RF) ile train ve prediction islemleri

column = [col for col in train.columns if col not in ['item_count_monthly']]

x1 = train[train['date_block_num'] < 33]

y1 = np.log1p(x1['item_count_monthly'].clip(0., 30.))

x1 = x1[column]



x2 = train[train['date_block_num'] == 33]

y2 = np.log1p(x2['item_count_monthly'].clip(0., 30.))

x2 = x2[column]



# RF bir ensemble modeldir. ensmble modeller, çoklu ogrenme algoritmalari kullanarak daha iyi prediction yapmayi amaclar.

# çoklu ogrenme alg. kullanarak daha iyi bir model egitmeyi amaclamaktadir.

rf_model = ensemble.ExtraTreesRegressor(n_estimators=25, n_jobs=-1, max_depth=25, random_state=18)

rf_model.fit(x1, y1)



print('RMSE degeri: ', np.sqrt(metrics.mean_squared_error(y2.clip(0., 30.), rf_model.predict(x2).clip(0., 30.))))



print(test.columns.tolist())
# modeli train verisi icin fit edelim ve comp.ta istenen predictionlari hesaplayalim.

rf_model.fit(train[column], train['item_count_monthly'].clip(0., 30.))

# test verisinde daha oncesinde 0 olarak init ettigimiz kasım 2015 aylik satislari icin prediction islemi yapiyoruz.

test['item_count_monthly'] = rf_model.predict(test[column]).clip(0., 30.)

test['item_count_monthly'] = np.expm1(test['item_count_monthly'])



# predict edilen aylik satis adedini csv'ye yaziyoruz.

test[['ID', 'item_count_monthly']].to_csv('submission.csv', index=False)