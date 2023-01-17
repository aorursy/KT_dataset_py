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
import matplotlib.pyplot as plt
## set seed

import random

random.seed(1973)
## import dataset and explore

nyc_bnb = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

## เนื่องจากข้อมูลขนาดใหญ่ ตัด feature ออกหมดจนเหลือแค่ columns เดียวแล้ว memory ในการรันยังไม่สามารถรับได้จึงขอทำการ sampling ข้อมูลออกมาครับ

nyc_bnb = nyc_bnb.sample(4800)

nrows, ncols = nyc_bnb.shape

nyc_bnb.info()

nyc_bnb.head()
## เลือก columns สำหรับแสดง histogram

cols_is_num = nyc_bnb.select_dtypes(include = ['float64','int64']).columns.difference(['id','host_id'])



## ดู histogram เฉพาะ column ที่เป็นจำนวน

nyc_bnb[cols_is_num].hist(figsize=(20, 20), bins = 30)

plt.show()
nyc_bnb.describe()
## price และ minimum_nights มีค่าค่อนข้างโดด

## ดูค่าโดดของ columns price

nyc_bnb[nyc_bnb['price'] > 0.9*5000]
nyc_bnb[nyc_bnb['minimum_nights'] > 400]
## ดูข้อมูลที่ไม่สมบูรณ์

## calculate NA proportion from each feature

nyc_bnb.isna().sum()/nrows
## look at name and host_name that have missing value

nyc_bnb[nyc_bnb['host_name'].isna()].shape
## host_id ไหนมีที่พักมากกว่า 1 แห่งบ้าง ?

nyc_bnb.groupby(by = ['host_id']).count().sort_values(['id'], ascending = False)[nyc_bnb.groupby(by = ['host_id']).count().sort_values(['id'], ascending = False)['id'] != 1]
## ดู last_review และ reviews_per_month ที่เท่ากับ NA

nyc_rv = nyc_bnb[['last_review','reviews_per_month']].isna()

nyc_rv[nyc_rv['last_review'] != nyc_rv['reviews_per_month']]

## ถ้าไม่มี reviews_per_month ก็จะไม่มี last_review ด้วย
## ดูความสัมพันธ์ของการรีวิวห้องกับ column อื่น

nyc_bnb[nyc_bnb['last_review'].isnull()]



## สังเกตเห็น number_of_reviews เป็น 0

nyc_bnb[nyc_bnb['last_review'].isnull()].sum()



## แถวที่ last_review และ reviews_per_month == NaN คือห้องพักที่ไม่เคยมีการ review
## ดังนั้นจะทำการ fill NaN ด้วย 0 ใน column reviews_per_month แต่เนื่องจาก last_review เป็นวันที่รีวิวครั้งล่าสุด จึงไม่สามารถเติมได้

nyc_bnb['reviews_per_month'] = nyc_bnb['reviews_per_month'].fillna(value = 0)

nyc_bnb.info()

nyc_bnb.isna().sum()/nrows
## ดู column name และ host name เนื่องจากมีข้อมูลไม่สมบูรณ์

nyc_bnb[['name','host_name']].isnull().sum()
## ดูความสัมพันธ์ระหว่าง 2 column นี้

nyc_bnb[nyc_bnb['name'].isnull() | nyc_bnb['host_name'].isnull()]

## ไม่มี row ใดซ้ำกัน
nyc_bnb.columns
## ดู column room_type

nyc_bnb.groupby(by = 'room_type').size()

room_dummies = pd.get_dummies(nyc_bnb['room_type'])
## ปรับ Dataframe ใหม่

nyc_bnb = nyc_bnb[nyc_bnb.columns.difference(['room_type','last_review'])].join(room_dummies)

nyc_bnb.info()
## import library สำหรับทำ Hirachical Clustering

from sklearn import preprocessing

from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch # draw dendrogram
nyc_bnb.columns
## Standardize ข้อมูลที่เป็นตัวเลขเพื่อให้อยู่ในหน่วยเดียวกัน

## เลือก column สำหรับทำ standadization และ clustering

cols = ['availability_365', 'calculated_host_listings_count', 'minimum_nights', 'price',

       'reviews_per_month', 'Entire home/apt', 'Private room', 'Shared room']

pt_standardize = preprocessing.PowerTransformer(method = 'yeo-johnson', standardize = True) # support only positive value

standardized = pt_standardize.fit_transform(nyc_bnb[cols])
standardized = pd.DataFrame(standardized, columns = cols)

standardized.head()
standardized[cols].hist(layout=(1, len(cols)), figsize=(3*len(cols), 3.5), color='salmon', alpha=.5)

plt.show()
#from sklearn.decomposition import PCA
#pca = PCA(random_state=42)
#pca.fit(standardized)
#pca.components_[0]
#pca.explained_variance_ratio_
#var_cumu = np.cumsum(pca.explained_variance_ratio_)

#var_cumu
#fig = plt.figure(figsize=[12,8],dpi=200)

#plt.vlines(x=3, ymax=1, ymin=0, colors="r", linestyles="--")

#plt.hlines(y=0.8, xmax=30, xmin=0, colors="g", linestyles="--")

#plt.plot(var_cumu)

#plt.ylabel("Cumulative variance explained")

#plt.show()
#from sklearn.decomposition import IncrementalPCA
#pca_final = IncrementalPCA(n_components=3)
#nyc_bnb_pca_final = pca_final.fit_transform(standardized)

#nyc_bnb_pca_final
## Plot Dendrogram

fig, ax = plt.subplots(figsize=(20, 7))

#ddg = sch.dendrogram(sch.linkage(standardized, method = 'ward'), ax = ax, labels = nyc_bnb['name'].values)

#ddg = sch.dendrogram(sch.linkage(nyc_bnb_pca_final, method = 'ward'), ax = ax)

ddg = sch.dendrogram(sch.linkage(standardized, method = 'ward'), ax = ax)

plt.show()

## จาก Dendrogram จะทำการแบ่งเป็น 3 clusters

hc = AgglomerativeClustering(n_clusters = 3, linkage = 'ward')

hc
hc.fit(standardized)
hc.labels_

hc.labels_.shape
## Labelling Cluster ไปใน Data

nyc_bnb['cluster'] =hc.labels_
nyc_bnb.info()
nyc_bnb.groupby(by = ['cluster']).mean()