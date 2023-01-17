import pandas as pd

import seaborn as sns

import numpy as np # linear algebra

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

data=pd.read_csv("../input/kc_house_data.csv",parse_dates = ['date'])

data['waterfront'] = data['waterfront'].astype('category',ordered=True)

data['view'] = data['view'].astype('category',ordered=True)

data['condition'] = data['condition'].astype('category',ordered=True)

data['grade'] = data['grade'].astype('category',ordered=False)

data['zipcode'] = data['zipcode'].astype(str)

data = data.sort('date')

data.dtypes

data.describe()
t3=pd.cut (data['yr_built'],bins = [1900, 1925, 1950, 1975, 2000,2025])

data['cat_yr']=t3

table1=data.pivot_table(['price','sqft_living15'],['bedrooms','cat_yr'], aggfunc=('sum','count'), fill_value = 0)
t2=table1['price']['sum']/table1['price']['count']

round(t2)

t2.name='avg'

t2=t2.astype('int64')

table2 = pd.concat([table1['price'], t2], axis=1, join='inner')

table2['avg']
#table2['avg'].plot()

table2['avg'].plot()

#sns.kdeplot(x, bw=.2, label="bw: 0.2")

#sns.kdeplot(x, bw=2, label="bw: 2")

plt.legend();