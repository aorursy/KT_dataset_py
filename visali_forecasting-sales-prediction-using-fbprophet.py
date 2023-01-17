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
sales=pd.read_csv("../input/sales_train.csv")

item_catg=pd.read_csv("../input/item_categories.csv")

items=pd.read_csv("../input/items.csv")

sample=pd.read_csv("../input/sample_submission.csv")

shops=pd.read_csv("../input/shops.csv")

test=pd.read_csv("../input/test.csv")
import warnings

warnings.filterwarnings("ignore")
sales.head()
item_catg.head()
items.head()
sample.head()
shops.head()
test.head()
import datetime

sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

sales.head()
monthly_sales=sales.groupby(["date_block_num","shop_id","item_id"])[

    "date","item_price","item_cnt_day"].agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})



monthly_sales.head()
# number of items per cat 

x=items.groupby(['item_category_id']).count()

x=x.sort_values(by='item_id',ascending=False)

#x=x.iloc[0:20].reset_index()

#x
import matplotlib.pyplot as plt

import seaborn as sns 





plt.figure(figsize=(8,4))

ax= sns.barplot(items.item_category_id, items.item_id, alpha=0.8)

plt.title("Items per Category")

plt.ylabel('# of items', fontsize=12)

plt.xlabel('Category', fontsize=12)

plt.show()
ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()

ts.astype('float')
ts.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

ts=ts.reset_index()

ts.head()
ts.rename({'index':'ds','item_cnt_day':'y'},axis=1) 

ts.head()
from fbprophet import Prophet


ts.columns=['ds','y']

model = Prophet( yearly_seasonality=True) #instantiate Prophet  

model.fit(ts) #fit the model with your dataframe
future = model.make_future_dataframe(periods = 2, freq = 'MS') 

forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
model.plot(forecast)
model.plot_components(forecast)
# get the unique combinations of item-store from the sales data at monthly level

monthly_sales=sales.groupby(["shop_id","item_id","date_block_num"])["item_cnt_day"].sum()

# arrange it conviniently to perform the hts 

monthly_sales=monthly_sales.unstack(level=-1).fillna(0)

monthly_sales=monthly_sales.T

dates=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

monthly_sales.index=dates

monthly_sales=monthly_sales.reset_index()

monthly_sales.head()
import time

start_time=time.time()

forecastsDict = {}

for node in range(len(monthly_sales)):

    # take the date-column and the col to be forecasted

    nodeToForecast = pd.concat([monthly_sales.iloc[:,0], monthly_sales.iloc[:, node+1]], axis = 1)

    print(nodeToForecast.head())  # just to check

    

    # rename for prophet compatability

    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[0] : 'ds'})

    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[1] : 'y'})

    #nodeToForecast=nodeToForecast.reset_index() 

    print(nodeToForecast.head())

    nodeToForecast.columns = ['ds','y']

    growth = 'linear'

    

    m = Prophet(growth, yearly_seasonality=True)

    m.fit(nodeToForecast) 

    

    future = m.make_future_dataframe(periods = 1, freq = 'MS')

    forecastsDict[node] = m.predict(future)

    
monthly_shop_sales=sales.groupby(["date_block_num","shop_id"])["item_cnt_day"].sum()

# get the shops to the columns

monthly_shop_sales=monthly_shop_sales.unstack(level=1)

monthly_shop_sales=monthly_shop_sales.fillna(0)

monthly_shop_sales.index=dates

monthly_shop_sales=monthly_shop_sales.reset_index()

monthly_shop_sales.head()
start_time=time.time()

forecastsDict = {}

for node in range(len(monthly_shop_sales)):

    # take the date-column and the col to be forecasted

    nodeToForecast = pd.concat([monthly_shop_sales.iloc[:,0], monthly_shop_sales.iloc[:, node+1]], axis = 1)

#     print(nodeToForecast.head())  # just to check

# rename for prophet compatability

    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[0] : 'ds'})

    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[1] : 'y'})

    nodeToForecast.columns = ['ds','y']

    growth = 'linear'

    m = Prophet(growth, yearly_seasonality=True)

    m.fit(nodeToForecast)

    future = m.make_future_dataframe(periods = 1, freq = 'MS')

    forecastsDict[node] = m.predict(future)

    

end_time=time.time()

print("forecasting took",end_time-start_time,"s") 
nCols = len(list(forecastsDict.keys()))+1

for key in range(0, nCols-1):

    f1 = forecastsDict[key].yhat

    # print('key:',key,'forecast',f1)

print(forecastsDict)

df = pd.DataFrame([forecastsDict])

df.head()
metric_df = nodeToForecast.set_index('ds')[['y']].join(monthly_shop_sales.set_index(["index"])).reset_index()
metric_df.head()