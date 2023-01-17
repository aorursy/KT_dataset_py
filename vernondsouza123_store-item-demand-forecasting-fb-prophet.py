import numpy as np



import pandas as pd



import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



import seaborn as sns



import scipy.cluster.hierarchy as hac



import math



import random



from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf



from fbprophet import Prophet



import random

# https://alpynepyano.github.io/healthyNumerics/posts/time_series_clustering_with_python.html

# https://kourentzes.com/forecasting/2014/11/09/additive-and-multiplicative-seasonality/

# https://dius.com.au/2018/09/04/time-series-forecasting-with-fbprophet/

# https://www.kaggle.com/beebopjones/bleepblop-prophet-model

# https://www.kaggle.com/viridisquotient/sarima
df = pd.read_csv("../input/demand-forecasting-kernels-only/train.csv")

print(df.head(10))
# Find unique columns

print("length of database",len(df))

print("Unique Stores",df['store'].unique())

print("Unique items",df['item'].unique())

print("Unique dates",df['date'].nunique())



# Include holidays

holidays = pd.read_csv("../input/federal-holidays-usa-19662020/usholidays.csv")





holidays = holidays.drop(columns=["Unnamed: 0"])

print(holidays.head(10))

df['date'] = pd.to_datetime(df['date']) 

holidays['Date'] = pd.to_datetime(holidays['Date']) 

needed_holidays = holidays[(holidays['Date']>df.iloc[0]['date'])&(holidays['Date']<df.iloc[-1]['date'])]['Date'].to_list()

print(len(needed_holidays))



# For later Analysis

holidays = holidays[(holidays['Date']>df.iloc[0]['date'])&(holidays['Date']<df.iloc[-1]['date'])]

holidays.rename(columns={"Date": "ds", "Holiday": "holiday"},inplace=True)
select_values = needed_holidays

df_cut = df[df['date'].isin(select_values)]['date']

print(df_cut.nunique())



# Holidays have been included
# Check for Missing  data

total = df.isnull().sum()

print(total)
# Getting individual distributions of sales for each item. Taking sum to summate sales for different stores

df['date'] = pd.to_datetime(df['date']) 



df_sales_item = df.groupby(['date','item']).sum()  

df_sales_item.reset_index(level=0, inplace=True)

df_sales_item.reset_index(level=0, inplace=True)

#print(df_sales_item)



grid = sns.FacetGrid(df_sales_item, col="item", col_wrap=5)

grid = grid.map(plt.scatter, "date", "sales", marker="o", s=1, alpha=.5)
# Getting individual distributions of sales for each store. Taking sum to summate sales for different items

df['date'] = pd.to_datetime(df['date']) 



df_sales_store = df.groupby(['date','store']).sum()  

df_sales_store.reset_index(level=0, inplace=True)

df_sales_store.reset_index(level=0, inplace=True)





grid = sns.FacetGrid(df_sales_store, col="store", col_wrap=5)

gris = grid.map(plt.scatter, "date", "sales", s=1, alpha=.5)
# Lets look at sales for a specified store and a item

plt.figure(figsize=(30,5))

df_store_item = df[(df.store==10) & (df.item==5)]

plt.plot(df_store_item['date'],df_store_item['sales'])

plt.show()
stores = df['store'].unique()

items = df['item'].unique()



Sales_series = []

count = 0

IndexSale_Series = []

Dates = []



#df['date'] = pd.to_datetime(df['date']) 



for store in stores:

    for item in items:

        Sales = df[(df.store==store) & (df.item==item)]['sales'].to_list()

        dates = df[(df.store==store) & (df.item==item)]['date'].to_list()

        

        Dates.append(dates)

        Sales_series.append(Sales)

        

        index = [count,item,store]

        IndexSale_Series.append(index)

        

        count +=1



print("Total Elements: ",count)
def plot_dendogram(Z):

    with plt.style.context('fivethirtyeight' ): 

         plt.figure(figsize=(100, 40))

         plt.title('Dendrogram of time series clustering',fontsize=25, fontweight='bold')

         plt.xlabel('sample index', fontsize=25, fontweight='bold')

         plt.ylabel('distance', fontsize=25, fontweight='bold')

         hac.dendrogram( Z, leaf_rotation=90.,    # rotates the x axis labels

                            leaf_font_size=15., ) # font size for the x axis labels

         plt.show()

        

def plot_resultsAndReturnClusters(timeSeries, D, cut_off_level):

    result = pd.Series(hac.fcluster(D, cut_off_level, criterion='maxclust'))

    clusters = result.unique()       

    figX = 100; figY = 20

    fig = plt.subplots(figsize=(figX, figY))   

    mimg = math.ceil(cut_off_level/2.0)

    gs = gridspec.GridSpec(mimg,2, width_ratios=[1,1])

    cluster = []

    for ipic, c in enumerate(clusters):

        cluster_index = result[result==c].index

        cluster.append(cluster_index)

        

        print(ipic, "Cluster number %d has %d elements" % (c, len(cluster_index)))

        ax1 = plt.subplot(gs[ipic])

        timeSeries = np.array(timeSeries)

        ax1.plot(timeSeries.T[:,cluster_index])

        ax1.set_title(('Cluster number '+str(c)), fontsize=15, fontweight='bold')      

    

    plt.show()

    return cluster
D = hac.linkage(Sales_series, method='ward', metric='euclidean')

plot_dendogram(D)



#---- evaluate the dendogram

cut_off_level = 2   # level where to cut off the dendogram

clusters = plot_resultsAndReturnClusters(Sales_series, D, cut_off_level)

# Random time series in cluster 1

no = random.randint(0,101)



index = clusters[0][no]
# Check for weekly seasonality using ACF

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(211)

fig = plot_acf(Sales_series[index],lags=40,ax=ax)



# Check for monthly seasonality using ACF

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(211)

fig = plot_acf(Sales_series[index],lags=400,ax=ax)
# Check for yearly seasonality using ACF

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(211)

fig = plot_acf(Sales_series[index],lags=1800,ax=ax)
plt.figure(figsize=(30,10))

print("Image Corresponding to Item:",IndexSale_Series[index][1],"And Store:",IndexSale_Series[index][2])

plt.plot(Sales_series[index][:100])

plt.show()

plt.figure(figsize=(30,10))

plt.plot(Sales_series[index][1500:])

plt.show()
NonStationary = []



for saleID in clusters[0]:

    #print(Sales_series[saleID])

    result = adfuller(Sales_series[saleID])

    

    if result[1] > 0.05:

        

        NonStationary.append(saleID)

    else:

        pass      



print(NonStationary)

SalesSample = pd.DataFrame()

SalesSample['y'] = Sales_series[index]

SalesSample['ds'] = Dates[index]
test_size = 90 # 3 months of data



train = SalesSample[:-test_size]

test = SalesSample[-test_size:]



plt.subplots(figsize=(20, 5))



plt.plot(train['ds'], train['y'],color='blue', label='Train')

plt.plot(test['ds'], test['y'], color='red', label='Test')

model = Prophet(daily_seasonality=False,

weekly_seasonality=True,

yearly_seasonality=True,

holidays = holidays, seasonality_mode='additive',holidays_prior_scale=0.5,)



# holiday prior scale to bring down the effect of holidays in the model
model.fit(train)

forecast = model.predict(test)

model.plot_components(forecast)
plt.figure(figsize=(30, 5))



plt.plot(test['ds'], test['y'], c='r', label='Test')

plt.plot(forecast['ds'], forecast['yhat'], c='blue', marker='o',label='Forecast')

plt.show()
# Calculate SMAPE

y_true = test['y'].to_list()

y_true = np.array(y_true)

y_forecast = forecast['yhat'].to_list()

y_forecast = np.array(y_forecast)



smape = (np.absolute(y_true - y_forecast) / (np.absolute(y_true) + np.absolute(y_forecast))).mean() * 200

print('SMAPE is:', smape)
train['y'] = np.log1p(train['y'])
model = Prophet(daily_seasonality=False,

weekly_seasonality=True,

yearly_seasonality=True,

holidays = holidays, seasonality_mode='additive',holidays_prior_scale=0.5,)



model.fit(train)

forecast = model.predict(test)

model.plot_components(forecast)
forecast['yhat'] = np.expm1(forecast['yhat'])

print(forecast['yhat'])
plt.figure(figsize=(30, 5))

plt.plot(test['ds'], test['y'], c='r', label='Test')

plt.plot(forecast['ds'], forecast['yhat'], c='blue', marker='o',label='Forecast')

plt.show()
y_true = test['y'].to_list()

y_true = np.array(y_true)

y_forecast = forecast['yhat'].to_list()

y_forecast = np.array(y_forecast)



smape = (np.absolute(y_true - y_forecast) / (np.absolute(y_true) + np.absolute(y_forecast))).mean() * 200

print('SMAPE  is:', smape)
# Random time series in cluster 2

no = random.randint(0,101)



index = clusters[1][no]
# Check for weekly seasonality using ACF

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(211)

fig = plot_acf(Sales_series[index],lags=24,ax=ax)
# Check for monthly seasonality using ACF

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(211)

fig = plot_acf(Sales_series[index],lags=120,ax=ax)
# Check for yearly seasonality using ACF

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(211)

fig = plot_acf(Sales_series[index],lags=750,ax=ax)
plt.figure(figsize=(30,10))

print("Image Corresponding to Item:",IndexSale_Series[index][1],"And Store:",IndexSale_Series[index][2])

plt.plot(Sales_series[index][:100])

plt.show()

plt.figure(figsize=(30,10))

plt.plot(Sales_series[index][1500:])

plt.show()
NonStationary = []



for saleID in clusters[1]:

    #print(Sales_series[saleID])

    result = adfuller(Sales_series[saleID])

    

    if result[1] > 0.05:

        

        NonStationary.append(saleID)

    else:

        pass      



print(NonStationary)
train = pd.read_csv("../input/demand-forecasting-kernels-only/train.csv", parse_dates=['date'], index_col=['date'])

test = pd.read_csv("../input/demand-forecasting-kernels-only/test.csv",parse_dates=['date'],index_col=['date'])



results = test.reset_index()

results['sales'] = 0



stores = df['store'].unique()

items = df['item'].unique()



for store in stores :

    for item in items:

        

        to_train = train.loc[(train['store'] == store) & (train['item'] == item)].reset_index()

        to_train.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)

        

        to_train['y'] = np.log1p(to_train['y'])

        

        model = Prophet(daily_seasonality=False,

        weekly_seasonality=True,

        yearly_seasonality=True,

        holidays = holidays, seasonality_mode='additive',holidays_prior_scale=0.5,)

        

        model.fit(to_train[['ds', 'y']])

        

        future = model.make_future_dataframe(periods=len(test.index.unique()),include_history=False)

        forecast = model.predict(future)

        

        results.loc[(results['store'] == store) & (results['item'] == item),'sales'] = np.expm1(forecast['yhat']).values
results.drop(['date', 'store', 'item'], axis=1, inplace=True)

results.head()
results['sales'] = np.round(results['sales']).astype(int)
results.to_csv('submission.csv', index=False)