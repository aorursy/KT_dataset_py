import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

df_train = load_df()
df_test = load_df("../input/test.csv")
#Lets have a look at the data
df_train.head(10)
df_train.columns
print('Is there more than one transaction by VisitorId in train dataset?',
      len(df_train['fullVisitorId'])!=df_train['fullVisitorId'].nunique())
print('Is there more than one transaction by VisitorId in test dataset?',
      len(df_test['fullVisitorId'])!=df_test['fullVisitorId'].nunique())
#Confirming that we have more rows than unique Visitors Id's in both train and test datasets.
type(df_train['totals.transactionRevenue'])
#Lets see how the transaction revenues behave among all unique users
import matplotlib.pyplot as plt
df_train["totals.transactionRevenue"] = df_train["totals.transactionRevenue"].astype('float')
gdf = df_train.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()
plt.figure(figsize=(10,8))
plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totals.transactionRevenue"])))
plt.xlabel('index Users', fontsize=12)
plt.ylabel('TransactionRevenue', fontsize=12)
plt.show()
#By these two plots we can conclude that the majority of the TransactionRevenue comes from a very little portion of costumers.
#Therefore, the marketing teams must direct carefully their effor on investments.
#Lets see a distribution plot of the target variable to confirm our hypothesis
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
fig, ax = plt.subplots(figsize=(12, 8))
sns.distplot(np.log1p(gdf["totals.transactionRevenue"]),ax=ax)
#As expected, we se that the majority of users don't contribute to the revenue.
print('\n',pd.notnull(df_train["totals.transactionRevenue"]).sum(),
' Non-null revenue Instances ocurred, which represent',
100*(pd.notnull(df_train["totals.transactionRevenue"]).sum()/len(df_train)),'% of all Instances')
print('\n',gdf["totals.transactionRevenue"][gdf["totals.transactionRevenue"]>0].count(),
' Customers contribute with non-zero revenue which is equivalent to',
100*(gdf["totals.transactionRevenue"][gdf["totals.transactionRevenue"]>0].count()/len(gdf)),'% of all Customers')
dropcols = [c for c in df_train.columns if df_train[c].nunique()==1]
dropcols_test = [c for c in df_test.columns if df_test[c].nunique()==1]
df_train = df_train.drop(dropcols,axis=1)
df_test = df_test.drop(dropcols_test,axis=1)
#Definition of the aggregation funtion, that we will use to build some % features
def aggregations(feature):
    df = df_train.groupby(feature)['totals.transactionRevenue'].agg(['size', 'count'])
    df.columns = ["count of instances", "count of non-zero revenue"]
    df['percent of instances[%]'] = (df['count of instances']*100)/df['count of instances'].sum()
    df['percent of non-zero revenue[%]'] = (df['count of non-zero revenue']*100)/df['count of non-zero revenue'].sum()
    df = df.sort_values(by="percent of non-zero revenue[%]", ascending=False)
    return df
# Device Browser Analisys:
cnt_srs = aggregations('device.browser')
top_rev = cnt_srs["percent of non-zero revenue[%]"].nunique()
cnt_srs[:6].plot.barh(y=['percent of non-zero revenue[%]','percent of instances[%]'], rot=0,figsize=(12,9))
#It is pretty clear that the majority of instances and non-zero revenues comes from chromes users.
# Device Category (Desktop,...)
cnt_cat = aggregations('device.deviceCategory')
cnt_cat.plot.barh(y=['percent of non-zero revenue[%]','percent of instances[%]'], rot=0,figsize=(12,9))
#On the device category front, desktop seem to have higher percentage of non-zero revenue counts compared to mobile devices.
# Device Operating System (Windows,...)
cnt_os = aggregations('device.operatingSystem')
cnt_os[:8].plot.barh(y=['percent of non-zero revenue[%]','percent of instances[%]'], rot=0,figsize=(12,9))
#In device operating system, although the number of counts is more from windows, 
#the number of counts where revenue is not zero is more for Macintosh.
#Date Information: Lets split the information in Years/Months/Days and split the analisys.
df_train['year']= df_train['date'].astype(str).str[:4]
df_test['year']= df_test['date'].astype(str).str[:4]
df_train['month']= df_train['date'].astype(str).str[4:6]
df_test['month']= df_test['date'].astype(str).str[4:6]
df_train['day']= df_train['date'].astype(str).str[6:8]
df_test['day']= df_test['date'].astype(str).str[6:8]
df_train.drop('date',axis=1,inplace=True)
df_test.drop('date',axis=1,inplace=True)
# Year Analisys
cnt_y = aggregations('year')
cnt_y.plot.barh(y=['percent of non-zero revenue[%]','percent of instances[%]'], rot=0,figsize=(12,9))
#Slight differences beetween years, probably not worth to keep this information on dataset
df_train.drop('year',axis=1,inplace=True)
df_test.drop('year',axis=1,inplace=True)
# Months Analisys
cnt_m = aggregations('month')
cnt_m.plot.barh(y=['percent of non-zero revenue[%]','percent of instances[%]'], rot=0,figsize=(12,9))
#Looks like at Dezember people are more likely to buy, most likely because of New Year parties like Christmas.
#An interesting observation here is that at November there is a peak of instances, proably because people 
#are researching what they should buy in December.
# Days of the month Analisys
cnt_d = aggregations('day')
cnt_d.plot.barh(y=['percent of non-zero revenue[%]','percent of instances[%]'], rot=0,figsize=(12,9))
#Looks like every 12th day of the month ppl are more likely to buy.

# Continent Analysis
cnt_con = aggregations('geoNetwork.continent')
cnt_con.plot.barh(y=['percent of non-zero revenue[%]','percent of instances[%]'], rot=0,figsize=(12,9))
#Clearly the Americas englobe the absolut majority of revenue. I was expecting the Europe to go on second, 
#but it is behind Asia.
# SubContinent Analysis
cnt_scon = aggregations('geoNetwork.subContinent')
cnt_scon[:12].plot.barh(y=['percent of non-zero revenue[%]','percent of instances[%]'], rot=0,figsize=(12,9))
#Again, the absolute majority of revenue comes from North America, followed by South America, Eastern Asia and South Asia. 
#From forth position on there is almost no representativity.
#Traffic Source
cnt_ts = aggregations('trafficSource.source')
cnt_ts[:20].plot.barh(y=['percent of non-zero revenue[%]','percent of instances[%]'], rot=0,figsize=(12,9))
#Google Search and Google Sales plataforms are domaining the counts and non-zero revenue.
#It is interesting to see that Youtube has a lot of counts, but has a very low non-zero ratio revenue.
#Traffic Source Medium:
cnt_tsm = aggregations('trafficSource.medium')
cnt_tsm.plot.barh(y=['percent of non-zero revenue[%]','percent of instances[%]'], rot=0,figsize=(12,9))
#Even though "organic users" have more counts overall, the counts from referral were converted into non-zero 
#revenue in a higher ratio.
#This shows us the big influence that referral has in sales.
#Hits
cnt_hit = aggregations('totals.hits').reset_index()
cnt_hit['totals.hits'] = cnt_hit['totals.hits'].astype(int)
cnt_hit.sort_values(by=['totals.hits'],inplace=True)
cnt_hit.set_index('totals.hits',inplace=True)
plt.figure(1,figsize=(12,6))
plt.plot(cnt_hit.index,cnt_hit['count of instances'],'r',
         markersize=4,linewidth=2,label='count of instances per total hits')
plt.legend()
plt.xlabel('totals.hits')
plt.ylabel('count of instances')

plt.figure(2,figsize=(12,6))
plt.plot(cnt_hit.index,cnt_hit['count of non-zero revenue'],'b',
         markersize=4,linewidth=2,label='count of non-zero revenue per total hits')
plt.legend()
plt.xlabel('totals.hits')
plt.ylabel('count of non-zero revenue')
#We can see that at low number of hits we have a low number of non-zero revenues, but as we increse the number 
#of hits we get more non-zero revenues, reaching its peak at around hits=25, and then decreasing the number of 
#non-zero revenues.

#PageViews
cnt_pv = aggregations('totals.pageviews').reset_index()
cnt_pv['totals.pageviews'] = cnt_pv['totals.pageviews'].astype(int)
cnt_pv.sort_values(by=['totals.pageviews'],inplace=True)
cnt_pv.set_index('totals.pageviews',inplace=True)

cnt_pv.sort_index(inplace=True)
plt.figure(3,figsize=(12,6))
plt.plot(cnt_pv.index[:100],cnt_pv['count of instances'][:100],'r',
         markersize=4,linewidth=2,label='count of instances per page view')
plt.legend()
plt.xlabel('totals.hits')
plt.ylabel('count of instances')

plt.figure(4,figsize=(12,6))
plt.plot(cnt_pv.index[:100],cnt_pv['count of non-zero revenue'][:100],'b',
         markersize=4,linewidth=2,label='count of non-zero revenue per page view')
plt.legend()
plt.xlabel('totals.hits')
plt.ylabel('count of instances')
#So we have a very similar behaviour between totals.hits and totals.pageviews in relation to both instances 
#and non-zero revenues.
#Most of users make very little instances, but the non-zero revenues comes users that acess the google plataform 
#more frenquently, around 20 times.
#Missing Data on Train Dataset
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#So there are many features with more than 90% of missing data, so we should look carefully at them and decide wheter to Imput
#this data or drop them.