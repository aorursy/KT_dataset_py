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

import seaborn as sns

import datetime as dt

import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_excel('/kaggle/input/online-retail-data-set-from-uci-ml-repo/Online Retail.xlsx')
df.head()
df.describe()
df.shape[0]
missing_percent = pd.DataFrame(df.isnull().mean()*100)

missing_percent.rename(columns={0:'Percentage'}

           ,inplace= True)

missing_percent.sort_values(ascending=False, by= ['Percentage'])

#This allows us to find the percentage of missing vales. As we can see below, nearly 25% of customer ids are missing.
#Creating a column holding information on whether customer id is null or not

df['id_null'] = np.where(df['CustomerID'].isnull(), 1, 0)





# Percentage of missing customer ids per country

null_by_country = pd.DataFrame(df.groupby(['Country'])['id_null'].mean()*100)

null_by_country.rename(columns={'id_null':'Percentage'}

           ,inplace= True)

null_by_country.sort_values(ascending=False, by= ['Percentage'])[:10]
plt.figure(figsize=(12,9))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')
df= df.dropna(subset=['CustomerID'])
df.isna().sum()
df.describe()
df[df['Quantity'] < 0]
df[df['CustomerID']  == 17548]
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df = df.drop_duplicates()
df.describe()
df['TotalCost'] = df['Quantity'] * df['UnitPrice']
df.head()
df[df['TotalCost'] > 160000]
df = df[df['TotalCost'] < 160000]
most_purchased_prods = pd.DataFrame(df['Description'].value_counts().head(20)).rename(columns={'Description':'Count'})





plt.figure(figsize=(24,9))

plt.xticks(rotation=45)

plt.plot(most_purchased_prods, 'b*-')

plt.grid()
most_purchased_bycountry = pd.DataFrame(df['Country'].value_counts().head(20)).rename(columns={'Country':'Count'})



plt.figure(figsize=(24,9))

plt.xticks(rotation=45)

plt.plot(most_purchased_bycountry, 'g*-')

plt.grid()
nonuk_df = df[df['Country'] != 'United Kingdom']

most_purchased_bycountry_nonuk = pd.DataFrame(nonuk_df['Country'].value_counts().head(20)).rename(columns={'Country':'Count'})



plt.figure(figsize=(24,9))

plt.xticks(rotation=45)

plt.plot(most_purchased_bycountry_nonuk, 'g*-')

plt.grid()
del df['id_null']
print('Number of unique Invoice Nos: {}'.format(

    len(df.InvoiceNo.unique())))



print('Number of unique StockCode: {}'.format(

    len(df.StockCode.unique())))



print('Number of different Descriptions: {}'.format(

    len(df.Description.unique())))



print('Number of unique Customer Ids: {}'.format(

    len(df.CustomerID.unique())))



print('Number of different Countries: {}'.format(

    len(df.Country.unique())))
#Creating a datetime value for out InvoiceDate Column

df['InvoiceDateCon'] = df['InvoiceDate'].dt.date
df['InvoiceDateCon'] = pd.to_datetime(df['InvoiceDateCon'])
def get_month(x): return dt.datetime(x.year, x.month, 1)

df['InvoiceDateCon'] = df['InvoiceDateCon'].apply(get_month)
df.tail()
grouping = df.groupby('CustomerID')['InvoiceDateCon']

df['CohortMonth'] = grouping.transform('min')
df.tail()
def get_date_int(df, column):

    year = df[column].dt.year

    month = df[column].dt.month

    return year, month





invoice_year, invoice_month = get_date_int(df, 'InvoiceDateCon')



cohort_year, cohort_month = get_date_int(df, 'CohortMonth')



years_diff = invoice_year - cohort_year

months_diff = invoice_month - cohort_month



df['CohortIndex'] = years_diff * 12 + months_diff + 1
plt.figure(figsize=(25,9))

sns.countplot(x='CohortIndex', data=df)
df['TimeOfPurchase'] = df['InvoiceDate'].dt.time
time_of_purchase = df.groupby(['TimeOfPurchase']).agg({'TimeOfPurchase':'count'}).rename(columns={'TimeOfPurchase':'Count'})



plt.figure(figsize=(25,9))

plt.plot(time_of_purchase, 'ro', alpha=0.7)

plt.grid()

plt.title('Time of purchase')

plt.xlabel('Time')

plt.ylabel('Number of purchases')
grouping = df.groupby(['CohortMonth', 'CohortIndex'])

cohort_data = grouping['CustomerID'].nunique()

cohort_data = cohort_data.reset_index()

cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
cohort_counts
cohort_sizes = cohort_counts.iloc[:,0]

retention = cohort_counts.div(cohort_sizes, axis=0)

retention_round = retention.round(3) * 100
retention_round
retention_round.index = retention_round.index.date
plt.figure(figsize=(15,8))

plt.title('Retention by Monthly Cohorts')

sns.heatmap(retention_round, annot=True, cmap="BuPu_r")

plt.show()
del df['TimeOfPurchase']
last_timestamp = df['InvoiceDate'].max() + dt.timedelta(days =1)
last_timestamp
rfm = df.groupby(['CustomerID']).agg({'InvoiceDate': lambda x : (last_timestamp - x.max()).days,

                                      'InvoiceNo':'count','TotalCost': 'sum'})
rfm.rename(columns={'InvoiceDate':'Recency', 'InvoiceNo':'Frequency', 'TotalCost':'MonetaryValue'}

           ,inplace= True)
rfm.head()
rfm.describe()
#Building RFM segments

r_labels =range(4,0,-1)

f_labels=range(1,5)

m_labels=range(1,5)

r_quartiles = pd.qcut(rfm['Recency'], q=4, labels = r_labels)

f_quartiles = pd.qcut(rfm['Frequency'],q=4, labels = f_labels)

m_quartiles = pd.qcut(rfm['MonetaryValue'],q=4,labels = m_labels)

rfm = rfm.assign(R=r_quartiles,F=f_quartiles,M=m_quartiles)



# Build RFM Segment and RFM Score

def add_rfm(x) : return str(x['R']) + str(x['F']) + str(x['M'])

rfm['RFM_Segment'] = rfm.apply(add_rfm,axis=1 )

rfm['RFM_Score'] = rfm[['R','F','M']].sum(axis=1)



rfm.tail()
rfm.groupby('RFM_Score').agg({'Recency': 'mean','Frequency': 'mean',

                             'MonetaryValue': ['mean', 'count'] }).round(1)
def segments(df):

    if df['RFM_Score'] > 9 :

        return 'Top'

    elif (df['RFM_Score'] > 5) and (df['RFM_Score'] <= 9 ):

        return 'Middle'

    else:  

        return 'Low'



rfm['General_Segment'] = rfm.apply(segments,axis=1)



rfm.groupby('General_Segment').agg({'Recency':'mean','Frequency':'mean',

                                    'MonetaryValue':['mean','count']}).round(1)
rfm_cluster = rfm.iloc[:,0:3]
rfm_cluster.describe()
f,ax = plt.subplots(figsize=(25, 5))

plt.subplot(1, 3, 1); sns.distplot(rfm_cluster['Recency'])

plt.subplot(1, 3, 2); sns.distplot(rfm_cluster['Frequency'])

plt.subplot(1, 3, 3); sns.distplot(rfm_cluster['MonetaryValue'])
rfm_cluster_log = np.log(rfm_cluster)
rfm_cluster_log.describe()
f,ax = plt.subplots(figsize=(25, 5))

plt.subplot(1, 3, 1); sns.distplot(rfm_cluster_log['Recency'])

plt.subplot(1, 3, 2); sns.distplot(rfm_cluster_log['Frequency'])

plt.subplot(1, 3, 3); sns.distplot(rfm_cluster_log['MonetaryValue'])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(rfm_cluster_log)
rfm_norm = scaler.fit_transform(rfm_cluster_log)
rfm_norm = pd.DataFrame(data=rfm_norm, index=rfm_cluster_log.index, columns=rfm_cluster_log.columns)
rfm_norm.describe()
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 20):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(rfm_norm)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(15,8))

plt.plot(range(1, 20), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.xticks(range(1, 20))

plt.grid(True)

plt.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)

kmeans.fit(rfm_norm)
kmeans.cluster_centers_
kmeans.labels_
rfm['Cluster'] = kmeans.labels_
rfm.groupby(['Cluster']).agg({'Recency': 'mean','Frequency': 'mean',

                                         'MonetaryValue': ['mean', 'count'],}).round(0)
df_trim = df.iloc[:,[6,7,8,9,11]]
mean_clus = rfm.groupby(['Cluster']).mean()

pop_avg = rfm.mean()

relative_imp = mean_clus / pop_avg - 1

relative_imp = relative_imp.dropna(axis=1).iloc[:,[3,0,1]]
relative_imp
sns.heatmap(relative_imp, annot=True, cmap="BuPu_r")
rfm_c1 = rfm[rfm['Cluster'] == 0]

rfm_c2 = rfm[rfm['Cluster'] == 1]

rfm_c3 = rfm[rfm['Cluster'] == 2]

rfm_c4 = rfm[rfm['Cluster'] == 3]

rfm_c5 = rfm[rfm['Cluster'] == 4]
rfm_c1.groupby(['General_Segment']).agg({'Recency': 'mean','Frequency': 'mean',

                                         'MonetaryValue': ['mean', 'count'],}).round(0)
rfm_c2.groupby(['General_Segment']).agg({'Recency': 'mean','Frequency': 'mean',

                                         'MonetaryValue': ['mean', 'count'],}).round(0)
rfm_c3.groupby(['General_Segment']).agg({'Recency': 'mean','Frequency': 'mean',

                                         'MonetaryValue': ['mean', 'count'],}).round(0)
rfm_c4.groupby(['General_Segment']).agg({'Recency': 'mean','Frequency': 'mean',

                                         'MonetaryValue': ['mean', 'count'],}).round(0)
rfm_c5.groupby(['General_Segment']).agg({'Recency': 'mean','Frequency': 'mean',

                                         'MonetaryValue': ['mean', 'count'],}).round(0)
df_trim
rfm['CustomerID'] = rfm.index
rfm
rfm.reset_index(drop=True, inplace=True)
df_rework = pd.merge(df_trim, rfm, on='CustomerID')
df_rework.head()
df_rework.describe()