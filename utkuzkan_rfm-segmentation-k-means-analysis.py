# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy
import math
import scipy.stats as stats
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Set Options
pd.options.display.max_columns
pd.options.display.max_rows = 500
_2010_2011_data =pd.read_excel("/kaggle/input/uci-online-retail-ii-data-set/online_retail_II.xlsx",sheet_name= "Year 2010-2011" )
#Get copy of dataFrame
_2010_2011_df = _2010_2011_data.copy()
def setTotalPrice(data):
    #Add a column for total price to calculate monetary attritube
    data["TotalPrice"] = data["Quantity"]*data["Price"]
setTotalPrice(_2010_2011_df)
#Remove missing Values From CustomerID
_2010_2011_df.dropna(subset= ["Customer ID"],inplace= True)
#Remove zero negative quantity
deleteRows = _2010_2011_df[~_2010_2011_df['Quantity'] > 0].index
_2010_2011_df.drop(deleteRows, axis=0,inplace=True)
#Some rows start with C means refund so we will remove them
deleteRows =  _2010_2011_df[_2010_2011_df["Invoice"].str.contains("C", na=False)].index
_2010_2011_df.drop(deleteRows, axis=0,inplace=True)
def CalculateRFM(data):
    #Calculate recency
    #Find out the first and last order dates in the data.
    max_date = data['InvoiceDate'].max()
    import datetime as dt
    today_date = dt.datetime(max_date.year,max_date.month,max_date.day)
    recency_df = data.groupby("Customer ID").agg({'InvoiceDate': lambda x: (today_date - x.max()).days})
    recency_df.rename(columns={"InvoiceDate":"Recency"}, inplace= True)
    #calculate Frequency
    temp_df =  data.groupby(['Customer ID','Invoice']).agg({'Invoice': "count"}).groupby(['Customer ID']).agg({"Invoice": "count"})
    freq_df = temp_df.rename(columns={"Invoice": "Frequency"})
    monetary_df=data.groupby("Customer ID").agg({'TotalPrice': "sum"})
    monetary_df.rename(columns={"TotalPrice": "Monetary"}, inplace=True)
    rfm = pd.concat([recency_df,freq_df,monetary_df], axis = 1)
    return rfm 
_2010_2011_rfm = CalculateRFM(_2010_2011_df)
#Set RFM Score
def setRFMScore(rfm):
    # Get RFM scores for 3 attribute
    rfm["RecencyScore"] = pd.qcut(rfm['Recency'],5, labels=[5,4,3,2,1])
    #if you calculate only transaction operations(unique invoice per customer) add rank(method="first")
    #if you sum all operations in per invoice no need to add rank method
    rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'].rank(method="first"),5, labels=[1,2,3,4,5])
    rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'],5, labels=[1,2,3,4,5])
    rfm["RFM_SCORE"] = rfm["RecencyScore"].astype(str) +rfm["FrequencyScore"].astype(str)+rfm["MonetaryScore"].astype(str) 
setRFMScore(_2010_2011_rfm)
def setSegment(rfm):
    seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Can\'t Loose',
    r'3[1-2]': 'About to Sleep',
    r'33': 'Need Attention',
    r'[3-4][4-5]': 'Loyal Customers',
    r'41': 'Promising',
    r'51': 'New Customers',
    r'[4-5][2-3]': 'Potential Loyalists',
    r'5[4-5]': 'Champions'}
    rfm["Segment"] = rfm["RecencyScore"].astype(str) + rfm["FrequencyScore"].astype(str)
    rfm["Segment"] = rfm["Segment"].replace(seg_map,regex=True)
setSegment(_2010_2011_rfm)
_2010_2011_rfm.head(100)
_2010_2011_rfm['RFM_Segment_Score'] = _2010_2011_rfm[['RecencyScore','FrequencyScore','MonetaryScore']].sum(axis=1)
_2010_2011_rfm['Score'] = 'Green'
_2010_2011_rfm.loc[_2010_2011_rfm['RFM_Segment_Score']>5,'Score'] = 'Bronze' 
_2010_2011_rfm.loc[_2010_2011_rfm['RFM_Segment_Score']>7,'Score'] = 'Silver' 
_2010_2011_rfm.loc[_2010_2011_rfm['RFM_Segment_Score']>9,'Score'] = 'Gold' 
_2010_2011_rfm.loc[_2010_2011_rfm['RFM_Segment_Score']>10,'Score'] = 'Platinum'
# define function for the values below 0
def neg_to_zero(x):
    if x <= 0:
        return 1
    else:
        return x
# apply the function to Recency and MonetaryValue column 
_2010_2011_rfm['Recency'] = [neg_to_zero(x) for x in _2010_2011_rfm.Recency]
_2010_2011_rfm['Monetary'] = [neg_to_zero(x) for x in _2010_2011_rfm.Monetary]
# unskew the data
_2010_2011_rfm_log = _2010_2011_rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1).round(3)
# scale the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(_2010_2011_rfm_log)
# transform into a dataframe
rfm_scaled = pd.DataFrame(rfm_scaled, index = _2010_2011_rfm.index, columns = _2010_2011_rfm_log.columns)
from sklearn import cluster
import seaborn as sns
# the Elbow method
wcss = {}
for k in range(1, 11):
    kmeans = cluster.KMeans(n_clusters= k, init= 'k-means++', max_iter= 300)
    kmeans.fit(rfm_scaled)
    wcss[k] = kmeans.inertia_
# plot the WCSS values
sns.pointplot(x = list(wcss.keys()), y = list(wcss.values()))
plt.xlabel('K Numbers')
plt.ylabel('WCSS')
plt.show()
# clustering
clus = cluster.KMeans(n_clusters= 3, init= 'k-means++', max_iter= 300)
clus.fit(rfm_scaled)
# Assign the clusters to datamart
_2010_2011_rfm['K_Cluster'] = clus.labels_
_2010_2011_rfm.head(100)