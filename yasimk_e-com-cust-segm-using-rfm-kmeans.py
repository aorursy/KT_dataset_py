import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
df = pd.read_csv('eCom_cust_segmentation.csv')
df.head()
df.describe()
df['CustomerID'].nunique()
## Quantity and Unit price are highly skewed
df.hist(figsize=(10,5))
plt.show()
df.isnull().sum()
## Invoice date is a string. Also StockCode, description and Country
df.info()
figure = plt.figure(figsize=(20,20))
sns.pairplot(df);
plt.show()
df.groupby('Country').CustomerID.count().plot.bar(ylim=0)
plt.show()
def plot_corr_matrix(df):
    df_corr = df.corr()
    fig, ax = plt.subplots(figsize=(12,12))
    cax = ax.matshow(df_corr.values, interpolation='nearest')
    fig.colorbar(cax)
    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(len(df.columns)), df.columns)
plot_corr_matrix(df)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['InvoiceDate'].head()
df.head()
## Find the last txion date for each customer
df['LastTxionDate'] = df.groupby('CustomerID')['InvoiceDate'].transform('max')
df['FirstTxionDate'] = df.groupby('CustomerID')['InvoiceDate'].transform('min')
df[df['CustomerID'] == 12747].sort_values('InvoiceDate')
snapshot_dt = df['InvoiceDate'].max() + dt.timedelta(1)
snapshot_dt
df_group = df.groupby('CustomerID').agg(
    {
        'InvoiceDate':lambda x:(snapshot_dt - x.max()).days,
        'InvoiceNo':'count',
        'UnitPrice':'sum',
        #'InvoiceDate':lambda x:(snapshot_dt - x.min()).days,
    }
)
df_group.head()
df_group.rename(columns=
                {
                    'InvoiceDate':'Recency',
                    'InvoiceNo':'Frequency',
                    'UnitPrice':'Monetary'
                },inplace=True
               )
df_group.head()
import numpy as np
df_log = np.log(df_group)
df_log.head()
### Standardize the values using StandardScaler. Output will be in Numpy format
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_log)
rfm_std = scaler.transform(df_log)
rfm_std
df_rfm_std = pd.DataFrame(rfm_std, columns=df_log.columns, index=df_log.index)
#df_rfm_data_norm = pd.DataFrame(rfm_data_norm, index=rfm_data_log.index, columns=rfm_data_log.columns)
df_rfm_std.head()
#df_rfm_std.describe()
df_rfm_std.describe().round(5)
from sklearn.cluster import KMeans
sse={}
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, random_state=1)
    ## Fit the data to this normalized data set
    kmeans.fit(df_rfm_std)
    ## Assign sum of squared errors to k element of dictionary
    sse[k]=kmeans.inertia_

##Plot the elbow plot
plt.title('Elbow plot to determine optimum clusters')
plt.xlabel('K')
plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()
### Doing the log and std means data is normalized
sns.distplot(df_rfm_std['Recency'])
plt.show()
### Doing the log and std means data is normalized
sns.distplot(df_rfm_std['Frequency'])
plt.show()
### Doing the log and std means data is normalized
sns.distplot(df_rfm_std['Monetary'])
plt.show()
### k = 4 is the optimum cluster size
kmeans = KMeans(n_clusters=4, random_state=1)
kmeans.fit(df_rfm_std)
cluster_labels = kmeans.labels_
df_rfm_cluster = df_rfm_std.assign(Cluster=cluster_labels)
df_rfm_cluster.head(10)
df_rfm_cluster_grp = df_rfm_cluster.groupby(['Cluster']).agg(
    {
        'Recency':'mean',
        'Frequency':'mean',
        'Monetary':['sum','count']
    }
)
df_rfm_cluster_grp.head()
df_group['Cluster'] = df_rfm_cluster['Cluster']
df_group.head()
df_group = df_group.reset_index()
df_group[df_group['CustomerID']==12747]
df_group.info()
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(df_group.values)
df[df['CustomerID'] == 12747].sort_values('InvoiceDate')
df.info()
df_group1 = df.groupby('CustomerID').agg(
    {
        'LastTxionDate':'max',
        'FirstTxionDate':'min',
        'Country':'unique'
    }
)
df_group1 = df_group1.reset_index()
df_group1.head()
df_group1[df_group1['CustomerID']==12747]
df_group2 = pd.merge(df_group, df_group1, on='CustomerID', how='inner')
df_group2[df_group2['CustomerID']==12747]
df_group2.head()
## Cluster 0 - Not recent, Not frequent, Largest no of transactions, High cost purchases ($38/transaction)-Gold customers
## Cluster 1 - Most Recent, Most Frequent, Large no of transactions and spent the most ($177/transaction)-Platinum customers
## Cluster 2 - Least recent, lease frequent, Fewer no of transactions and least spent ($6/transaction) - Churned customers
## Cluster 3 - Recent, Not that frequent, Fewer no of transactions, Medium cost purchases ($27/transaction) - regular customers
df_group2.groupby(['Cluster']).agg(
    {
        'Recency':'mean',
        'Frequency':'mean',
        'Monetary':['count','sum']
    }
)
df_group2.describe()
df_group2.to_csv("ecom_clustering_export.csv", encoding='utf-8', sep=',', index=False)
## Cluster 0 - Not recent, Not frequent, Largest no of transactions, High cost purchases ($38/transaction)-Gold customers
## Cluster 1 - Most Recent, Most Frequent, Large no of transactions and spent the most ($177/transaction)-Platinum customers
## Cluster 2 - Least recent, lease frequent, Fewer no of transactions and least spent ($6/transaction) - Churned customers
## Cluster 3 - Recent, Not that frequent, Fewer no of transactions, Medium cost purchases ($27/transaction) - regular customers
def customer_segment(df):
    if df['Cluster'] == 0:
        return 'Gold'
    elif df['Cluster'] == 1:
        return 'Platinum'
    elif df['Cluster'] == 2:
        return 'Churned'
    else:
        return 'Regular'
df_group2['CustomerSegment'] = df_group2.apply(customer_segment, axis=1)
df_group2.head()
## Cluster 0 - Not recent, Not frequent, Largest no of transactions, High cost purchases ($38/transaction)-Gold customers
df_group2[df_group2['CustomerSegment']=='Gold']
## Cluster 1 - Most Recent, Most Frequent, Large no of transactions and spent the most ($177/transaction)-Platinum customers
df_group2[df_group2['CustomerSegment']=='Platinum']
df_group2.groupby('CustomerSegment')['CustomerID'].describe().round(0)
df_group2.groupby('CustomerSegment')['CustomerID'].mean().round(0).sort_values()