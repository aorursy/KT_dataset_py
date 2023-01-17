import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
df = pd.read_csv("../input/online-retail-customer-clustering/OnlineRetail.csv", delimiter=',', encoding = "ISO-8859-1")
df.head()
df.info()
df.describe()
msno.bar(df)
df.count()
df[df['CustomerID'].isnull()].count()
100 - ((541909-135000)/541909 * 100)
df.dropna(inplace=True)
msno.bar(df)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')
df['Total Amount Spent']= df['Quantity'] * df['UnitPrice']

total_amount = df['Total Amount Spent'].groupby(df['CustomerID']).sum()
total_amount = pd.DataFrame(total_amount).reset_index()
total_amount.head()
transactions = df['InvoiceNo'].groupby(df['CustomerID']).count()
transaction = pd.DataFrame(transactions).reset_index()
transaction.head()
final = df['InvoiceDate'].max()
df['Last_transact'] = final - df['InvoiceDate']
LT = df.groupby(df['CustomerID']).min()['Last_transact']
LT = pd.DataFrame(LT).reset_index()
LT.head()
df_new = pd.merge(total_amount, transaction, how='inner', on='CustomerID')
df_new = pd.merge(df_new, LT, how='inner', on='CustomerID')
df_new.head()
df_new['Last_transact'] = df_new['Last_transact'].dt.days
df_new.head()
from sklearn.cluster import KMeans
kmeans= KMeans(n_clusters=2)
kmeans.fit(df_new[['Total Amount Spent', 'InvoiceNo', 'Last_transact']])
pred = kmeans.predict(df_new[['Total Amount Spent', 'InvoiceNo', 'Last_transact']])
kmeans.cluster_centers_
kmeans.labels_
pred = pd.DataFrame(pred, columns=['pred'])
df_new = df_new.join(pred)
fig, ax =plt.subplots(nrows= 1, ncols = 3, figsize= (14,6))
ty=sns.stripplot(x='pred', y='Total Amount Spent', data=df_new, s=8, ax = ax[0], palette='magma_r')
sns.despine(left=True)
ty.set_title('Clusters based on different Amounts')
ty.set_ylabel('Total Spent')
ty.set_xlabel('Clusters')

tt=sns.boxplot(x='pred', y='InvoiceNo', data=df_new, ax = ax[1], palette='coolwarm_r')
tt.set_title('Clusters based on Number of Transactions')
tt.set_ylabel('Total Transactions')
tt.set_xlabel('Clusters')

tr=sns.boxplot(x='pred', y='Last_transact', data=df_new, ax = ax[2], palette='magma_r')
tr.set_title('Clusters based on Last Transaction')
tr.set_ylabel('Last Transactions (Days ago)')
tr.set_xlabel('Clusters')
sns.pairplot(hue='pred', data=df_new, diag_kind='kde', palette='magma')
kmeans.inertia_
error_rate = []
for clusters in range(1,16):
    kmeans = KMeans(n_clusters = clusters)
    kmeans.fit(df_new)
    kmeans.predict(df_new)
    error_rate.append(kmeans.inertia_)
    
error_rate = pd.DataFrame({'Cluster':range(1,16) , 'Error':error_rate})
error_rate
plt.figure(figsize=(12,8))
p = sns.barplot(x='Cluster', y= 'Error', data= error_rate, palette='coolwarm_r')
sns.despine(left=True)
p.set_title('Error Rate and Clusters')

country_wise = df.groupby('Country').sum()
country_codes = pd.read_csv('../input/iso-country-codes-global/wikipedia-iso-country-codes.csv', names=['Country', 'two', 'three', 'numeric', 'ISO'])
country_codes.head()
country_wise = pd.merge(country_codes,country_wise, on='Country')
country_wise.head()
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()
import plotly.graph_objs as go
data = dict(type='choropleth',colorscale='GnBu', locations = country_wise['three'], locationmode = 'ISO-3', z= country_wise['Total Amount Spent'], text = country_wise['Country'], colorbar={'title':'Revenue'},  marker = dict(line=dict(width=0))) 
layout = dict(title = 'European Countries According to Revenue!', geo = dict(scope='europe',showlakes=False, projection = {'type': 'winkel tripel'}))
Choromaps2 = go.Figure(data=[data], layout=layout)
iplot(Choromaps2)

data = dict(type='choropleth',colorscale='rainbow', locations = country_wise['three'], locationmode = 'ISO-3', z= country_wise['Total Amount Spent'], text = country_wise['Country'], colorbar={'title':'Revenue'},  marker = dict(line=dict(width=0))) 
layout = dict(title = 'All Countries According to Revenue!', geo = dict(scope='world',showlakes=False, projection = {'type': 'winkel tripel'}))
Choromaps2 = go.Figure(data=[data], layout=layout)
iplot(Choromaps2)


from IPython.display import Image
Image("../input/image1/giphy (1).gif")