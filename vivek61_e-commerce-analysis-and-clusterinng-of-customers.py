import datetime as dt



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



# from paretochart import pareto
data = pd.read_excel('../input/online-dataset/Online Retail.xlsx')
data.head(n=5)
data.sample(n = 5)
data.info()
data.describe()
data.corr()
# Creating a column of Sales. qty*price

data['Sales'] = data.UnitPrice * data.Quantity



# Extracting year and month from the invoicedate.

data['year']=pd.DatetimeIndex(data.InvoiceDate).year

data['month']=pd.DatetimeIndex(data.InvoiceDate).month

data['week'] = pd.DatetimeIndex(data.InvoiceDate).week



# Creating a concatenated column

data['Month'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str), format = '%Y-%m')

data['Week'] = data['year'].astype(str) + '-' + data['week'].astype(str)

 

# Dropping the columns year and month

data.drop(['year','month','week'],axis =1, inplace =True)



data.head()
miss_data = data.isna().sum()

miss_data_percent = 100*data.isna().sum()/len(data)



miss_table = pd.concat([miss_data, miss_data_percent], axis= 1)

miss_table.columns = ['missing values', '% of data missing']



miss_table.sort_values(by='missing values', ascending=False, inplace= True)



miss_table
data.dropna(axis = 0, subset = ['CustomerID'], inplace= True)

data.describe()
data.drop_duplicates(subset=['InvoiceNo', 'StockCode','Quantity'])
data[data['CustomerID'].isna()]['Sales'].sum()
# Creating a new grouped table. grouping by customer wise sales and sort descending. 

cust_sales = data.groupby('CustomerID').sum().drop(['Quantity', 'UnitPrice'], axis =1)

cust_sales = cust_sales.sort_values(ascending= False, by="Sales")



# CustomerID into numeric indexes.

cust_sales.index = np.arange(1,(len(cust_sales)+1))



# Creating a new column for Cummulative Sales.

cust_sales['cumSales'] = cust_sales['Sales'].cumsum()



# Rounding to 0 decimals

cust_sales = cust_sales.round(decimals=0)



print(cust_sales.head())
cust_sales['index'] =cust_sales.index

cust_sales.head()
fig, ax = plt.subplots()

ax.bar(cust_sales.index, cust_sales["Sales"], color="C0")

ax2 = ax.twinx()

ax2.plot(cust_sales.index, cust_sales["cumSales"], color="C1", marker="D", ms=7)

# ax2.yaxis.set_major_formatter(PercentFormatter())



ax.tick_params(axis="y", colors="C0")

ax2.tick_params(axis="y", colors="C1")

plt.show()
fig = px.line(cust_sales, x="index", y="cumSales", width=600, height=400)

fig.show()
# plt.figure(figsize= (15,8))



# pareto(x = cust_sales.index, y = cust_sales['Sales'], axes = axes[0,1], limit = 0.8)



# plt.show



# # plt.bar(x= cust_sales.index, height=cust_sales['Sales'])



# #plt.plot(x = cust_sales.index, height = cust_sales['cumSales'] )



# #plt.ylim(0,20000)
#Mon_Sales=data.groupby('Month').sum().drop(['CustomerID', 'Quantity', 'UnitPrice'], axis =1)
data.groupby('InvoiceNo').sum().drop(['CustomerID', 'Quantity', 'UnitPrice'], axis =1)
cohorts = data.groupby('CustomerID')['Month']

data['coh_month'] = cohorts.transform('min')

data.sample(n=10)
data.coh_month.value_counts()
cohorts.transform('min')
pd.crosstab(data['coh_month'],data['Month'], values=data['CustomerID'],aggfunc=pd.Series.nunique)
# Indexing the months as 1,2,3 with reference to the coh_month.



# Extracting year and month from the invoicedate.

data['year'] = (pd.DatetimeIndex(data.Month).year) - (pd.DatetimeIndex(data.coh_month).year)

data['month'] = (pd.DatetimeIndex(data.Month).month) - (pd.DatetimeIndex(data.coh_month).month)

                                                                  

# Creating the cohort indices for cohort analysis                                                                  

data['coh_index'] = data['year']*12 + data['month'] + 1



# Dropping the columns year & month                                                              

data.drop(['year', 'month'], axis = 1, inplace= True)
data.info()
data['coh_index'].value_counts()
return_cust = pd.crosstab(data['coh_month'], data['coh_index'], values=data['CustomerID'],aggfunc=pd.Series.nunique)

return_cust
cohorts = data.groupby('CustomerID')['Month']

data['coh_month'] = cohorts.transform('min')



# Indexing the months as 1,2,3 with reference to the coh_month.



# Extracting year and month from the invoicedate.

data['year'] = (pd.DatetimeIndex(data.Month).year) - (pd.DatetimeIndex(data.coh_month).year)

data['month'] = (pd.DatetimeIndex(data.Month).month) - (pd.DatetimeIndex(data.coh_month).month)

                                                                  

# Creating the cohort indices for cohort analysis                                                                  

data['coh_index'] = data['year']*12 + data['month'] + 1



# Dropping the columns year & month                                                              

data.drop(['year', 'month'], axis = 1, inplace= True)



return_cust = pd.crosstab(data['coh_month'], data['coh_index'], 

                          values=data['CustomerID'],aggfunc=pd.Series.nunique)
coh_size = return_cust.iloc[:,0]

coh_size
return_rate = return_cust.divide(coh_size,axis=0)*100



return_rate.round(0)
list(return_rate.max().sort_values(ascending = False))[1]
mon_names = ['Dec-10', 'Jan-11', 'Feb-11', 'Mar-11', 'Apr-11', 'May-11', 'Jun-11', 

             'Jul-11', 'Aug-11', 'Sep-11', 'Oct-11', 'Nov-11', 'Dec-11']



plt.figure(figsize=(15,7))



sns.heatmap(data = round(return_rate,0),

            annot = True,

            cmap = ('Greens'),

            vmax = list(return_rate.max().sort_values(ascending = False))[1],

            yticklabels = mon_names,

            fmt = '.0f',

            linewidth = 0.01, square=True)



plt.title("Cohort Retention %")

plt.xlabel("------Retention Months ------->")

plt.ylabel("<------Cohort Months-------")
data.groupby('coh_month')['coh_index']
data.head()
coh_spends = pd.crosstab(data['coh_month'], data['coh_index'], values=data['Sales'], aggfunc= pd.Series.mean).round(0)



coh_spends
mon_names = ['Dec-10', 'Jan-11', 'Feb-11', 'Mar-11', 'Apr-11', 'May-11', 'Jun-11', 'Jul-11', 'Aug-11', 'Sep-11', 'Oct-11', 'Nov-11', 'Dec-11']



plt.figure(figsize=(15,7))







sns.heatmap(data = round(coh_spends,0),

            annot=True,

            cmap = ('Greys'),

            vmax=list(coh_spends.max().sort_values(ascending = False))[0], 

            yticklabels = mon_names,

            fmt = '.0f',

            linewidth = 0.01

           )



plt.title("Cohort Avg Spend")

plt.xlabel("------Avg Spend ------->")

plt.ylabel("<------Cohort Months-------")
data.info()
# Converting to proper date format

data['InvoiceDate'] = data['InvoiceDate'].dt.date



# Creating a column of latest Purchase date, max of invoice date for each customer.

cus_invoices = data.groupby('CustomerID')['InvoiceDate']

data['latest_date'] = cus_invoices.transform('max')



# Days since the last purchase

data['recency'] = tdy - data['latest_date']



# dropping the column latest_date 

data.drop(['latest_date'], axis=1, inplace= True)



rfm_table = data.groupby(['CustomerID', 'recency']).agg({'InvoiceNo':'count','Sales':sum})

rfm_table = rfm_table.reset_index()

rfm_table = rfm_table.sort_values(by=['Sales', 'recency'], ascending= [False,True])

rfm_table.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

rfm_table.head()
rfm_table.describe()
rfm_table['Rec_grd'] = pd.qcut(rfm_table['Recency'], q = 5, labels=np.arange(5,0,-1))

rfm_table['Frq_grd'] = pd.qcut(rfm_table['Frequency'], q = 5, labels=np.arange(1,6))

rfm_table['Mon_grd'] = pd.qcut(rfm_table['Monetary'], q = 5, labels=np.arange(1,6))



rfm_table.sample(10)
rfm_table['MRF_score'] = (rfm_table['Mon_grd'].astype('str')) + (rfm_table['Frq_grd'].astype('str')) + (rfm_table['Rec_grd'].astype('str'))



rfm_table['MRF_score']



rfm_table.head(20)
rfm_table.to_excel('rfm_knime_export.xlsx')
a = rfm_table.groupby(['MRF_score'])['CustomerID'].agg({'CustomerID': 'count', 'Monetary':sum})



a.reset_index()
a = a.sort_values(by=['Monetary'], ascending= [False])



a.head(20)
from sklearn.cluster import KMeans



model = KMeans(7)

model.fit(rfm_table[['Mon_grd', 'Frq_grd', 'Rec_grd']])

y_clusters = model.predict(rfm_table[['Mon_grd', 'Frq_grd', 'Rec_grd']])



print(y_clusters[:5])



rfm_table['cluster'] = y_clusters
type(y_clusters)
rfm_clusters = rfm_table[['Mon_grd', 'Frq_grd', 'Rec_grd', 'cluster']]
unique_elements, counts_elements = np.unique(y_clusters, return_counts=True)

print("Frequency of unique values of the said array:")

print(np.asarray((unique_elements, counts_elements)))
import plotly.express as px

fig = px.scatter_3d(rfm_table, x="Recency", z="Frequency", y="Monetary", color="cluster")

fig.show()
fig = px.scatter_3d(rfm_table, x="Rec_grd", y="Frq_grd", z="Mon_grd", color="cluster")

fig.show()



# Rec_grd 	Frq_grd 	Mon_grd
df = rfm_table[['Recency', 'Frequency', 'Monetary', 'cluster']]

df.head()

from sklearn.preprocessing import MinMaxScaler, StandardScaler



for i in ['Recency', 'Frequency', 'Monetary']:

    scaler = StandardScaler()

    df[i] = scaler.fit_transform(df[[i]])

    

df.head()
fig = px.scatter_3d(df, x="Recency", y="Frequency", z="Monetary", color="cluster")

fig.show()
rfm_clusters = rfm_table[['Recency', 'Frequency', 'Monetary', 'cluster']]

rfm_clusters.groupby('cluster').mean()
rfm_table[['Recency', 'Frequency', 'Monetary', 'cluster']].head()