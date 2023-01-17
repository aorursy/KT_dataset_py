import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as dt

import warnings
warnings.simplefilter(action = 'ignore')
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);  # to display all columns and rows
pd.set_option('display.float_format', lambda x: '%.2f' % x) # The number of numbers that will be shown after the comma.

onret = pd.read_excel('../input/online-retail-ii-data-set-from-ml-repository/online_retail_II.xlsx', 
                      sheet_name = 'Year 2010-2011', sep=';')
onret.head()
onret.info()
df = onret.dropna() # Deleting missing values
df.info()
df.head(2)
df["Customer ID"] = df["Customer ID"].astype(int) # Values of float are converted to integer.
df["TotalPrice"] = df["Quantity"] * df["Price"]  # it is necessary to create a new variable by multiplying two variables
df.head()
df[df['Invoice'].str.startswith("C", na = False)]  # Values starting with 'c' in the invoice variable indicate the returned products.
df["InvoiceDate"].max()
today = df["InvoiceDate"].max()
today
temp_df = (today - df.groupby("Customer ID").agg({"InvoiceDate":"max"})) # Show the last shopping dates of each customer.
temp_df.head()
temp_df.rename(columns={"InvoiceDate":"Recency"},inplace=True)
recency_df = temp_df["Recency"].apply(lambda x: x.days)
recency_df.head()
recency_df = pd.DataFrame(recency_df)
recency_df.head()
temp_df = df.groupby(["Customer ID", "Invoice"]).agg({"Invoice": "count"})
freq_df = temp_df.groupby("Customer ID").agg({"Invoice":"count"})
freq_df.rename(columns={"Invoice":"Frequency"},inplace = True)
freq_df.head()
monetary_df = df.groupby("Customer ID").agg({"TotalPrice": "sum"})
monetary_df.rename(columns={"TotalPrice":"Monetary"},inplace=True)
monetary_df.head()
rfm_df = pd.concat([recency_df, freq_df, monetary_df], axis = 1)
rfm_df.head()
# Outlier Analysis of Amount Frequency and Recency

attributes = ['Monetary','Frequency','Recency']
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot(data = rfm_df[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Range", fontweight = 'bold')
plt.xlabel("Attributes", fontweight = 'bold')
# Removing (statistical) outliers for Monetary
Q1 = rfm_df.Monetary.quantile(0.05)
Q3 = rfm_df.Monetary.quantile(0.95)
IQR = Q3 - Q1
rfm_df = rfm_df[(rfm_df.Monetary >= Q1 - 1.5*IQR) & (rfm_df.Monetary <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Recency
Q1 = rfm_df.Recency.quantile(0.05)
Q3 = rfm_df.Recency.quantile(0.95)
IQR = Q3 - Q1
rfm_df = rfm_df[(rfm_df.Recency >= Q1 - 1.5*IQR) & (rfm_df.Recency <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Frequency
Q1 = rfm_df.Frequency.quantile(0.05)
Q3 = rfm_df.Frequency.quantile(0.95)
IQR = Q3 - Q1
rfm_df = rfm_df[(rfm_df.Frequency >= Q1 - 1.5*IQR) & (rfm_df.Frequency <= Q3 + 1.5*IQR)]
rfm_1 = rfm_df.copy()
rfm_df.head()
rfm_df["RecencyScore"] = pd.qcut(rfm_df['Recency'], 5, labels = [5, 4, 3, 2, 1])
rfm_df["FrequencyScore"] = pd.qcut(rfm_df['Frequency'].rank(method="first"), 5, labels = [1, 2, 3, 4, 5])
rfm_df["MonetaryScore"] = pd.qcut(rfm_df['Monetary'], 5, labels = [1, 2, 3, 4, 5])
rfm_df["RFM_SCORE"] = (rfm_df['RecencyScore'].astype(str) 
                                + rfm_df['FrequencyScore'].astype(str) 
                                + rfm_df['MonetaryScore'].astype(str))
rfm_df.head()
seg_map = {r'[1-2][1-2]': 'Hibernating',
            r'[1-2][3-4]': 'At Risk',
            r'[1-2]5': 'Can\'t Loose',
            r'3[1-2]': 'About to Sleep',
            r'33': 'Need Attention',
            r'[3-4][4-5]': 'Loyal Customers',
            r'41': 'Promising',
            r'51': 'New Customers',
            r'[4-5][2-3]': 'Potential Loyalists',
            r'5[4-5]': 'Champions'}
    
rfm_df['Segment'] = rfm_df['RecencyScore'].astype(str) + rfm_df['FrequencyScore'].astype(str)
rfm_df['Segment'] = rfm_df['Segment'].replace(seg_map, regex=True)
rfm_df.head()
rfm = rfm_1
rfm.head()
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler((0,1))
cols = rfm.columns
index = rfm.index
scaled_rfm = mms.fit_transform(rfm)
scaled_rfm = pd.DataFrame(scaled_rfm, columns=cols, index = index)
scaled_rfm.head()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4)
kmeans = kmeans.fit(scaled_rfm)
kmeans
kmeans.cluster_centers_
kmeans.labels_
ssd = []

K = range(1,30)

for k in K:
    kmeans = KMeans(n_clusters = k).fit(rfm)
    ssd.append(kmeans.inertia_)
    
plt.plot(K, ssd, "bx-")
plt.xlabel("Distance Residual Sums Versus Different k Values")
plt.title("Elbow method for Optimum number of clusters")
from yellowbrick.cluster import KElbowVisualizer

kmeans = KMeans()
visu = KElbowVisualizer(kmeans, k = (2,20))
visu.fit(rfm)
visu.poof();
# Final model with k=6
kmeans = KMeans(n_clusters = 5, max_iter=50).fit(rfm)
kmeans.labels_
# assign the label
rfm['Cluster_Id'] = kmeans.labels_
# Box plot to visualize Cluster Id vs Monetary
sns.boxplot(x = 'Cluster_Id', y = 'Monetary', data = rfm);
# Box plot to visualize Cluster Id vs Frequency
sns.boxplot(x='Cluster_Id', y='Frequency', data = rfm);
# Box plot to visualize Cluster Id vs Recency
sns.boxplot(x='Cluster_Id', y='Recency', data = rfm);
rfm.groupby("Cluster_Id").agg({"Cluster_Id":"count"})
rfm.groupby("Cluster_Id").agg(np.mean)
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
hc_complete = linkage(rfm, "complete") #Complete Linkage
hc_average = linkage(rfm, "average") # Average Linkage

plt.figure(figsize = (15,10))
plt.title("Hierarchical Cluster Dendrogram")
plt.xlabel("Observation Unit")
plt.ylabel("Distance")
dendrogram(hc_complete,
           truncate_mode = "lastp",
           p = 10,
           show_contracted = True,
          leaf_font_size = 10);
# Cutting the Dendrogram based on K
cluster_labels = cut_tree(hc_complete, n_clusters = 5).reshape(-1, )
cluster_labels
# Assign cluster labels
rfm['Cluster_Labels'] = cluster_labels
rfm['Cluster_Labels'] = rfm['Cluster_Labels'] + 1
rfm.head()
rfm.groupby("Cluster_Labels").agg(np.mean)
# Plot Cluster Id vs Monetary
sns.boxplot(x = 'Cluster_Labels', y = 'Monetary', data = rfm);
# Plot Cluster Id vs Frequency
sns.boxplot(x = 'Cluster_Labels', y = 'Frequency', data = rfm);
# Plot Cluster Id vs Recency
sns.boxplot(x='Cluster_Labels', y='Recency', data=rfm);
