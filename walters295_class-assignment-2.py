# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data_path = '../input/credit-card-data.csv' # Path to data file
data = pd.read_csv(data_path) 
data.head(15)
# What columns are in the data set ? Do they have spaces that I should consider
data.columns
data.describe()

data['tenure'].value_counts().plot(kind='bar')
# Let's view the distribution of the data, where is it possible to find groups?
# We are using boxplots of all the columns except the first (cust_id which is a string)
for col in data.columns[2:]:
    data[col].plot(kind='box')
    plt.title('Box Plot for '+col)
    plt.show()
cluster_data = data[['purchases','payments']]
cluster_data.head()
cluster_data.plot(kind='scatter',x='purchases',y='payments')
# Is there any missing data
missing_data_results = cluster_data.isnull().sum()
print(missing_data_results)

# perform imputation with median values
# not require since none missing
#cluster_data = cluster_data.fillna( data.median() )
#retrieve just the values for all columns except customer id
data_values = cluster_data.iloc[ :, :].values
data_values
#import KMeans algorithm
from sklearn.cluster import KMeans
# Use the Elbow method to find a good number of clusters using WCSS (within-cluster sums of squares)
wcss = []
for i in range( 1, 15 ):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300) 
    kmeans.fit_predict( data_values )
    wcss.append( kmeans.inertia_ )
    
plt.plot( wcss, 'ro-', label="WCSS")
plt.title("Computing WCSS for KMeans++")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=300) 
cluster_data["cluster"] = kmeans.fit_predict( data_values )
cluster_data
cluster_data['cluster'].value_counts()
cluster_data['cluster'].value_counts().plot(kind='bar',title='Distribution of Customers across groups')
sns.pairplot( cluster_data, hue="cluster")
grouped_cluster_data = cluster_data.groupby('cluster')
grouped_cluster_data
grouped_cluster_data.describe()
grouped_cluster_data.plot(subplots=True,)
kmeans.predict(pd.DataFrame({'purchases':[300],'payments':[40000]}))
kmeans.predict([[3,2]])
kmeans.algorithm
kmeans.cluster_centers_
kmeans
data
tenureGroup = data.groupby('tenure')
tenureGroup
tenureGroup.describe()
tenureGroup.cust_id.count()
tenureGroup.cust_id.count().plot(kind="bar", title="Bar grapgh showing the number of customers and their respective tenure")
data.purchases.describe()
notMuchSpent = data[data.purchases < 5000].cust_id.count()
alotSpent = data[data.purchases > 5000].cust_id.count()
avgSpent = np.mean(data.purchases)
objects = ('Average Spent','Low Spenders', 'Big Spenders')
y_pos = np.arange(len(objects))
performance = [avgSpent,notMuchSpent,alotSpent]
plt.bar(y_pos, performance, align='center')
plt.xticks(y_pos, objects)
plt.ylabel('Amount of Persons Spending')
plt.title('Grapgh showing the amount of customers spending alot of money (actively using credit card)')
data.columns
ScatterData = data[['credit_limit','purchases']]
ScatterData.head()
ScatterData.corr()
ScatterData.plot(kind="scatter", x= 'purchases' , y='credit_limit',c="R")
ScatterData[ScatterData.purchases > 45000].purchases.count()
ScatterData[ScatterData.purchases > 45000].purchases
cluster_data.head()
grouped_cluster_data.head()
cluster_data['cluster'].value_counts().plot(kind='bar',title='Distribution of Customers across groups')
cluster_data['cluster'].value_counts()
cluster_data.head()
cluster_data['tenure'] = data['tenure']
cluster_data.corr()
sns.heatmap(cluster_data.corr())
#PairPlot
sns.pairplot(cluster_data)
cluster_data2 = data[['tenure','credit_limit']]
cluster_data2.head()
cluster_data2.plot(kind="scatter", x="tenure", y="credit_limit")
# Is there any missing data
missing_data_results = cluster_data2.isnull().sum()
print(missing_data_results)
cluster_data2 = cluster_data2.fillna( data.median() )
#retrieve just the values for all columns except customer id
data_values_2 = cluster_data2.iloc[ :, :].values
data_values_2
# Use the Elbow method to find a good number of clusters using WCSS (within-cluster sums of squares)
wcss = []
for i in range( 1, 15 ):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300) 
    kmeans.fit_predict( data_values_2 )
    wcss.append( kmeans.inertia_ )
    
plt.plot( wcss, 'ro-', label="WCSS")
plt.title("Computing WCSS for KMeans++")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300) 
cluster_data2["cluster"] = kmeans.fit_predict( data_values_2 )
cluster_data2
cluster_data2['cluster'].value_counts()
cluster_data2.describe()
cluster_data2['cluster'].value_counts().plot(kind='pie',title='Distribution of Customers across groups')
cluster_data2['cluster'].value_counts().plot(kind='bar',title='Distribution of Customers across groups')
sns.pairplot( cluster_data2, hue="cluster")
grouped_cluster_data2 = cluster_data2.groupby('cluster')
grouped_cluster_data2
grouped_cluster_data2.describe()
grouped_cluster_data2.plot(subplots=True,)

kmeans.predict(pd.DataFrame({'tenure':[12],'credit_limit':[40000]}))
kmeans.predict([[3,2]])
kmeans.algorithm
kmeans.cluster_centers_
kmeans
