import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers_df = pd.read_csv('../input/customers-seg/Mall_Customers.csv')
customers_df.head()
customers_df.describe()
plt.figure(figsize=(8,5))
sns.countplot(x='Gender',data=customers_df)
plt.show()
plt.figure(figsize=(8,5))
customers_df.Gender.value_counts().plot.pie( autopct='%.2f%%')
plt.show()
customers_df.Age.describe()
plt.figure(figsize=(8,5))
customers_df.Age.plot.hist(bins=15,alpha=.8)
plt.show()
sns.boxplot(x='Age',data=customers_df)
customers_df['Annual Income (k$)'].describe()
plt.figure(figsize=(8,5))
customers_df['Annual Income (k$)'].plot.hist(bins=14,color="#660033")
plt.show()
plt.figure(figsize=(8,5))
sns.kdeplot(customers_df['Annual Income (k$)'], color="blue", shade=True)
plt.show()
customers_df['Spending Score (1-100)'].describe()
customers_df['Spending Score (1-100)'].plot.box()
plt.show()
sns.distplot(customers_df['Spending Score (1-100)'])
plt.show()
customers_df['Gender'] = customers_df.Gender.map({'Male':1,'Female':0})
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

l=[]
X = customers_df.iloc[:,1:]
for n_clusters in range(2,11):
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(15, 7)


    ax1.set_xlim([-.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    l.append(silhouette_avg)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    

    plt.suptitle(("Silhouette analysis for KMeans clustering on Customers data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    
clusters = list(range(2,11))
plt.plot(clusters,l,'-bo')
axes = plt.axes()
axes.set_xlabel("Number of clusters k")
axes.set_ylabel("Average silhouette Width")
plt.show()
from sklearn.decomposition import PCA
reduced = PCA(n_components=2).fit_transform(X)

kmeans = KMeans(init='k-means++', n_clusters=6, random_state=10)
kmeans.fit(reduced)


pca_df = pd.DataFrame(reduced,columns=['Component1','Component2'])
pca_df['Segment'] = kmeans.labels_
pca_df.head(10)
plt.figure(figsize=(12,7))
sns.scatterplot(x='Component1',y='Component2',data=pca_df,hue='Segment',palette='bright')
plt.title('K-means clustering on the customers segmentation dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.legend()
plt.show()
