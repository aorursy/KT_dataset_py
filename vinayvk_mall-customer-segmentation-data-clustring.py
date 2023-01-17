# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

base_dataset=pd.read_csv("../input/Mall_Customers.csv")

base_dataset.head(15)

base_dataset.columns
base_dataset.shape
def nullvalue_function(base_dataset,percentage):

    

    # Checking the null value occurance

    

    print(base_dataset.isna().sum())



    # Printing the shape of the data 

    

    print(base_dataset.shape)

    

    # Converting  into percentage table

    

    null_value_table=pd.DataFrame((base_dataset.isna().sum()/base_dataset.shape[0])*100).sort_values(0,ascending=False )

    

    null_value_table.columns=['null percentage']

    

    # Defining the threashold values 

    

    null_value_table[null_value_table['null percentage']>percentage].index

    

    # Drop the columns that has null values more than threashold 

    base_dataset.drop(null_value_table[null_value_table['null percentage']>30].index,axis=1,inplace=True)

    

    # Replace the null values with median() # continous variables 

    for i in base_dataset.describe().columns:

        base_dataset[i].fillna(base_dataset[i].median(),inplace=True)

    # Replace the null values with mode() #categorical variables

    for i in base_dataset.describe(include='object').columns:

        base_dataset[i].fillna(base_dataset[i].value_counts().index[0],inplace=True)

  

    print(base_dataset.shape)

    

    return base_dataset
base_dataset.describe()
base_dataset.isnull().sum()
def outliers(base_dataset):

    import numpy as np

    import statistics as sts



    for i in base_dataset.describe().columns:

        x=np.array(base_dataset[i])

        p=[]

        Q1 = base_dataset[i].quantile(0.25)

        Q3 = base_dataset[i].quantile(0.75)

        IQR = Q3 - Q1

        LTV= Q1 - (1.5 * IQR)

        UTV= Q3 + (1.5 * IQR)

        for j in x:

            if j <= LTV or j>=UTV:

                p.append(sts.median(x))

            else:

                p.append(j)

        base_dataset[i]=p

    return base_dataset
base_dataset_out=outliers(base_dataset)
base_dataset_out.head()
X= base_dataset_out.iloc[:, [3,4]].values

from sklearn.cluster import KMeans

wcss=[]

for i in range(1,11):

    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)



plt.plot(range(1,11), wcss)

plt.title('The Elbow Method')

plt.xlabel('no of clusters')

plt.ylabel('wcss')

plt.show()
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X[0:10], method = 'single'))

plt.title('Dendrogram')

plt.xlabel('Customers')

plt.ylabel('WCSS')

plt.show()
y= sns.clustermap(X,method='single')
#Model Build

kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)

y_kmeans= kmeansmodel.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()

