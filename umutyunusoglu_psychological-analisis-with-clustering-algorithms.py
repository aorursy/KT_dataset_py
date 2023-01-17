import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from yellowbrick.cluster import KElbowVisualizer



import os



print(os.listdir("../input"))
dataset=pd.read_excel("../input/dataset.xlsx")

dataset #this is our data 
dataset.isnull().sum() #Here we see if there are empty answers.
dataset.describe().T # Statistical properties of numeric data.
dataset.hist(figsize=(10,10),layout=(3,1)); #Histograms of our numeric data.
#number of unique values for each column

for col in dataset:

    print(len(dataset[col].unique()))
#Here we seperate our structural data from numeric data.

structural1=dataset.iloc[:,0:5]

structural2=dataset.iloc[:,8:]

numeric_data=dataset.iloc[:,5:8]

structural_data=pd.concat([structural1,structural2],axis=1)

structural_data.head()#first 5 rows of our structural data
numeric_data.head()#first 5 rows of our numeric data
#here we convert our structural data into numeric data

le=LabelEncoder()



for col in structural_data:

    structural_data[col]=le.fit_transform(structural_data[col])

structural_data
#Finally we merge our processed structural data with our numeric data.

final_data=pd.concat([structural_data,numeric_data],axis=1)

final_data
pca = PCA(n_components=2)



reduced_data=pca.fit_transform(final_data)



plt.scatter(reduced_data[:,0],reduced_data[:,1])

plt.show()

kmeans=KMeans() #this is our clustering algorithm ->KMeans



#We will determine our optimal number of clusters with elbow method from both our actual data and reduced data.



visualiser=KElbowVisualizer(kmeans,k=(2,10))

visualiser.fit(final_data)

visualiser.poof()



kmeans=KMeans()

visualiser=KElbowVisualizer(kmeans,k=(2,10))

visualiser.fit(reduced_data)

visualiser.poof()
algorithm=KMeans(n_clusters=4)#our algorithm

k_fit=algorithm.fit(final_data)



clusters=k_fit.labels_

clusters #here are our clusters
plt.scatter(reduced_data[:,0],reduced_data[:,1],c=clusters,cmap="viridis")#we visualise the results

plt.show()
final_data["cluster"]=clusters+1

final_data.head() #first 5 values of final data
final_data[final_data["cluster"]==1].head() #first 5 values of cluster 1
final_data[final_data["cluster"]==1].describe().T # Statistical properties of cluster 1
final_data[final_data["cluster"]==2].head() #first 5 values of cluster 2
final_data[final_data["cluster"]==2].describe().T # Statistical properties of cluster 2
final_data[final_data["cluster"]==3].head() #first 5 values of cluster 3
final_data[final_data["cluster"]==3].describe().T # Statistical properties of cluster 3
final_data[final_data["cluster"]==4].head() #first 5 values of cluster 4
final_data[final_data["cluster"]==4].describe().T # Statistical properties of cluster 4