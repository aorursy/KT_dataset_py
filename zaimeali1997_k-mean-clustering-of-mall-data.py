import numpy as np 

import pandas as pd 

from matplotlib import pyplot as plt
data = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
data.head()
print('Length of DataSet: ' + str(len(data)))
print('Shape of Dataset: ', data.shape)
print("Dataset Summary: ")

print(data.info()) 

# no null value
data.describe()
# here we having two two indexes one that pandas assigned to a dataset and another index is of dataset which is CustomerID
data.CustomerID.value_counts()
data.CustomerID.unique()
data.index.name = 'Customer ID'

data.index = data.iloc[:, 0]
data.drop(axis=1, columns='CustomerID', inplace=True)
data.head()
print("Gender Count:") 

print(data['Gender'].value_counts())
import seaborn as sns



sns.countplot(data.Gender)
plt.hist(data.Age)

plt.title('Age Frequency')

plt.xlabel('Age')

plt.ylabel('Count')

plt.show()
sns.pairplot(data[['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']])
data.columns = ['Gender', 'Age', 'Annual_Income', 'Score']
data.head()
from sklearn.cluster import KMeans



X = data.loc[:, ['Annual_Income', 'Score']].values
X
n_clusters = range(1, 11)

inertia = []

predict = []



for i in n_clusters:

    KM = KMeans(n_clusters=i, init='k-means++', random_state=42)

    KM.fit(X)

    inertia.append(KM.inertia_)

    predict.append(KM.predict(X))
plt.plot(range(1, 11), inertia)

plt.title('The Elbow Method')

plt.xlabel('No. of Clusters')

plt.ylabel('Inertia')

plt.grid()

plt.show()
# here we can see Elbow comes at 5

inertia
KM = KMeans(n_clusters=5, init='k-means++', random_state=42)

pred_KM = KM.fit_predict(X)
pred_KM
len(pred_KM)
np.unique(pred_KM)
# so there are 5 clusters

data['Predict_Cluster'] = pred_KM
data.head(15)
plt.figure(1, figsize=(18, 10))

plt.scatter(

    X[pred_KM == 0, 0],

    X[pred_KM == 0, 1],

    s=100,

    c='red',

    label='Cluster1'

)

plt.scatter(

    X[pred_KM == 1, 0],

    X[pred_KM == 1, 1],

    s=100,

    c='blue',

    label='Cluster2'

)

plt.scatter(

    X[pred_KM == 2, 0],

    X[pred_KM == 2, 1],

    s=100,

    c='green',

    label='Cluster3'

)

plt.scatter(

    X[pred_KM == 3, 0],

    X[pred_KM == 3, 1],

    s=100,

    c='yellow',

    label='Cluster4'

)

plt.scatter(

    X[pred_KM == 4, 0],

    X[pred_KM == 4, 1],

    s=100,

    c='violet',

    label='Cluster5'

)

plt.scatter(

    KM.cluster_centers_[:, 0],

    KM.cluster_centers_[:, 1],

    s=200, #size

    c='black',

    label='Centroid'

)

plt.title('Clustering of Customers based on their Annual Income and Spending Score')

plt.xlabel('Annual Income in K$')

plt.ylabel('Spending Score out of 100')

plt.legend()

plt.show()
# Cluster 1(Red): The annual income of these people are low but their spending are high

# Cluster 2(Blue): The annual income of these people are normal and their spending are normal too, not high or low

# Cluster 3(Green): These are the people who are rich

# Cluster 4(Yellow): There are annual income and spending both are low

# Cluster 5(Violet): There Income are high but their spending score are low