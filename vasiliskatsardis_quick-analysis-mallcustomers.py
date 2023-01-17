import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

from sklearn.cluster import KMeans

import warnings

import os

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

df.head()
df.isna().sum()
df.dtypes
df.describe()
#We notice that we are lucky to have no missing data(small dataset anyway) , lets move on to visualize our data!
plt.rcParams['figure.figsize'] = (18, 10)



plt.subplot(1, 2, 1)

sns.set(style = 'whitegrid')

sns.distplot(df['Age'],color='black')

plt.title('Distribution of Age', fontsize = 20)

plt.xlabel('Range of Age')

plt.ylabel('Count')





plt.subplot(1, 2, 2)

sns.set(style = 'whitegrid')

sns.distplot(df['Annual Income (k$)'], color = 'green')

plt.title('Distribution of Annual Income (k$)', fontsize = 20)

plt.xlabel('Range of Annual Income')

plt.ylabel('Count')

plt.show()
#We note that most people in our data range from 20-50 years old with most customers being close to 35 years of age and most of our customers earn from 50-85k per annum.

#Let's further investigate our Ages and Annual Incomes to see if anything stands out and we have missed it.
plt.rcParams['figure.figsize'] = (18, 8)

sns.countplot(df['Age'],palette='hsv')

plt.title('Ages on a Scale',fontsize = 20)

plt.show()
plt.rcParams['figure.figsize']=(18,10)

sns.swarmplot(df['Annual Income (k$)'],palette='copper',size=10)

plt.title('Yearly Earning of Our Customers',fontsize=20)

plt.show()
plt.rcParams['figure.figsize']=(18,10)

sns.swarmplot(df['Annual Income (k$)'],df['Spending Score (1-100)'],palette='copper',size=10)

plt.title('Yearly Earning of Our Customers',fontsize=20)

plt.show()
new_metric = df['Annual Income (k$)']*df['Spending Score (1-100)']/100
sns.distplot(new_metric)
plt.rcParams['figure.figsize']=(18,10)

sns.swarmplot(x='Annual Income (k$)',y='Spending Score (1-100)',data=df,hue="Gender",size=10)

plt.title('Income & Spending According to Sex',fontsize=20)

plt.show()
#from sklearn.cluster import KMeans

#It is general practice to normalize our variables before clustering but we are not going to do that in this example as it is not needed.

x = df[['Annual Income (k$)','Spending Score (1-100)']].values
km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 200, n_init = 10, random_state = 0)
ykm = km.fit_predict(x)
#Now we have clustered our data. Let's see our centers!

km.cluster_centers_
#Each customer is classified as the number of cluster it is closer to : km.cluster_centers_[0] classifies close nodes as 0 etc..

#lets visualize our clusters !

plt.rcParams['figure.figsize']=(18,10)

plt.scatter(x[ykm == 0, 0], x[ykm == 0, 1], s = 100, c = 'yellow', label = 'Average')

plt.scatter(x[ykm == 1, 0], x[ykm == 1, 1], s = 100, c = 'red', label = 'Firework')

plt.scatter(x[ykm == 2, 0], x[ykm == 2, 1], s = 100, c = 'green', label = 'Target')

plt.scatter(x[ykm == 3, 0], x[ykm == 3, 1], s = 100, c = 'brown', label = 'Advertisers')

plt.scatter(x[ykm == 4, 0], x[ykm == 4, 1], s = 100, c = 'orange', label = 'Potential')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'Cluster Centers')



plt.style.use('fivethirtyeight')

plt.title('K Means Clustering', fontsize = 20)

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.legend()

plt.show()
df['Metric']=new_metric
X = df.iloc[:,[3,4,5]].values
km_2 = KMeans(n_clusters = 5,)
#Let's see how many clusters we should include now with another added feature

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)



plt.rcParams['figure.figsize'] = (15, 5)

plt.plot(range(1, 11), wcss)

plt.title('K-Means Clustering(The Elbow Method)', fontsize = 20)

plt.xlabel('Age')

plt.ylabel('Count')

plt.grid()

plt.show()
#Let's try with 2, seems like a good classification to start!

kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

ymeans = kmeans.fit_predict(X)



plt.rcParams['figure.figsize'] = (10, 10)

plt.title('Cluster of Ages', fontsize = 30)



plt.scatter(x[ymeans == 0, 0], x[ymeans == 0, 1], s = 100, c = 'pink', label = 'non Targets' )

plt.scatter(x[ymeans == 1, 0], x[ymeans == 1, 1], s = 100, c = 'orange', label = 'Targets')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'black')



plt.style.use('fivethirtyeight')

plt.xlabel('Annual Income')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.grid()

plt.show()
#We notice that we end up with the same results and this strongly suggests that we have to take these two groups into consideration, let's investigate further using our new feature.

df['Metric'].mean()
#We can easily classify if  a customer is in our Target group by checking only his metric! If it is above 50 he does belong in our Target Group . Therefore simply to find our Target Audience!

Target_Audience = df[df['Metric']>50]

Target_Audience
#Visualizing key features.

sns.countplot(Target_Audience['Age'])
the_rest = df[df['Metric']<=50]
sns.swarmplot(Target_Audience['Spending Score (1-100)'],color = 'green',size = 5)

sns.swarmplot(the_rest['Spending Score (1-100)'])