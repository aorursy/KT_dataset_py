import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.cluster import KMeans



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import zscore

import sklearn.metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



from sklearn.model_selection import train_test_split
df=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()
df['quality'].value_counts()
from pandas_profiling import ProfileReport
profile = ProfileReport(df)
profile
sns.pairplot(df,hue='quality')
df.describe()
df2=df.drop('quality',axis=1)
df2.head()
from scipy.stats import zscore
df_scaled=df2.apply(zscore)

df_scaled.head()
model=KMeans(n_clusters=2)
model
cluster_range=range(1,15)

cluster_error=[]

for a in cluster_range:

    cluster=KMeans(a,n_init=10)

    cluster.fit(df_scaled)

    cluster_error.append(cluster.inertia_)

cluster_df=pd.DataFrame({'num_cluster':cluster_range,'cluster_error':cluster_error})

cluster_df
plt.figure(figsize=(20,10))

plt.plot(cluster_df['num_cluster'],cluster_df['cluster_error'],marker='o')
kmeans=KMeans(n_clusters=7,n_init=15,random_state=3)
kmeans.fit(df_scaled)
centroids=kmeans.cluster_centers_

pd.DataFrame(centroids,columns=df2.columns)
df_scaled['class']=kmeans.labels_.astype('object')
df_scaled.head()
df_k=df_scaled


df_k['class']=df_k['class'].astype('object')
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6))

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=100)

kmeans.fit(df_scaled)

labels = kmeans.labels_

ax.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], df_scaled.iloc[:, 3],c=labels.astype(np.float), edgecolor='k')

ax.w_xaxis.set_ticklabels([])

ax.w_yaxis.set_ticklabels([])

ax.w_zaxis.set_ticklabels([])

ax.set_xlabel('Length')

ax.set_ylabel('Height')

ax.set_zlabel('Weight')

ax.set_title('3D plot of KMeans Clustering')
from scipy.cluster.hierarchy import linkage, dendrogram

plt.figure(figsize=[10,10])

merg = linkage(df2, method='ward')

dendrogram(merg, leaf_rotation=90)

plt.title('Dendrogram')

plt.xlabel('Data Points')

plt.ylabel('Euclidean Distances')

plt.show()
from sklearn.cluster import AgglomerativeClustering



hie_clus = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

cluster2 = hie_clus.fit_predict(df_scaled)



df_h = df_scaled.copy(deep=True)

df_h['class'] = cluster2
df_h
df_h['class']=df_h['class'].astype('object')
print('Original Data Classes:')

print(df['quality'].value_counts())

print('-' * 30)

print('K-Means Predicted Data Classes:')

print(df_k['class'].value_counts())

print('-' * 30)

print('Hierarchical Predicted Data Classes:')

print(df_h['class'].value_counts())
x= df_k.drop('class',axis=1)

y= pd.DataFrame(df_k['class'].astype('float64'))
test_size = 0.30 # taking 70:30 training and test set

seed = 7  # Random numbmer seeding for reapeatability of the code

x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=test_size, random_state=seed)
from sklearn.preprocessing import StandardScaler

independent_scalar = StandardScaler()

x_train = independent_scalar.fit_transform (x_train) #fit and transform

x_validate = independent_scalar.transform (x_validate) # only transform
y.info()
from sklearn.tree import DecisionTreeClassifier 

#DecisionTreeClassifier is the corresponding Classifier

Dtree = DecisionTreeClassifier(max_depth=3)

Dtree.fit(x_train, y_train)
predictValues_train = Dtree.predict(x_train)

#print(predictValues_train)

accuracy_train=accuracy_score(y_train, predictValues_train)







predictValues_validate = Dtree.predict(x_validate)

#print(predictValues_validate)

accuracy_validate=accuracy_score(y_validate, predictValues_validate)



print("Train Accuracy  :: ",accuracy_train)

print("Validation Accuracy  :: ",accuracy_validate)
print('Classification Report')

print(classification_report(y_validate, predictValues_validate))
RFclassifier = RandomForestClassifier(n_estimators = 100, random_state = 0,min_samples_split=5,criterion='gini',max_depth=5)

RFclassifier.fit(x_train, y_train)
predictValues_validate = RFclassifier.predict(x_validate)

#print(predictValues_validate)

accuracy_validate=accuracy_score(y_validate, predictValues_validate)







predictValues_train = RFclassifier.predict(x_train)

#print(predictValues_train)

accuracy_train=accuracy_score(y_train, predictValues_train)





print("Train Accuracy  :: ",accuracy_train)

print("Validation Accuracy  :: ",accuracy_validate)
print('Classification Report')

print(classification_report(y_validate, predictValues_validate))
from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import zscore
KNN = KNeighborsClassifier(n_neighbors= 8 , weights = 'uniform', metric='euclidean')

KNN.fit(x_train, y_train)
predictValues_train = KNN.predict(x_train)

print(predictValues_train)

accuracy_train=accuracy_score(y_train, predictValues_train)

print("Train Accuracy  :: ",accuracy_train)
predictValues_validate = KNN.predict(x_validate)

print(predictValues_validate)

accuracy_validate=accuracy_score(y_validate, predictValues_validate)

print("Validation Accuracy  :: ",accuracy_validate)
x= df_h.drop('class',axis=1)

y= pd.DataFrame(df_h['class'].astype('float64'))
test_size = 0.30 # taking 70:30 training and test set

seed = 7  # Random numbmer seeding for reapeatability of the code

x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=test_size, random_state=seed)
from sklearn.preprocessing import StandardScaler

independent_scalar = StandardScaler()

x_train = independent_scalar.fit_transform (x_train) #fit and transform

x_validate = independent_scalar.transform (x_validate) # only transform
from sklearn.tree import DecisionTreeClassifier 

#DecisionTreeClassifier is the corresponding Classifier

Dtree = DecisionTreeClassifier(max_depth=3)

Dtree.fit (x_train, y_train)
predictValues_train = Dtree.predict(x_train)

#print(predictValues_train)

accuracy_train=accuracy_score(y_train, predictValues_train)







predictValues_validate = Dtree.predict(x_validate)

#print(predictValues_validate)

accuracy_validate=accuracy_score(y_validate, predictValues_validate)



print("Train Accuracy  :: ",accuracy_train)

print("Validation Accuracy  :: ",accuracy_validate)
print('Classification Report')

print(classification_report(y_validate, predictValues_validate))
RFclassifier = RandomForestClassifier(n_estimators = 100, random_state = 0,min_samples_split=5,criterion='gini',max_depth=5)

RFclassifier.fit(x_train, y_train)
predictValues_validate = RFclassifier.predict(x_validate)

#print(predictValues_validate)

accuracy_validate=accuracy_score(y_validate, predictValues_validate)







predictValues_train = RFclassifier.predict(x_train)

#print(predictValues_train)

accuracy_train=accuracy_score(y_train, predictValues_train)





print("Train Accuracy  :: ",accuracy_train)

print("Validation Accuracy  :: ",accuracy_validate)
RFclassifier = RandomForestClassifier(n_estimators = 11, random_state = 0,min_samples_split=5,criterion='gini',max_depth=5)

RFclassifier.fit(x_train, y_train)
print('Classification Report')

print(classification_report(y_validate, predictValues_validate))
KNN = KNeighborsClassifier(n_neighbors= 8 , weights = 'uniform', metric='euclidean')

KNN.fit(x_train, y_train)
predictValues_train = KNN.predict(x_train)

print(predictValues_train)

accuracy_train=accuracy_score(y_train, predictValues_train)

print("Train Accuracy  :: ",accuracy_train)
predictValues_validate = KNN.predict(x_validate)

print(predictValues_validate)

accuracy_validate=accuracy_score(y_validate, predictValues_validate)

print("Validation Accuracy  :: ",accuracy_validate)