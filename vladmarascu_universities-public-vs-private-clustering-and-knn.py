import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
uni = pd.read_csv('../input/college-data/data.csv')
uni.head()
uni.info()
uni.describe()
plt.figure(figsize=(15,3))

sns.heatmap(uni.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# SEABORN SETTINGS

sns.set() # DEFAULTS

sns.set_style('whitegrid')

sns.set_palette("coolwarm")
sns.lmplot('room_board','grad_rate',data=uni, hue='private',palette='coolwarm')
sns.lmplot('f_undergrad','outstate',data=uni, hue='private',palette='coolwarm')
uni_priv=uni[uni['private']=='Yes']

uni_public=uni[uni['private']=='No']
plt.figure(figsize=(13,6))

sns.distplot(uni_priv['outstate'],bins=20, kde=False,color='red',label='Private')

sns.distplot(uni_public['outstate'],bins=20, kde=False,color='blue',label='Public')

plt.legend()
plt.figure(figsize=(13,6))

sns.distplot(uni_priv['grad_rate'],bins=20, kde=False,color='red',label='Private')

sns.distplot(uni_public['grad_rate'],bins=20, kde=False,color='blue',label='Public')

plt.legend()
uni[uni['grad_rate'] > 100]
uni['grad_rate'].iloc[95]
uni['grad_rate'].iloc[95] = 100
uni_priv=uni[uni['private']=='Yes']

uni_public=uni[uni['private']=='No']



plt.figure(figsize=(13,6))

sns.distplot(uni_priv['grad_rate'],bins=20, kde=False,color='red',label='Private')

sns.distplot(uni_public['grad_rate'],bins=20, kde=False,color='blue',label='Public')

plt.legend()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)

X=uni.drop('private', axis=1)

kmeans.fit(X)
kmeans.cluster_centers_
kmeans.labels_
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

uni['Cluster'] = labelencoder.fit_transform(uni['private'])
uni.head(5)
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(uni['Cluster'],kmeans.labels_))

print(classification_report(uni['Cluster'],kmeans.labels_))
uni = pd.read_csv('../input/college-data/data.csv')

uni.head()
uni['private']=uni['private'].astype('category').cat.codes

uni['private']
uni.head()
X=uni.drop('private',axis=1)

y=uni['private'] # OUR TARGET LABEL for predictions
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(X)

X_std=scaler.transform(X) # numpy array here



#Create new X dataframe

X_2=pd.DataFrame(X_std, columns=uni.drop('private',axis=1).columns)

X_2.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size=0.33, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train, y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
error_rate = []

scores = []



for i in range(1,40): # check all values of K between 1 and 40

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    score=accuracy_score(y_test,pred_i)

    scores.append(score)

    error_rate.append(np.mean(pred_i != y_test)) # ERROR RATE DEF and add it to the list
plt.style.use('seaborn-whitegrid')

plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
plt.style.use('seaborn-whitegrid')

plt.figure(figsize=(10,6))

plt.plot(range(1,40),scores,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Accuracy Score vs. K Value')

plt.xlabel('K')

plt.ylabel('Accuracy Score')
knn=KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train,y_train)
pred_7 = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

print(confusion_matrix(y_test,pred_7))

print('\n')

print(classification_report(y_test,pred_7))