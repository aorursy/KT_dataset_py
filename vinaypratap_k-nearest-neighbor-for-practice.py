import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/Classified Data",index_col=0)
df.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()
sns.pairplot(df,hue='TARGET CLASS')
df['TARGET CLASS'].value_counts()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df_feat,df['TARGET CLASS'],test_size=0.3)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import cross_val_score
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
accuracy_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)

    score = cross_val_score(knn,df_feat,df['TARGET CLASS'],cv=10)

    accuracy_rate.append(score.mean())  
error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)

    score = cross_val_score(knn,df_feat,df['TARGET CLASS'],cv=10)

    error_rate.append(1-score.mean())  

error_rate
error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

error_rate
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

        markerfacecolor='red', markersize=10)

#plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',

#         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
# NOW WITH K=23

knn = KNeighborsClassifier(n_neighbors=23)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=23')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))