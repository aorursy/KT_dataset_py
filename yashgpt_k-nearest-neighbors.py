import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import json

from pandas.io.json import json_normalize

%matplotlib inline

df=pd.read_csv("../input/KNN_Project_Data.csv")
sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')
from sklearn.preprocessing import StandardScaler 
scaler=StandardScaler()
scaler.fit(df.drop(('TARGET CLASS'),axis=1))
scaler_feature=scaler.transform(df.drop(('TARGET CLASS'),axis=1))
df_f=pd.DataFrame(scaler_feature,columns=df.columns[:-1])

df_f.head()
from sklearn.model_selection import train_test_split
X=scaler_feature

y=df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1

                        )
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test,pred)
print(classification_report(y_test,pred))
error_rate=[]

for i in range(1,40):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i=knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.style.use('ggplot')

plt.plot(range(1,40),error_rate,color='blue',linestyle='--',marker='o',markerfacecolor='red',markersize=10)

plt.title("Error rate Vs K Value")

plt.ylabel('Error Rate')

plt.xlabel('K')
knn=KNeighborsClassifier(n_neighbors=22)

knn.fit(X_train,y_train)

pred_i=knn.predict(X_test)



print('WITH K=22')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))