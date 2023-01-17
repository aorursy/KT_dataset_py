import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('../input/dataset-of-letter-predictiontraintest/train.csv')
data.head()
data.isnull().sum()
data['letter'].value_counts()
data['letter'].value_counts().plot.bar()
plt.plot()
X=data.iloc[:,1:-1]
y=data.iloc[:,0]
print(X.head())
print(y.head())
X.shape,y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=245,criterion='entropy',random_state=0,min_samples_split=2)

rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)

from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test,y_pred)
print('Accuracy is :',ac*100)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm='brute',n_neighbors =1 ,leaf_size=100,p=30)
knn.fit(X_train, y_train)

knn_predictions = knn.predict(X_test) 

acc=accuracy_score(y_test,knn_predictions)
print('Accuracy is :',acc*100)
from xgboost import XGBClassifier
model = XGBClassifier(learning_rate=1.0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1E01,tol=0.1)
model.fit(X_train, y_train)
predicted= model.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
accuracy = accuracy_score(y_test, predicted)
print("Accuracy: %.2f%%" % (accuracy * 100.0))#97.28
df1=pd.read_csv('../input/dataset-of-letter-predictiontraintest/test.csv')
df1.head()
X1=df1.iloc[:,:-1]
print(X1.head())
X1.shape
rfc.fit(X,y)
y_pred=rfc.predict(X1)
y_pred
submission = pd.DataFrame(y_pred,index=df1.id,columns=['letter'])
submission.to_csv('submissionr.csv')
model.fit(X,y)
y_pred=model.predict(X1)
submission = pd.DataFrame(y_pred,index=df1.id,columns=['letter'])
submission.to_csv('submissionsvm.csv')

