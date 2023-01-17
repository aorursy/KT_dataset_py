import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm 
data =pd.read_csv("../input/Dataset_spine.csv")
data.head()
del data['Unnamed: 13']
data.info()
data.describe()
data.isnull().any()
data.shape
correlation = data.corr()
plt.figure(figsize=(15,10))
sns.heatmap(correlation,annot=True)
plt.show()
data[data['Class_att']=='Abnormal'].shape[0]
data[data['Class_att']=='Normal'].shape[0]
plt.figure(figsize=(15,10))
data.boxplot(patch_artist=True)
plt.show()
data.drop(data[data['Col6']>400].index,inplace=True)
data.reset_index(inplace=True)
data.shape
data['Class_att']=data['Class_att'].apply(lambda x : '1' if x=='Abnormal' else '0')
data_feature = data[data.columns.difference(['Class_att'])]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_feature)
data_scaled= pd.DataFrame(data=scaled_data,columns=data_feature.columns)
data_scaled['Class_att']=data['Class_att']
data_scaled.describe()
X=data_scaled[data_scaled.columns.difference(['Class_att'])]
y=data_scaled['Class_att']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
neighbor = KNeighborsClassifier(n_neighbors=3)
neighbor.fit(X_train,y_train)
neighbor.score(X_test,y_test)
y_predict = neighbor.predict(X_test)
confusion_matrix(y_test,y_predict)
Svm=svm.SVC()
Svm.fit(X_train,y_train)
Svm.score(X_test,y_test)
y_predict = Svm.predict(X_test)
confusion_matrix(y_test,y_predict)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
logreg.score(X_test,y_test)
y_predict = logreg.predict(X_test)
confusion_matrix(y_test,y_predict)