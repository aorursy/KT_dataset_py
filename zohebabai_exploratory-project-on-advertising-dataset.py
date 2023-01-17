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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
ad= pd.read_csv('../input/advertising.csv')
ad.head()
ad.info()
sns.set_style('white')
sns.set_context('notebook')
#Lets see a summary with respect to clicked on ad
sns.pairplot(ad, hue='Clicked on Ad', palette='bwr')
#Lets see Click on Ad features based on Sex
plt.figure(figsize=(10,6))
sns.countplot(x='Clicked on Ad',data=ad,hue='Male',palette='coolwarm')
#Distribution of top 12 country's ad clicks based on Sex 
plt.figure(figsize=(15,6))
sns.countplot(x='Country',data=ad[ad['Clicked on Ad']==1],order=ad[ad['Clicked on Ad']==1]['Country'].value_counts().index[:12],hue='Male',
              palette='viridis')
plt.title('Ad clicked country distribution')
plt.tight_layout()
#We will change the datetime object
ad['Timestamp']=pd.to_datetime(ad['Timestamp'])
#Now we shall introduce new columns Hour,Day of Week, Date, Month from timestamp
ad['Hour']=ad['Timestamp'].apply(lambda time : time.hour)
ad['DayofWeek'] = ad['Timestamp'].apply(lambda time : time.dayofweek)
ad['Month'] = ad['Timestamp'].apply(lambda time : time.month)
ad['Date'] = ad['Timestamp'].apply(lambda t : t.date())
#Hourly distribution of ad clicks
plt.figure(figsize=(15,6))
sns.countplot(x='Hour',data=ad[ad['Clicked on Ad']==1],hue='Male',palette='rainbow')
plt.title('Ad clicked hourly distribution')
#Daily distribution of ad clicks
plt.figure(figsize=(15,6))
sns.countplot(x='DayofWeek',data=ad[ad['Clicked on Ad']==1],hue='Male',palette='rainbow')
plt.title('Ad clicked daily distribution')
#Monthly distribution of ad clicks
plt.figure(figsize=(15,6))
sns.countplot(x='Month',data=ad[ad['Clicked on Ad']==1],hue='Male',palette='rainbow')
plt.title('Ad clicked monthly distribution')
#Now we shall group by date and see the
plt.figure(figsize=(15,6))
ad[ad['Clicked on Ad']==1].groupby('Date').count()['Clicked on Ad'].plot()
plt.title('Date wise distribution of Ad clicks')
plt.tight_layout()
#Top Ad clicked on specific date
ad[ad['Clicked on Ad']==1]['Date'].value_counts().head(5)
ad['Ad Topic Line'].nunique()
#Lets see Age distribution
plt.figure(figsize=(10,6))
sns.distplot(ad['Age'],kde=False,bins=40)
#Lets see Age distribution
plt.figure(figsize=(10,6))
sns.swarmplot(x=ad['Clicked on Ad'],y= ad['Age'],data=ad,palette='coolwarm')
plt.title('Age wise distribution of Ad clicks')
#Lets see Daily internet usage and daily time spent on site based on age
fig, axes = plt.subplots(figsize=(10, 6))
ax = sns.kdeplot(ad['Daily Time Spent on Site'], ad['Age'], cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(ad['Daily Internet Usage'],ad['Age'] ,cmap="Blues", shade=True, shade_lowest=False)
ax.set_xlabel('Time')
ax.text(20, 20, "Daily Time Spent on Site", size=16, color='r')
ax.text(200, 60, "Daily Internet Usage", size=16, color='b')
#Lets see the distribution who clicked on Ad based on area income of sex 
plt.figure(figsize=(10,6))
sns.violinplot(x=ad['Male'],y=ad['Area Income'],data=ad,palette='viridis',hue='Clicked on Ad')
plt.title('Clicked on Ad distribution based on area distribution')
#Lets take country value as dummies
country= pd.get_dummies(ad['Country'],drop_first=True)
#Now lets drop the columns not required for building a model
ad.drop(['Ad Topic Line','City','Country','Timestamp','Date'],axis=1,inplace=True)
#Now lets join the dummy values
ad = pd.concat([ad,country],axis=1)
from sklearn.model_selection import train_test_split
X= ad.drop('Clicked on Ad',axis=1)
y= ad['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel= LogisticRegression()
logmodel.fit(X_train,y_train)
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1,10,100,1000]}
grid_log= GridSearchCV(LogisticRegression(),param_grid,refit=True, verbose=2)
grid_log.fit(X_train,y_train)
grid_log.best_estimator_
pred_log= grid_log.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(confusion_matrix(y_test,pred_log))
print(classification_report(y_test,pred_log))
from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
pred_rfc= rfc.predict(X_test)
print(confusion_matrix(y_test,pred_rfc))
print(classification_report(y_test,pred_rfc))
from sklearn.svm import SVC
svc= SVC(gamma='scale')
svc.fit(X_train,y_train)
param_grid = {'C': [0.1,1,10,100,1000,5000]}
grid_svc= GridSearchCV(SVC(gamma='scale',probability=True),param_grid,refit=True,verbose=2)
grid_svc.fit(X_train,y_train)
grid_svc.best_estimator_
pred_svc= grid_svc.predict(X_test)
print(confusion_matrix(y_test,pred_svc))
print(classification_report(y_test,pred_svc))
from sklearn.ensemble import VotingClassifier
vote= VotingClassifier(estimators=[('logmodel',grid_log),('rfc',rfc),('svc',grid_svc)],voting='soft')
vote.fit(X_train,y_train)
pred_vote= vote.predict(X_test)
print(confusion_matrix(y_test,pred_vote))
print(classification_report(y_test,pred_vote))
#let's first scale the variables
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(ad.drop('Clicked on Ad',axis=1))
scaled_features= scaler.transform(ad.drop('Clicked on Ad',axis=1))
#Changing it from numpy array to pandas dataframe
train_scaled = pd.DataFrame(scaled_features,columns=ad.columns.drop('Clicked on Ad'))
train_scaled.head()
X_train, X_test, y_train, y_test = train_test_split(train_scaled,ad['Clicked on Ad'],test_size=0.20,random_state=101)
from sklearn.neighbors import KNeighborsClassifier
error_rate=[]

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K-value')
plt.xlabel('K')
plt.ylabel('Error Rate')
knn= KNeighborsClassifier(n_neighbors=40)
knn.fit(X_train,y_train)
pred_knn=knn.predict(X_test)
print(confusion_matrix(y_test,pred_knn))
print(classification_report(y_test,pred_knn))
