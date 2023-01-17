#importing all the necessarry libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#importing the data
data=pd.read_csv('../input/winequality-red.csv')

#checking the head,info and description  of our data
data.head()
data.info()
data.describe()
fig,axes=plt.subplots(figsize=(10,6))
sns.distplot(data['fixed acidity'],kde=False,bins=50)
#citric acid vs quality
sns.boxplot(x='quality',y='citric acid',data=data)
#alcohol  vs quality
sns.boxplot(x='quality',y='alcohol',data=data)
#pH  vs quality
sns.boxplot(x='quality',y='pH',data=data)
#pairplot 
sns.pairplot(data)
#counting the number of varibles in quality column using countplot
sns.countplot(data['quality'])
data['quality'].value_counts()
#checking the correlation between different columns
data.corr()
#sulphates vs alcohol
sns.jointplot(x='sulphates',y='alcohol',data=data,kind='hex',color='red',size=8)
#checking for missing data
fig,axes=plt.subplots(figsize=(10,6))
sns.heatmap(data.isnull(),cmap='viridis',yticklabels=False,cbar=False)
#coversion of multivariable column target column into two varible (0 and 1 )
#converting the quality column into binary
#i.e bad or good using the pandas cut method
data['quality']=pd.cut(data['quality'],bins=(2,6.5,8),labels=['bad','good'])
#converting the categorical feature i.e bad or good into numerical feature 0 and 1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
new_data=le.fit_transform(data['quality'])
#saving the new quality column in our orignal dataframe and checking its head
data['quality']=new_data
data.head()
from sklearn.cross_validation import train_test_split
X=data.drop('quality',axis=1)
y=data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
#training different models and comparing the results
# K-nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier()
knc.fit(X_train,y_train)
knc_prediction=knc.predict(X_test)
# Decision Tree and Random Forest
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_prediction=dtc.predict(X_test)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_predictions=rfc.predict(X_test)
#Support Vector Machine
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)
svc_predictions=svc.predict(X_test)
from sklearn.grid_search import GridSearchCV
#for getting the best parameters in out model
grid=GridSearchCV(SVC(),param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.001,0.0001]},verbose=3)
grid.fit(X_train,y_train)
#checking the best parameters for our model
grid.best_params_
#Using the best parameters in our model
grid.best_estimator_
grid_predictions=grid.predict(X_test)
#Comparing the Predictions and Accuracy of our models
from sklearn.metrics import classification_report,accuracy_score
print('\t\t\tK-Nearest Neighbours\n\n',classification_report(y_test,knc_prediction))

print('\n\n\t\t\tDecision Tree\n\n',classification_report(y_test,dtc_prediction))

print('\n\n\t\t\tRandom Forest\n\n',classification_report(y_test,rfc_predictions))

print('\n\n\t\t\tSupport Vector Machine\n\n',classification_report(y_test,svc_predictions))

print('\n\n\t\t\tSVM With GridSearch\n\n',classification_report(y_test,grid_predictions))
