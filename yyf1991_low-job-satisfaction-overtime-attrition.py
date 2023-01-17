import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
data=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.head()
data.describe()
del data['EmployeeCount']
data.isnull().any()
data.shape
data.groupby('Attrition').size()
data['Attrition']=data['Attrition'].apply(lambda x : 1 if x=='Yes' else 0)
sns.boxplot(x='Attrition',y='Age',data=data)
sns.boxplot(x='Attrition',y='DailyRate',data=data)
sns.boxplot(x='Attrition',y='YearsAtCompany',data=data)
num_cat=['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction' , 'WorkLifeBalance']
for i in num_cat:
    data[i]=data[i].astype('category')
data=pd.get_dummies(data)
data.info()
data['Age'].describe()
del data['Over18_Y']
data.shape
X=data[data.columns.difference(['Attrition'])]
y=data['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2,random_state=42)
numeric_variables = list(data.select_dtypes(include='int64').columns.values)
numeric_variables.remove('Attrition')
numeric_variables
#First is to reset index for X_train and X_test
X_train.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)
#Separate into two dataframes for numeric and non-numeric variables 
X_train_num=X_train[numeric_variables]
X_train_nnum=X_train[X_train.columns.difference(numeric_variables)]
X_test_num=X_test[numeric_variables]
X_test_nnum=X_test[X_train.columns.difference(numeric_variables)]
#Set standard scaler 
scaler=StandardScaler()
#Fit and transform training set 
X_train_scaled=scaler.fit_transform(X_train_num)
X_train_scaled=pd.DataFrame(data=X_train_scaled,columns=X_train_num.columns)
X_train_scaled=pd.concat([X_train_scaled,X_train_nnum],axis=1)
#Transform training set
X_test_scaled=scaler.transform(X_test_num)
X_test_scaled=pd.DataFrame(data=X_test_scaled,columns=X_test_num.columns)
X_test_scaled=pd.concat([X_test_scaled,X_test_nnum],axis=1)
knn=KNeighborsClassifier()
knn.fit(X_train_scaled,y_train)
knn.score(X_test_scaled,y_test)
y_predict = knn.predict(X_test_scaled)
confusion_matrix(y_test,y_predict)
logis=LogisticRegression()
logis.fit(X_train_scaled,y_train)
logis.score(X_test_scaled,y_test)
y_predict = logis.predict(X_test_scaled)
confusion_matrix(y_test,y_predict)
logis.coef_
print('The most positive influent coefficient is {0}, with value equal to {1}'.format(X_test_scaled.columns[np.argmax(logis.coef_)],logis.coef_.max()))
print('The most negative influent coefficient is {0}, with value equal to {1}'.format(X_test_scaled.columns[np.argmin(logis.coef_)],logis.coef_.min()))
pd.crosstab(data.JobInvolvement_1,data.Attrition)
pd.crosstab(data.OverTime_No,data.Attrition)