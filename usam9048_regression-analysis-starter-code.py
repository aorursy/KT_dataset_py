import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("/kaggle/input/autompg-dataset/auto-mpg.csv")

data.head()
data.shape
data.columns
data.isnull().sum()
data.index
data.tail()
data.describe()
data.info()
data['horsepower'] = data['horsepower'].replace('?','100')
data['horsepower'].value_counts()
data.head()
#mpg as factors

print('Highest mpg is',data.mpg.max(),'millions per gallon')

print('Lowest mpg is',data.mpg.min(),'millions per gallon')

f,ax = plt.subplots(1,2,figsize=(12,6))

sns.boxplot(data.mpg,ax=ax[0])

sns.distplot(data.mpg,ax=ax[1])
print("Skewness: ",data['mpg'].skew())

print("Kurtosis: ",data['mpg'].kurtosis())
corr = data.corr()

corr
data.corr()['mpg'].sort_values()
plt.figure(figsize=(12,5))

sns.heatmap(corr,annot = True,cmap = 'Accent',linewidths = 0.2 )
## multivariate analysis

sns.boxplot(y='mpg',x='cylinders',data=data)

plt.show()
data['car name'].describe()
data['car name'].value_counts()
data['car name'].unique()
data['car name'] = data['car name'].str.split(' ').str.get(0)

data['car name'].value_counts()
data['car name'] = data['car name'].replace(['chevroelt','chevy'],'chevrolet')

data['car name'] = data['car name'].replace(['vokswagen','vw'],'volkswagen')

data['car name'] = data['car name'].replace('maxda','mazda')

data['car name'] = data['car name'].replace('toyouta','toyota')

data['car name'] = data['car name'].replace('mercedes','mercedes-benz')

data['car name'] = data['car name'].replace('nissan','datsun')

data['car name'] = data['car name'].replace('capri','ford')

data['car name'].value_counts()

plt.figure(figsize=(15,8))

sns.countplot(data['car name'])

plt.xticks(rotation = 90)
sns.scatterplot(x='cylinders',y='displacement',hue = 'mpg',data=data,cmap = 'rainbow')
x = data.iloc[:,1:].values

y = data.iloc[:,0].values

x
#Encoding categorical data

from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

x[:,7] = lb.fit_transform(x[:,7])



from sklearn.preprocessing import OneHotEncoder

onehot = OneHotEncoder(categorical_features = [7])

x = onehot.fit_transform(x).toarray()
#Splitting into training and test data

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 0)

#feature scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x = sc.fit_transform(x)

# multiple linear regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(xtrain,ytrain)

ypred = lr.predict(xtest)



lr.score(xtrain,ytrain)



from sklearn.metrics import r2_score

print("Accuracy of the linear model is:",round(r2_score(ytest,ypred)*100,2),'%')

#Decision tree Regression

from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state = 0)

dtr.fit(xtrain,ytrain)



ypred_dtr = dtr.predict(xtest) 

print('Accuracy of the decision tree model is:',round(r2_score(ytest,ypred_dtr)*100,2),'%')
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators = 200,random_state = 0)

rfr.fit(xtrain,ytrain)



ypred_rfr = rfr.predict(xtest)

print('Accuracy of the random forest model:',round(r2_score(ytest,ypred_rfr)*100,2),'%')
#Support Vector Regression

from sklearn.svm import SVR

svr = SVR(kernel = 'rbf',gamma = 'scale')

svr.fit(xtrain,ytrain)



ypred_svr = svr.predict(xtest)

print('Accuracy of the SVR model :',round(r2_score(ytest,ypred_svr)*100,2),'%')
