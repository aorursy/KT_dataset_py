import pandas as pd                 

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn import linear_model

import datetime

import pandas_profiling

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, ElasticNet, Ridge, Lasso
import matplotlib.pyplot as plt     

import seaborn as sns

%matplotlib inline
train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')
print("Train data shape:", train.shape)

print("Test data shape:", test.shape)
train.head(2)
train['price']=train['data-price']

#train.SalePrice.describe()

print (train.price.describe())
print ("Skew is:", train.price.skew())

sns.distplot(train.price, color='magenta')

plt.show()
target = np.log(train.price)
print ("Skew is:", target.skew())

sns.distplot(target, color='magenta')

plt.show()
cor=train.corr()

cor
#numeric features.dtypes

numeric_data = train.select_dtypes(include=[np.number])

print(numeric_data.dtypes)
corr = numeric_data.corr()
train.corr()['price'].sort_values(ascending=False)
sns.pairplot(numeric_data)
sns.heatmap(corr,annot=True)
#We set index='Arae' and values='SalePrice'. We chose to look at the median here.

quality_pivot = train.pivot_table(index='area', values='price', aggfunc=np.median)

print(quality_pivot)
#visualize this pivot table more easily, we can create a bar plot

plt.rcParams['figure.figsize'] = (10, 8)

quality_pivot.plot(kind='bar', color='b')

plt.xlabel('Area')

plt.ylabel('Median Price')

plt.xticks(rotation=0)

plt.show()
plt.scatter(x=train['bathroom'], y=train['price'],color='b')

plt.ylabel('Price')

plt.xlabel('bathroom')

plt.show()
plt.scatter(x=train['buildingSize'], y=train['price'],color='b')

plt.ylabel('Price')

plt.xlabel('buildingSize')

plt.show()
pandas_profiling.ProfileReport(train)
tes=test.merge(train,on='house-id')
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False))

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'

nulls.head()
plt.rcParams['figure.figsize'] = (10, 6)

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

train.columns
cols=[ 'bathroom','garage']
for i in cols:

    train[i].fillna(0,inplace=True)

    test[i].fillna(train[i].mean(),inplace=True)
col=[ 'buildingSize','erfSize','bedroom']
for i in col:

    train[i].fillna(train[i].mean(),inplace=True)

    test[i].fillna(train[i].mean(),inplace=True)
print ("Unique values are:", train.type.unique())
print(train.area.unique())
print(train['data-isonshow'].unique())
cat_data = train.select_dtypes(exclude=[np.number])

#categoricals.describe()

cat_data.dtypes
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train['data-date']= pd.to_datetime(train['data-date'])

train['year']= train['data-date'].dt.year

test['data-date']= pd.to_datetime(test['data-date'])

test['year']= test['data-date'].dt.year
train.head()
# Dummy variables added



train = pd.get_dummies(train, columns = ['data-isonshow'], prefix ='data', drop_first = True)

train = pd.get_dummies(train, columns = ['area'], prefix ='area', drop_first = True)

train = pd.get_dummies(train, columns = ['data-location'], prefix ='data-location', drop_first = True)

train= pd.get_dummies(train, columns = ['type'], prefix ='type', drop_first = True)
test = pd.get_dummies(test, columns = ['data-isonshow'], prefix ='data', drop_first = True)

test = pd.get_dummies(test, columns = ['area'], prefix ='area', drop_first = True)

test = pd.get_dummies(test, columns = ['data-location'], prefix ='data-location', drop_first = True)

test= pd.get_dummies(test, columns = ['type'], prefix ='type', drop_first = True)
test.head(2)
train.drop(columns=['data-date','data-price','data-url',],inplace=True)

test.drop(columns=['data-date','data-url'],inplace=True)
test.head(2)
y = target

X = train.drop(['price','house-id'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = model.predict(X_train)
print('RMSE is: \n', mean_squared_error(y_train, predictions))
plt.scatter(predictions, y_train,color='blue') 

plt.xlabel('Predicted Price')

plt.ylabel('y_train(Actual Price)')

plt.title('Linear Regression Model')

plt.show()
predictions = lm.predict(X_test)
print('RMSE is: \n', mean_squared_error(y_test, predictions))
plt.scatter(predictions, y_test,color='b') 

plt.xlabel('Predicted Price')

plt.ylabel('y_test(Actual Price)')

plt.title('Linear Regression Model')

plt.show()
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)
coeff = pd.DataFrame(ridge.coef_, X.columns, columns=['Coefficient'])
coeff.head()
predictions= ridge.predict(X_train)
# calculates the rmse

print('RMSE is: \n', mean_squared_error(y_train, predictions))
predictions= ridge.predict(X_test)

len(predictions)
# calculates the rmse

print('RMSE is: \n', mean_squared_error(y_test, predictions))
model_lasso = Lasso(alpha=0.00055)

model_lasso.fit(X_train, y_train)



predictions = model_lasso.predict(X_train)



print('RMSE is: \n', mean_squared_error(y_train, predictions))
predictions = model_lasso.predict(X_test)
print('RMSE is: \n', mean_squared_error(y_test, predictions))
from sklearn.ensemble import RandomForestRegressor

rfor = RandomForestRegressor(random_state=42)

rfor.fit(X_train,y_train)
predictions=rfor.predict(X_train)
print('RMSE is: \n', mean_squared_error(y_train, predictions))
predictions=rfor.predict(X_test)

len(predictions)
print('RMSE is: \n', mean_squared_error(y_test, predictions))