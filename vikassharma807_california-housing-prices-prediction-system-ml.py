# Importing the libraries :



# Explore the Data :

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns



# Data preprocessing Libraries :

from sklearn.preprocessing import LabelEncoder

from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



# Regression Libraries :

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import cross_val_score
# read the data :

df_house = pd.read_csv('../input/housing.csv')
df_house.columns
List_of_Labels = list(df_house['median_house_value'].head(10))

List_of_Labels
df_house.head(5)
df_house.tail(5)
df_house.describe()
df_house.isnull().sum()
df_house.isnull().sum().plot(kind = 'bar')
# filling zero on the place of NaN values in the data set 

df_house['total_bedrooms'].fillna(0,inplace = True)
df_house.isnull().sum()
plt.figure(figsize=(15,6))

plt.subplots_adjust(hspace = .25)

plt.subplot(1,2,1)

plt.xlabel('ocean_proximity',fontsize=12)

plt.ylabel('median_house_value',fontsize=12)

sns.stripplot(data=df_house,x='ocean_proximity',y='median_house_value',)

plt.subplot(1,2,2)

plt.xlabel('ocean_proximity',fontsize=12)

plt.ylabel('median_house_value',fontsize=12)

sns.boxplot(data=df_house,x='ocean_proximity',y='median_house_value')

plt.plot()
plt.figure(figsize=(15,5))

plt.subplots_adjust(hspace = .25)

plt.subplot(1,2,1)

plt.title('Corelation b/w longtitude and median_house_value')

plt.xlabel('longitude',fontsize=12)

plt.ylabel('median_house_value',fontsize=12)

plt.scatter(df_house['longitude'].head(100),df_house['median_house_value'].head(100),color='g')

plt.subplot(1,2,2)

plt.title('Corelation b/w latitude and median_house_value')

plt.xlabel('latitude',fontsize=12)

plt.ylabel('median_house_value',fontsize=12)

plt.scatter(df_house['latitude'].head(100),df_house['median_house_value'].head(100),color='r')
df_house.plot(kind='scatter', x='longitude', y='latitude', alpha=0.9, 

    s=df_house['population']/100, label='population', figsize=(14,10), 

    c='median_house_value', cmap=plt.get_cmap('prism'), colorbar=True)
df_house.plot(kind='scatter', x='longitude', y='latitude', alpha=0.9, 

    s=df_house['population']/10, label='population', figsize=(14,10), 

    c='median_house_value', cmap=plt.get_cmap('cool'), colorbar=True)
plt.figure(figsize=(10,6))

sns.distplot(df_house['median_house_value'],color='red')

plt.show()
df_house[df_house['median_house_value']>450000]['median_house_value'].value_counts().head()
df_house=df_house.loc[df_house['median_house_value']<500001,:]

df_house=df_house[df_house['population']<25000]

plt.figure(figsize=(10,6))

sns.distplot(df_house['median_house_value'],color='yellow')

plt.show()
plt.figure(figsize=(15,6))

plt.subplots_adjust(hspace = .25)

plt.subplot(1,2,1)

df_house['ocean_proximity'].value_counts().plot(kind = 'pie',colormap = 'jet')

plt.subplot(1,2,2)

df_house['median_income'].hist(color='purple')
df_house.hist(bins=100, figsize=(20,20) , color = 'b')
plt.figure(figsize=(10,4))

sns.heatmap(cbar=False,annot=True,data=df_house.corr(),cmap='Blues')

plt.title('% Corelation Matrix')

plt.show()
x=df_house.iloc[:,:-1].values

print(x)
y=df_house['median_house_value'].values

print(y)
df_house['ocean_proximity'].value_counts().plot(kind = 'bar')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()

x[:, 8] = labelencoder.fit_transform(x[:, 8])

onehotencoder = OneHotEncoder(categorical_features = [8])

x = onehotencoder.fit_transform(x).toarray()
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
print('xtrain :')

print(xtrain)

print('xtest :')

print(xtest)
print('ytrain :')

print(ytrain)

print('ytest :')

print(ytest)
from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()

linear_regressor.fit(xtrain,ytrain)
# predict the value of dependent variable y

ypred = linear_regressor.predict(xtest)

ypred
from sklearn.metrics import mean_squared_error

predictions = linear_regressor.predict(xtest)

lin_mse = mean_squared_error(ytest,predictions)

lin_rmse = np.sqrt(lin_mse)

print('rmse value is : ',lin_rmse)
lin_reg_score = linear_regressor.score(xtest,ytest)

print('r squared value is : ',lin_reg_score )
from sklearn.tree import DecisionTreeRegressor

tree_regressor = DecisionTreeRegressor(random_state=0)

tree_regressor.fit(xtrain,ytrain)
y_pred = tree_regressor.predict(xtest)

y_pred
from sklearn.metrics import mean_squared_error

predictions = tree_regressor.predict(xtest)

lin_mse = mean_squared_error(ytest,predictions)

lin_rmse = np.sqrt(lin_mse)

print('rmse value is : ',lin_rmse)
tree_score = tree_regressor.score(xtest,ytest)

print('r squared value is : ',tree_score )
from sklearn.ensemble import RandomForestRegressor

rn_forest_regressor = RandomForestRegressor(n_estimators=50,random_state=0)

rn_forest_regressor.fit(xtrain,ytrain)
rn_forest_regressor.predict(xtest)
from sklearn.metrics import mean_squared_error

predictions = rn_forest_regressor.predict(xtest)

lin_mse = mean_squared_error(ytest,predictions)

lin_rmse = np.sqrt(lin_mse)

print('rmse value is : ',lin_rmse)
rsq_rn_forest = rn_forest_regressor.score(xtest,ytest)

print('r squared value is : ',rsq_rn_forest )