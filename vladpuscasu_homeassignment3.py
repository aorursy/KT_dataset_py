#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#Reading CSV's
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
#SalePrice Data Info
describe_train = train.SalePrice.describe()
describe_train
#Finding Train Skew
print ("Train skew: ", train.SalePrice.skew())
plt.hist(train.SalePrice, color='lightblue')
plt.show()
#Finding Target Skew
target = np.log(train.SalePrice)
print ("Target skew: ", target.skew())
plt.hist(target, color='lightblue')
plt.show()
#Finding Graph Correlations
numeric_features = train.select_dtypes(include=[np.number])
correlation = numeric_features.corr()
first_comparator = correlation['SalePrice'].sort_values(ascending=False)[:5]
second_comparator = correlation['SalePrice'].sort_values(ascending=False)[-5:]
print ("First values \n", first_comparator, "\n")
print ("Second values \n", second_comparator)
#Categories
categories = train.select_dtypes(exclude=[np.number])
print (categories.describe())
#Removing Outliers from Selected Index
train = train[train['GrLivArea'] < 3500]
plt.scatter(x = train['GrLivArea'], y = np.log(train.SalePrice))
plt.ylabel('Sale Price')
plt.xlabel('GrLivArea')
plt.show()
#Handling Nulls
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)
#Interpolate missing data with an average value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
#Linear Model, Splitting the Test Size For a More Accurate Prediction
y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33)
lr = linear_model.LinearRegression()
#Fitting the Linear Regression Model
lin_reg = lr.fit(X_train, y_train)
#Test
model = lin_reg.score(X_test, y_test)
model
#Predictions, RMSE
y_pred = lin_reg.predict(X_test)
pred = mean_squared_error(y_test, y_pred)
pred
#Graphing the Model
val = y_test
plt.scatter(y_pred, val, color = 'lightblue')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()
#Creating Final Predictions and the Data Frame to Submit
final = pd.DataFrame()
final['Id'] = test.Id
features = test.select_dtypes(include = [np.number]).drop(['Id'], axis = 1).interpolate()
predictions = lin_reg.predict(features)
final_predictions = np.exp(predictions)
print ("Original: ", predictions[:10], "\n")
print ("Final Predictions: ", final_predictions[:10])
final['SalePrice'] = final_predictions
final.to_csv('Submission1.csv', index=False)
final