import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/big-mart-sales-dataset/Train_UWu5bXk.csv')
print("Num of Rows: ", data.shape[0])

print("Num of Cols:", data.shape[1])
print("Data Describe:")

data.describe()
print("Data Information:")

data.info()
print("To Check which column has a missing value and how much missing values it contains:")

data.isna().sum()
print("Unique values in Outlet Size: ")

data.Outlet_Size.unique()
print("Unique Values Count: ")

data.Outlet_Size.value_counts()
data.Outlet_Size.fillna(method='ffill', inplace=True)

print("There should be increase in unique values of Outlet Size: ")

print(data.Outlet_Size.value_counts())

print("\nTo Confirm that does the Outlet Size contain the missing values: ", data.Outlet_Size.unique())
print("Value Counts of Item Weight")

data.Item_Weight.value_counts()
# to see that if the data of Item Weight have any outliers in it

sns.boxplot(data.Item_Weight)

plt.show()
data.Item_Weight.plot(kind='box')

plt.show()
data.Item_Weight.fillna(data.Item_Weight.mean(), inplace=True)

print("To see that does now Item Weight contains the missing values: ", data.Item_Weight.isna().any())
sns.pairplot(data)

plt.show()
data.Item_Type.unique()
print("Number of Item Type occurs:")

data.Item_Type.value_counts()
data.groupby('Item_Type')['Item_Outlet_Sales'].count()
numCol = data.describe().columns

for nc in numCol:

    sns.distplot(data[nc])

    plt.show()
catCol = data.columns[data.dtypes == 'object']

for cc in catCol:

    print(cc, data[cc].unique())

    print()
data.Item_Fat_Content.value_counts()
data.Item_Fat_Content.value_counts().sum()
data['Item_Fat_Content'] = data.Item_Fat_Content.map({

    'Low Fat': 'LF',

    'Regular': 'R',

    'reg': 'R',

    'low fat': 'LF',

    'LF': 'LF'

    

})
print("Unique Values in Item Fat Content: ", data.Item_Fat_Content.unique())

print("Length of Item Fat Content: ", data.Item_Fat_Content.value_counts().sum())
catCol = list(catCol)

catCol.remove('Item_Identifier')

catCol.remove('Outlet_Identifier')

catCol
for cc in catCol:

    sns.countplot(y=data[cc])

    plt.show()
data.Outlet_Type.value_counts()
data.Outlet_Size.value_counts()
data.groupby(['Outlet_Type', 'Outlet_Size'])['Item_Outlet_Sales'].mean()
figsize = (10, 5)

fig, ax = plt.subplots(figsize = figsize)

sns.boxplot(x='Outlet_Type', y='Item_Outlet_Sales', data=data, ax = ax)

plt.show()
figsize = (10, 5)

fig, ax = plt.subplots(figsize = figsize)

sns.boxplot(x='Outlet_Size', y='Item_Outlet_Sales', data=data, ax = ax)

plt.show()
data.groupby(['Outlet_Type', 'Outlet_Size'])['Item_Outlet_Sales'].count().plot(kind='bar')

plt.show()
data.corr()
plt.matshow(data.corr())

plt.show()
data.groupby('Outlet_Identifier')['Item_Outlet_Sales'].count()
mostSalesOutlet = data[data.Outlet_Identifier == 'OUT027']

mostSalesOutlet.groupby(['Outlet_Size', 'Outlet_Type']).count()
df = data[['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales']]

df.head()
var1 = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales']

var2 = ['Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales']

var3 = ['Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Item_Outlet_Sales']

var4 = ['Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales']

var5 = ['Item_MRP', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales']

var6 = ['Item_MRP', 'Item_Outlet_Sales']
df1 = df[var1]

df2 = df[var2]

df3 = df[var3]

df4 = df[var4]

df5 = df[var5]

df6 = df[var6]
X1 = df1.iloc[:, :-1]

y1 = df1.iloc[:, -1]



X2 = df2.iloc[:, :-1]

y2 = df2.iloc[:, -1]



X3 = df3.iloc[:, :-1]

y3 = df3.iloc[:, -1]



X4 = df4.iloc[:, :-1]

y4 = df4.iloc[:, -1]



X5 = df5.iloc[:, :-1]

y5 = df5.iloc[:, -1]



X6 = df6.iloc[:, :-1]

y6 = df6.iloc[:, -1]
X1 = pd.get_dummies(X1)

X2 = pd.get_dummies(X2)

X3 = pd.get_dummies(X3)

X4 = pd.get_dummies(X4)

X5 = pd.get_dummies(X5)
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



train_X1, test_X1, train_y1, test_y1 = train_test_split(X1, y1, test_size=0.2, random_state=42)



train_X2, test_X2, train_y2, test_y2 = train_test_split(X2, y2, test_size=0.2, random_state=42)



train_X3, test_X3, train_y3, test_y3 = train_test_split(X3, y3, test_size=0.2, random_state=42)



train_X4, test_X4, train_y4, test_y4 = train_test_split(X4, y4, test_size=0.2, random_state=42)



train_X5, test_X5, train_y5, test_y5 = train_test_split(X5, y5, test_size=0.2, random_state=42)



train_X6, test_X6, train_y6, test_y6 = train_test_split(X6, y6, test_size=0.2, random_state=42)
def custom_model(train_X, test_X, train_y, test_y, model, model_name, var_col):

    model.fit(train_X, train_y)

    pred_y = model.predict(test_X)

    print(model_name + " with " + str(var_col))

    print("Accuracy Score: ", model.score(test_X, test_y))

    print("Mean Squared Error: ", mean_squared_error(test_y, pred_y))

    print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(test_y, pred_y)))

    print()
from sklearn.linear_model import LinearRegression

LR = LinearRegression()

custom_model(train_X1, test_X1, train_y1, test_y1, LR, 'Linear Regression', var1)

custom_model(train_X2, test_X2, train_y2, test_y2, LR, 'Linear Regression', var2)

custom_model(train_X3, test_X3, train_y3, test_y3, LR, 'Linear Regression', var3)

custom_model(train_X4, test_X4, train_y4, test_y4, LR, 'Linear Regression', var4)

custom_model(train_X5, test_X5, train_y5, test_y5, LR, 'Linear Regression', var5)

custom_model(train_X6, test_X6, train_y6, test_y6, LR, 'Linear Regression', var6)
from sklearn.tree import DecisionTreeRegressor

DTR = DecisionTreeRegressor()

custom_model(train_X1, test_X1, train_y1, test_y1, DTR, 'Decision Tree Regression', var1)

custom_model(train_X2, test_X2, train_y2, test_y2, DTR, 'Decision Tree Regression', var2)

custom_model(train_X3, test_X3, train_y3, test_y3, DTR, 'Decision Tree Regression', var3)

custom_model(train_X4, test_X4, train_y4, test_y4, DTR, 'Decision Tree Regression', var4)

custom_model(train_X5, test_X5, train_y5, test_y5, DTR, 'Decision Tree Regression', var5)

custom_model(train_X6, test_X6, train_y6, test_y6, DTR, 'Decision Tree Regression', var6)
from sklearn.ensemble import RandomForestRegressor

RFR = RandomForestRegressor()

custom_model(train_X1, test_X1, train_y1, test_y1, RFR, 'Random Forest Regression', var1)

custom_model(train_X2, test_X2, train_y2, test_y2, RFR, 'Random Forest Regression', var2)

custom_model(train_X3, test_X3, train_y3, test_y3, RFR, 'Random Forest Regression', var3)

custom_model(train_X4, test_X4, train_y4, test_y4, RFR, 'Random Forest Regression', var4)

custom_model(train_X5, test_X5, train_y5, test_y5, RFR, 'Random Forest Regression', var5)

custom_model(train_X6, test_X6, train_y6, test_y6, RFR, 'Random Forest Regression', var6)
from sklearn.svm import SVR

SV = SVR()

custom_model(train_X1, test_X1, train_y1, test_y1, SV, 'Support Vector Regression', var1)

custom_model(train_X2, test_X2, train_y2, test_y2, SV, 'Support Vector Regression', var2)

custom_model(train_X3, test_X3, train_y3, test_y3, SV, 'Support Vector Regression', var3)

custom_model(train_X4, test_X4, train_y4, test_y4, SV, 'Support Vector Regression', var4)

custom_model(train_X5, test_X5, train_y5, test_y5, SV, 'Support Vector Regression', var5)

custom_model(train_X6, test_X6, train_y6, test_y6, SV, 'Support Vector Regression', var6)
from sklearn.linear_model import Ridge

RR = Ridge()

custom_model(train_X1, test_X1, train_y1, test_y1, RR, 'Ridge Regression', var1)

custom_model(train_X2, test_X2, train_y2, test_y2, RR, 'Ridge Regression', var2)

custom_model(train_X3, test_X3, train_y3, test_y3, RR, 'Ridge Regression', var3)

custom_model(train_X4, test_X4, train_y4, test_y4, RR, 'Ridge Regression', var4)

custom_model(train_X5, test_X5, train_y5, test_y5, RR, 'Ridge Regression', var5)

custom_model(train_X6, test_X6, train_y6, test_y6, RR, 'Ridge Regression', var6)
from sklearn.linear_model import ElasticNet

ENR = ElasticNet()

custom_model(train_X1, test_X1, train_y1, test_y1, ENR, 'Elastic Net Regression', var1)

custom_model(train_X2, test_X2, train_y2, test_y2, ENR, 'Elastic Net Regression', var2)

custom_model(train_X3, test_X3, train_y3, test_y3, ENR, 'Elastic Net Regression', var3)

custom_model(train_X4, test_X4, train_y4, test_y4, ENR, 'Elastic Net Regression', var4)

custom_model(train_X5, test_X5, train_y5, test_y5, ENR, 'Elastic Net Regression', var5)

custom_model(train_X6, test_X6, train_y6, test_y6, ENR, 'Elastic Net Regression', var6)
from sklearn.linear_model import SGDRegressor

SGD = SGDRegressor()

custom_model(train_X1, test_X1, train_y1, test_y1, SGD, 'Stochastic Gradient Descent Regression', var1)

custom_model(train_X2, test_X2, train_y2, test_y2, SGD, 'Stochastic Gradient Descent Regression', var2)

custom_model(train_X3, test_X3, train_y3, test_y3, SGD, 'Stochastic Gradient Descent Regression', var3)

custom_model(train_X4, test_X4, train_y4, test_y4, SGD, 'Stochastic Gradient Descent Regression', var4)

custom_model(train_X5, test_X5, train_y5, test_y5, SGD, 'Stochastic Gradient Descent Regression', var5)

custom_model(train_X6, test_X6, train_y6, test_y6, SGD, 'Stochastic Gradient Descent Regression', var6)
from sklearn.ensemble import AdaBoostRegressor

ABR = AdaBoostRegressor()

custom_model(train_X1, test_X1, train_y1, test_y1, ABR, 'AdaBoost Regression', var1)

custom_model(train_X2, test_X2, train_y2, test_y2, ABR, 'AdaBoost Regression', var2)

custom_model(train_X3, test_X3, train_y3, test_y3, ABR, 'AdaBoost Regression', var3)

custom_model(train_X4, test_X4, train_y4, test_y4, ABR, 'AdaBoost Regression', var4)

custom_model(train_X5, test_X5, train_y5, test_y5, ABR, 'AdaBoost Regression', var5)

custom_model(train_X6, test_X6, train_y6, test_y6, ABR, 'AdaBoost Regression', var6)
from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor()

custom_model(train_X1, test_X1, train_y1, test_y1, GBR, 'Gradient Boosting Regression', var1)

custom_model(train_X2, test_X2, train_y2, test_y2, GBR, 'Gradient Boosting Regression', var2)

custom_model(train_X3, test_X3, train_y3, test_y3, GBR, 'Gradient Boosting Regression', var3)

custom_model(train_X4, test_X4, train_y4, test_y4, GBR, 'Gradient Boosting Regression', var4)

custom_model(train_X5, test_X5, train_y5, test_y5, GBR, 'Gradient Boosting Regression', var5)

custom_model(train_X6, test_X6, train_y6, test_y6, GBR, 'Gradient Boosting Regression', var6)