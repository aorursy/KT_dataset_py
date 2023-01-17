#importing the required variables

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sn
#reading the csv file

df = pd.read_csv("/kaggle/input/Train_UWu5bXk.txt")

print("read successfully")
df.head(10)
df1 = df

df.describe()
df.info()
df.shape
# counting the types of fat content

sn.countplot(df.Item_Fat_Content)
# replacing LF, loe fat with Loe Fat and reg with Regular

df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'low fat':'Low Fat', 'LF':'Low Fat', 'reg':'Regular'})

sn.countplot(df.Item_Fat_Content)
# calculating the no of years an outlet has worked

df['Outlet_Working_Years'] = 2020 - df['Outlet_Establishment_Year']

df.head()
# count of outlets depending on their working years

sn.countplot(df.Outlet_Working_Years)
# Sales VS outlet working years

sales_vs_Outlet_working_years = df.pivot_table(index = 'Outlet_Working_Years', values = 'Item_Outlet_Sales')

sales_vs_Outlet_working_years.sort_values(by ='Outlet_Working_Years', ascending = False)
sales_vs_Outlet_working_years.plot(kind = 'bar')
#counting the no of outlet sizes

sn.countplot(df.Outlet_Size)
# two cols have null values

df.isnull().any()
#fill Item_Weigth(mean) by their corresponding Item_Type

List=['Baking Goods','Breads','Breakfast','Canned','Dairy','Frozen Foods','Fruits and Vegetables','Hard Drinks','Health and Hygiene','Household','Meat','Others','Seafood','Snack Foods','Soft Drinks','Starchy Foods']



Mean_values_Item_Type = df.groupby('Item_Type')['Item_Weight'].mean()

for i in List:

    d={i:Mean_values_Item_Type[i]}

    s=df.Item_Type.map(d)

    df.Item_Weight=df.Item_Weight.combine_first(s)
# mean weight of items as per item_type

Mean_values_Item_Type.plot(kind = 'bar')
df.Item_Weight.isnull().any()
mode_outlet_size = df['Outlet_Size'].mode()

mode_outlet_size
# filling the missing values of outlet size with medium

df['Outlet_Size'].fillna('Medium', inplace = True)

df.head()
# no null values

df.isnull().any()
df.info()
df.head()
# since there are item_visibility with 0.0000 values

df['Item_Visibility'].replace(0.00000,np.nan)

df['Item_Visibility'].fillna(df.groupby('Item_Identifier')['Item_Visibility'].transform('mean'))
df.head()
# label encoding multiple features

from sklearn.preprocessing import LabelEncoder

features_encoding = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

labelencoder = LabelEncoder()



for i in features_encoding:

    df[i]=labelencoder.fit_transform(df[i])
df.head()
# creating heatmaps

corr = df.corr()

plt.subplots(figsize=(14,14))

sn.heatmap(corr, annot = True, cmap = 'cool')

plt.show()
# selecting independent features

features=['Item_Identifier','Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Identifier','Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Working_Years']

X = df[features]

X.head()
# the dependent feature

Y = df['Item_Outlet_Sales']

Y.head()
# spliting the data into training and testing samples(test size 30%)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 10)
# fiting the training data to multiple linear regression model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,Y_train)
#predicting 

y_pred = regressor.predict(X_test)

y_pred
# calculating the scores of linear regression model

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.metrics import r2_score



r2_score = r2_score(Y_test,y_pred)



mse = mean_squared_error(Y_test, y_pred)



rmse = sqrt(mean_squared_error(Y_test, y_pred))
print('r2-score -> ', r2_score)

print('mean square error -> ', mse)

print('root mean square error -> ', rmse)
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
# selecting parameters and its values for gridsearchcv

parameters = {

              'n_estimators' : [10,15,20,25,30,35,40,45,50,55],

              'max_depth': [3,4,5,None],

              'min_samples_split' : [2,3,4],

              'max_features' : [5,6,7,8,9,10,11,'sqrt'],

              'bootstrap' : [True,False]

             }

rf = RandomForestRegressor(random_state = 5) 



rf_grid_search = GridSearchCV(rf,parameters)
# to get the selected parameters

rf_grid_search.get_params()
#fitting training data to random forest regressor with the gridsearchcv selected parameters

rf_grid_search.fit(X_train,Y_train)
# predicting

rf_y_pred = rf_grid_search.predict(X_test)

rf_y_pred
# calculating scores for random forest regressor model

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.metrics import r2_score



r2_score = r2_score(Y_test,rf_y_pred)



mse = mean_squared_error(Y_test, rf_y_pred)



rmse = sqrt(mean_squared_error(Y_test, rf_y_pred))



print('r2-score -> ', r2_score)

print('mean square error -> ', mse)

print('root mean square error -> ', rmse)