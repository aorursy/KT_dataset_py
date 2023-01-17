import numpy as np
import pandas as pd

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline

from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
df = pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/car data.csv')
df.head()
print(df.columns)
from datetime import datetime
currentYear = datetime.now().year
df['number_years'] = currentYear - df['Year']
df.head()
df[['Car_Name', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
df.drop(['Car_Name','Year'], axis=1, inplace=True)
print(df['Fuel_Type'].unique())
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
print(df['Fuel_Type'].isnull().sum())
print(df['Seller_Type'].isnull().sum())
print(df['Transmission'].isnull().sum())
print(df['Owner'].isnull().sum())
# How to draw a 3d plot - For tutorial

from mpl_toolkits import mplot3d
fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection = "3d")
z = np.linspace(0, 30, 10000)
x = np.sin(z)
y = np.cos(z)
ax.plot3D(x,y,z)
plt.show()
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20, 9))
ax  = fig.gca(projection = "3d")

plot =  ax.scatter(df["number_years"],
           df["Present_Price"],
           df["Kms_Driven"],
           linewidth=1,edgecolor ="k",
           c=df["Selling_Price"],s=100,cmap="hot")

ax.set_xlabel("Year")
ax.set_ylabel("Present_Price")
ax.set_zlabel("Kms_Driven")

lab = fig.colorbar(plot,shrink=.5,aspect=5)
lab.set_label("Selling_Price",fontsize = 25)

plt.title("3D plot for Year, Present price and Kms driven",color="red")
plt.show()
df.dtypes
df = pd.get_dummies(df, drop_first=True)
df.head()
df.corr()
corrmat=df.corr() 
top_corr_features=corrmat.index 
plt.figure(figsize=(20,20)) 
#plot heat map 
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
df.head()
df.columns
X = df[['Present_Price', 'Kms_Driven', 'Owner', 'number_years',
       'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Seller_Type_Individual',
       'Transmission_Manual']]

y = df['Selling_Price']
X.head()
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
from sklearn import linear_model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
y_test_prediction = lm.predict(X_test)

plt.scatter(y_test, y_test_prediction)
plt.xlabel("True Values")
plt.ylabel("Predictions")
print("Score:", model.score(X_test, y_test)*100, "%")
sns.distplot(y_test - y_test_prediction)
# https://www.pluralsight.com/guides/linear-lasso-ridge-regression-scikit-learn
rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train) 
pred_train_rr= rr.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_rr)))
print(r2_score(y_train, pred_train_rr))

pred_test_rr= rr.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_rr))) 
print(r2_score(y_test, pred_test_rr))
print("Score:", rr.score(X_test, y_test)*100, "%")
model_lasso = Lasso(alpha=0.01)
model_lasso.fit(X_train, y_train) 
pred_train_lasso= model_lasso.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
print(r2_score(y_train, pred_train_lasso))

pred_test_lasso= model_lasso.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_lasso))) 
print(r2_score(y_test, pred_test_lasso))
print("Score:", model_lasso.score(X_test, y_test)*100, "%")
model_enet = ElasticNet(alpha = 0.01)
model_enet.fit(X_train, y_train) 
pred_train_enet= model_enet.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_enet)))
print(r2_score(y_train, pred_train_enet))

pred_test_enet= model_enet.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_enet)))
print(r2_score(y_test, pred_test_enet))
print("Score:", model_enet.score(X_test, y_test)*100, "%")
print("Linear Accuracy Score:", model.score(X_test, y_test)*100, "%")
print("Ridge Accuracy Score:", rr.score(X_test, y_test)*100, "%")
print("Lasso Accuracy Score:", model_lasso.score(X_test, y_test)*100, "%")
print("Enet Accuracy Score:", model_enet.score(X_test, y_test)*100, "%")

df.head()
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(projection = '3d')

ax.scatter(df['Selling_Price'], df['Present_Price'], df['Kms_Driven'], c= 'red', marker = 'o')
ax.set_xlabel('Selling_Price')
ax.set_ylabel('Present_Price')
ax.set_zlabel('Kms_Driven')

from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
r_2 = [] # List for r^2 score
CV = [] # list for CV scores mean

# Main function for models
def model(algorithm, x_train_, y_train_, x_test_, y_test_, cv=10): 
    algorithm.fit(x_train_,y_train_)
    predicts = algorithm.predict(x_test_)
    prediction = pd.DataFrame(predicts) # create a dataframe from the predictions
    R_2 = r2_score(y_test_,prediction)
    cross_val = cross_val_score(algorithm,x_train_,y_train_,cv=cv)
    
    # Appending results to Lists 
    r_2.append(R_2)
    CV.append(cross_val.mean())
    
    # Printing results  
    print(algorithm,"\n") 
    print("r_2 score :",R_2,"\n")
    print("CV scores:",cross_val,"\n")
    print("CV scores mean:",cross_val.mean(), "\n")
    print("Accuracy Score: ", algorithm.score(x_test_, y_test_)*100, "%") #Same as R2 Error
    
    # Plot for prediction vs originals
    test_index=y_test_.reset_index()["Selling_Price"]
    ax=test_index.plot(label="True",figsize=(12,6),linewidth=2,color="red")
    ax=prediction[0].plot(label = "Predicted",figsize=(12,6),linewidth=2,color="black")
    plt.legend(loc='upper right')
    plt.title("True VS Predicted")
    plt.xlabel("index")
    plt.ylabel("values")
    plt.show()
lm = LinearRegression()
linear_model = model(lm, X_train, y_train, X_test, y_test)
rr = Ridge(alpha=0.01)
ridge_model = model(rr, X_train, y_train, X_test, y_test)
lr = Lasso(alpha=0.01)
lasso_model = model(lr, X_train, y_train, X_test, y_test)
model_enet = ElasticNet(alpha = 0.01)
enet_model = model(model_enet, X_train, y_train, X_test, y_test)
dtr = DecisionTreeRegressor()
model(dtr, X_train, y_train, X_test, y_test)
rfr = RandomForestRegressor(n_estimators = 100, random_state = 42)
model(rfr, X_train, y_train, X_test, y_test)
Model = ["LinearRegression","Lasso","Ridge","Elastic Net ", "DecisionTreeRegressor","RandomForestRegressor"]
results = pd.DataFrame({'Model': Model,'R Squared': r_2,'CV Mean Score': CV})
results