# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/property-prices-in-tunisia/Property Prices in Tunisia.csv")
df.head()
col=['room_count','bathroom_count','size', 'price','log_price']

df.mask(df[col] < 0,inplace=True)
df[col]=df[col].astype(np.float32)
df.head()
df.info()
df.dropna(inplace=True)
df.head()
df.drop(["city","region"], axis=1,inplace=True)
df.head()
X = pd.get_dummies(df.drop(columns=['price','log_price'])).values

y = df.price.values
df.describe()
X_train.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
df[df.city=="Kébili"]
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

polynomial_features= PolynomialFeatures(degree=2)

x_poly = polynomial_features.fit_transform(X_train)



model = LinearRegression()

model.fit(x_poly, y_train)

y_poly_pred = model.predict(x_poly)
from sklearn.metrics import mean_squared_error, r2_score

rmse = np.sqrt(mean_squared_error(y_train,y_poly_pred))

r2 = r2_score(y_train,y_poly_pred)

print("RMSE",rmse)

print("R2",r2)
from sklearn.svm import SVR

regressor = SVR(kernel='linear')

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
print('mean_squared_error: ',mean_squared_error(y_test, y_pred),

     '\nr2_score: ',r2_score(y_test, y_pred))
pd.DataFrame(y_pred).describe()
import statsmodels.api as sm

X_1=sm.add_constant()

ols_regr = sm.OLS(endog = y_train, exog = X_1).fit()
col_names = ['const']+pd.get_dummies(df.drop(["price","log_price"], axis=1)).columns.tolist()
ols_regr.summary(xname=col_names, yname='Price')
pd.get_dummies(df.drop(columns=['price','log_price'])).drop("category_Colocations",axis=1).values.shape
import seaborn as sns

sns.scatterplot(df.size,df.price)
df.head()
df_arenda=df[df.type=="À Louer"]

df_prodaja=df[df.type=="À Vendre"]
df_arenda["size"].value_counts()
np.percentile(df_arenda["size"].values,[1,99])
df_arenda["price"].describe()
np.percentile(df_arenda["price"].values,[1,99])
sns.scatterplot(df_arenda[(df_arenda["size"]<800) & (df_arenda["price"]<10000)]["size"],df_arenda[(df_arenda["size"]<800) & (df_arenda["price"]<10000)]["price"])
sns.scatterplot(df_arenda.size,df_arenda.price)
import statsmodels.api as sm

X_1=pd.get_dummies(df.drop(columns=['price','log_price'])).drop(["category_Colocations","category_Locations de vacances","category_Bureaux et Plateaux","category_Appartements","room_count","type_À Louer","category_Maisons et Villas","type_À Vendre","bathroom_count"],axis=1).values

ols_regr = sm.OLS(endog = df.price.values, exog = X_1).fit()



col_names = pd.get_dummies(df.drop(["price","log_price"], axis=1)).drop(["category_Colocations","category_Locations de vacances",'category_Bureaux et Plateaux',"category_Appartements","room_count","type_À Louer","category_Maisons et Villas","type_À Vendre","bathroom_count"],axis=1).columns.tolist()



ols_regr.summary(xname=col_names, yname='Price')
df.head()
col_names = ['const']+pd.get_dummies(df.drop(["price","log_price"], axis=1)).columns.tolist()
ols_regr.summary(xname=col_names, yname='Price')
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
print('mean_squared_error: ',mean_squared_error(y_test, y_pred),

     '\nr2_score: ',r2_score(y_test, y_pred))
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
print('mean_squared_error: ',mean_squared_error(y_test, y_pred),

     '\nr2_score: ',r2_score(y_test, y_pred))
X_1=sm.add_constant(X_train)

ols_regr = sm.OLS(endog = y_train, exog = X_1).fit()
col_names = ['const']+pd.get_dummies(df.drop(["price","log_price"], axis=1)).columns.tolist()
ols_regr.summary(xname=col_names, yname='Log_Price')