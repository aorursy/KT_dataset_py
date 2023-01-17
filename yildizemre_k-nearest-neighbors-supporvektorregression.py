# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import Ridge , Lasso ,LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split , cross_val_score, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import neighbors

from sklearn.svm import SVR







from sklearn import model_selection

import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeCV,LassoCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/k-data.csv")
y=df["Salary"].astype('float64')
a = pd.DataFrame(data=df, columns=['Salary'])

a = a.fillna(0)
y=a.Salary.astype(int)



y

from warnings import filterwarnings

filterwarnings('ignore')
df.dropna()
dms = pd.get_dummies(df[['League','Division','NewLeague']])

y = df["Salary"]

x_= df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')

x=pd.concat([x_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)

x_train, x_test , y_train , y_test = train_test_split(x,y,test_size=0.25, random_state=42)
x_train.head()
y_train = y_train.fillna(0)

y_train = y_train.astype(int)

y_train
y_test = y_test.fillna(0)

y_test = y_test.astype(int)

y_test
knn_model = KNeighborsRegressor().fit(x_train,y_train)
dir(knn_model)
knn_model
knn_model.n_neighbors
knn_model.predict(x_test)[0:5]
y_pred =knn_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))
RMSE = []



for k in range(10):

    k=k+1

    knn_model=KNeighborsRegressor(n_neighbors=k).fit(x_train,y_train)

    y_pred=knn_model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test,y_pred))

    RMSE.append(rmse)

    print("k = ",k,"için RMSE degeri",rmse)

    

    
#GRİDSEARCHCV

knn_params= {"n_neighbors":np.arange(1,30,1)}

knn = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn,knn_params,cv=10).fit(x_train,y_train)
knn_cv_model.best_params_
#final 
knn_final= KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"]).fit(x_train,y_train)

y_final=knn_final.predict(x_test)

rmse_final=np.sqrt(mean_squared_error(y_test,y_final))

rmse_final

svr_model=SVR('linear')
svr_model
svr_model.fit(x_train,y_train)
svr_model.predict(x_test)[0:10]
dir(svr_model)
svr_model.intercept_
svr_model.coef_
y_pred=svr_model.predict(x_test)

np.sqrt(mean_squared_error(y_test,y_pred))
svr_params = {"C": [0.1,0.5,1.3]}

svr_cv_model=GridSearchCV(svr_model,svr_params,cv=5).fit(x_train,y_train)

svr_cv_model.best_params_

svr_final=SVR("linear", C= 0.1).fit(x_train,y_train)
y_pred = svr_final.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))