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
import pandas as pd

df = pd.read_csv("../input/knearest/k-data.csv")
y=df["Salary"].astype('float64')

a = pd.DataFrame(data=df, columns=['Salary'])

a = a.fillna(0)

y=a.Salary.astype(int)



from warnings import filterwarnings

filterwarnings('ignore')



df.dropna()

dms = pd.get_dummies(df[['League','Division','NewLeague']])

y = df["Salary"]

x_= df.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')

x=pd.concat([x_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)

x_train, x_test , y_train , y_test = train_test_split(x,y,test_size=0.25, random_state=42)
y_train = y_train.fillna(0)

y_train =y_train.astype(int)

y_train = pd.DataFrame(y_train)

y_train
rf_model = RandomForestRegressor(random_state=42)

rf_model.fit(x_train,y_train)
dir(rf_model)
rf_model.predict(x_test)[0:10]
y_pred = rf_model.predict(x_test)
y_test = y_test.fillna(0)

y_test =y_test.astype(int)

y_test = pd.DataFrame(y_test)

y_test
np.sqrt(mean_squared_error(y_test,y_pred))
rf_params = {"max_depth": [1,2,3],"max_features":[4,5,6,20],"n_estimators" : [200,500,1000,2000],"min_samples_split": [2,10,51,70],}
rf_cv_model=GridSearchCV(rf_model,rf_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)
rf_cv_model.best_params_
rf_model = RandomForestRegressor(max_depth=3 , max_features=5 ,n_estimators=70 ,min_samples_split=200)

rf_tun=rf_model.fit(x_train,y_train)

y_pred=rf_tun.predict(x_test)

np.sqrt(mean_squared_error(y_test,y_pred))
