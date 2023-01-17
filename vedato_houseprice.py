# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install ycimpute



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder

from sklearn import datasets, metrics, model_selection, svm

import missingno as msno

from ycimpute.imputer import iterforest,EM

from fancyimpute import KNN

from sklearn.preprocessing import OrdinalEncoder



import numpy as np

import pandas as pd 

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier



from warnings import filterwarnings

filterwarnings('ignore')



pd.set_option('display.max_columns', None)

import gc
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn import model_selection

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import neighbors

from sklearn.svm import SVR



from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from catboost import CatBoostRegressor



from sklearn.model_selection import cross_val_score
encoder=OrdinalEncoder()

imputer=KNN()



def encode(data):

    '''function to encode non-null data and replace it in the original data'''

    #retains only non-null values

    nonulls = np.array(data.dropna())

    #reshapes the data for encoding

    impute_reshape = nonulls.reshape(-1,1)

    #encode date

    impute_ordinal = encoder.fit_transform(impute_reshape)

    #Assign back encoded values to non-null values

    data.loc[data.notnull()] = np.squeeze(impute_ordinal)

    return data
Ktrain= pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv") 

Ktest = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
Ktrain.head()
pd.DataFrame(Ktrain.nunique()).T
(Ktrain.describe().columns)[1:-1]
msno.matrix(Ktrain)

plt.show()

msno.matrix(Ktest)
msno.matrix(Ktrain.sort_values(by="SalePrice", ascending=False));
Ktrain["SalePrice"].describe()
plt.subplots(figsize=(45,40))

sns.heatmap(Ktrain.corr(), annot=True);
Ktrain.groupby("MSSubClass")["SalePrice"].mean().sort_values(ascending=False).plot(kind="bar",figsize=(10,5));
sns.scatterplot(x='GrLivArea', y='SalePrice' ,  data=Ktrain)

plt.xticks(rotation=90, color="r")

plt.yticks(color="r")

plt.xlabel("GrLivArea",color="r")

plt.ylabel("SalePrice",color="r");
plt.subplots(figsize=(16,8))

sns.boxplot(x="YearBuilt", y="SalePrice", data=Ktrain)

plt.xticks(rotation=90, color="r")

plt.yticks(color="r")

plt.xlabel("Built Year",color="r")

plt.ylabel("Sales Price",color="r");
Ktrain[['PoolQC', 'MiscFeature', 'Alley','FireplaceQu','GarageYrBlt', 'GarageCars', 'GarageArea','BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']]
Ktrain=Ktrain.drop(['PoolQC', 'MiscFeature', 'Alley','FireplaceQu','GarageYrBlt', 

                    'GarageCars', 'GarageArea','BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF', 

                    'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'], axis=1)



Ktest=Ktest.drop(['PoolQC', 'MiscFeature', 'Alley','FireplaceQu','GarageYrBlt', 

                    'GarageCars', 'GarageArea','BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF', 

                    'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'], axis=1)
Ktrain_cat=Ktrain.select_dtypes(include="object")

Ktest_cat =Ktest.select_dtypes(include="object")
for i in Ktrain_cat:

    encode(Ktrain_cat[i])

for i in Ktest_cat:

    encode(Ktest_cat[i])
print(Ktrain.shape)

Ktest.shape
Ktrain_=Ktrain.drop(Ktrain_cat, axis=1)

Ktrain=pd.concat([Ktrain_,Ktrain_cat], axis=1)



Ktest_=Ktest.drop(Ktest_cat, axis=1)

Ktest=pd.concat([Ktest_,Ktest_cat], axis=1)
print(Ktrain.shape)

Ktest.shape
Ktrain
"""

from sklearn import preprocessing

Ktrain_scaler =pd.DataFrame(preprocessing.scale(Ktrain.drop(["Id","SalePrice"], axis=1)))

Ktest_scaler = pd.DataFrame(preprocessing.scale(Ktest.drop(["Id"], axis=1)))

Ktrain= pd.concat([Ktrain_scaler,Ktrain[["Id","SalePrice"]]], axis=1)

Ktest= pd.concat([Ktest_scaler,Ktrain[["Id"]]], axis=1)

"""
Ktrain=Ktrain.fillna(-999)

Ktest=Ktest.fillna(-999)
def compML (df, y, alg):

    

    # train test ayrimi



    y=df[y]    

    X= df.drop(['SalePrice','Id'], axis=1).astype('float64')    



    X_train,X_test,y_train,y_test = train_test_split(X,

                                                     y, 

                                                     test_size=.25, 

                                                     random_state=42)

    

    # modeling

    model = alg().fit(X_train, y_train)

    y_pred = model.predict(X_test)

    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

    model_name = alg.__name__

    print(model_name,"model test error:", RMSE)
y=Ktrain["SalePrice"]    

X= Ktrain.drop(['SalePrice','Id'], axis=1).astype('float64')    

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=42)
models = [LGBMRegressor,

          XGBRegressor,

          GradientBoostingRegressor,

          RandomForestRegressor,

          DecisionTreeRegressor,

          MLPRegressor,

          KNeighborsRegressor,

          SVR]
for i in models:

    compML(Ktrain, "SalePrice", i)
bestRegModel= GradientBoostingRegressor().fit(X_train, y_train)
bestRegModel
params = {"learning_rate":[ 0.01, 0.1, 0.5],

             "max_depth":[3,6,8],

             "n_estimators":[100,200,500],

             "subsample":[1, 0.5, 0.8],

             "loss":["ls","lad","quartile"]

         }
# RegModel_cv = GridSearchCV(bestRegModel, params,cv=10, n_jobs=-1,verbose=2).fit(X_train, y_train)
# RegModel_cv.best_params_
bestRegModel_tuned= GradientBoostingRegressor(learning_rate=0.1, 

                                     loss='lad',

                                     max_depth=8,

                                     n_estimators=200,

                                     subsample=1).fit(X_train, y_train)

y_pred = bestRegModel.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))

linearmodels=[LinearRegression,

              Ridge,

              Lasso,

              ElasticNet]
for i in linearmodels:

    compML(Ktrain, "SalePrice", i)
bestLinearModel= Lasso().fit(X_train, y_train)
bestLinearModel.intercept_, bestLinearModel.coef_
tuned_LinearModel = cross_val_score(bestLinearModel, X_train, y_train, cv=10, scoring="neg_mean_squared_error")
np.sqrt(np.mean(-tuned_LinearModel))
ids=Ktest["Id"]

X_Ktest= Ktest.drop(["Id"], axis=1).astype("float64")
prediction=bestRegModel.predict(X_Ktest)

output=pd.DataFrame({"Id":ids, "SalePrice":prediction})

output.to_csv("submission_RegModel.csv", index=False)
output.shape
# This is by GradientBoostingRegressor 

Importance = pd.DataFrame({'Importance':bestRegModel.feature_importances_*100},

                         index = X_train.columns)

Importance.sort_values(by = 'Importance',

                      axis = 0,

                      ascending = True).plot(kind = 'barh',

                                            color = 'r',

                                            figsize=(45,40))

plt.xlabel('Variable Importance')

plt.gca().legend_ = None