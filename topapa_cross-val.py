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
#Importar biblios:



from sklearn.linear_model import LassoLarsCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LassoLarsCV

from sklearn import linear_model

from sklearn.svm import SVR



from yellowbrick.regressor import CooksDistance

from sklearn.linear_model import SGDRegressor

from sklearn.linear_model import Ridge

from sklearn import svm



from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import OrdinalEncoder



import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import seaborn as sns

from scipy import stats

import statsmodels.api as sm

from statsmodels.formula.api import ols



import pandas as pd

from sklearn.preprocessing import PolynomialFeatures

import warnings



#Para o pipeline:

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from pandas.plotting import scatter_matrix



from sklearn.base import TransformerMixin
#Mostrar todas colunas:

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
from sklearn.model_selection import cross_val_score
X = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
XT = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
X.head()
Y = X['SalePrice'].copy()



X = X.drop('SalePrice', axis=1)

X = X.drop('Id', axis=1)
#Criando o modelo:



regr = LinearRegression()

regr.fit(X[['OverallQual','OverallCond', 'LotArea', 'TotRmsAbvGrd']] , Y)



#Cross validation:



scores = cross_val_score(regr, X[['OverallQual','OverallCond', 'LotArea', 'TotRmsAbvGrd']], np.log(Y), cv=10, scoring="neg_mean_squared_error")



#Scikit-Learn cross-validation features expect a utility function (greater is better) rather than a cost function (lower is better), so

#the scoring function is actually the opposite of the MSE (i.e., a negative value), which is why the preceding code computes -

#scores before calculating the square root.



scores
scores = -scores

scores = np.sqrt(scores)

print('media: ', scores.mean() , 'std: ', scores.std())

#Separar numerico de categorico



X_numeros = X.select_dtypes(include=[np.number])

X_cat = X.select_dtypes(include=['object'])



#Drop id



#Achando infinitos:

np.isfinite(X_numeros).all()

#Substituindo valores infinitos por NaN:

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X[['LotArea']].max()
X[['GrLivArea']].idxmax()
X.loc[X['GrLivArea']==4676]
X.drop(X.loc[(X['LotArea']==215245) & (X['GrLivArea']==5642) ].index, inplace=True)
class log1ptransformer(TransformerMixin):

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        #X é dataframe

        Xlog = np.log1p(X)

        return Xlog

#e log apenas



class logtransformer(TransformerMixin):

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        #X é dataframe

        Xlog = np.log(X)

        return Xlog



    

class DFStandardScaler(TransformerMixin):

    # StandardScaler but for pandas DataFrames



    def __init__(self):

        self.ss = None

        self.mean_ = None

        self.scale_ = None



    def fit(self, X, y=None):

        self.ss = StandardScaler()

        self.ss.fit(X)

        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)

        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)

        return self



    def transform(self, X):

        # assumes X is a DataFrame

        Xss = self.ss.transform(X)

        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)

        return Xscaled



class DFSimpleImputer(TransformerMixin):

    # Imputer but for pandas DataFrames



    def __init__(self, strategy='median'):

        self.strategy = strategy

        self.imp = None

        self.statistics_ = None



    def fit(self, X, y=None):

        self.imp = SimpleImputer(strategy=self.strategy)

        self.imp.fit(X)

        self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)

        return self



    def transform(self, X):

        # assumes X is a DataFrame

        Ximp = self.imp.transform(X)

        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)

        return Xfilled

    



        



    

    #Pipeline dos números e categorias:





#num_pipeline = Pipeline([('Log', log1ptransformer() ),

#        ('imputer', SimpleImputer(strategy='median')),

#        ('std_scaler', DFStandardScaler())])



num_pipeline = Pipeline([

        ('log', log1ptransformer()),

        #('box', power_transform(method='box-cox')),

        ('std_scaler', DFStandardScaler()),

        ('imputer', DFSimpleImputer(strategy='median'))

    ])





cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),

                         ('cat', OneHotEncoder()),

                        ])



ordinal_pipeline = Pipeline([('ordinal', OrdinalEncoder())])
len(cat_attribs)
### OBS USANDO LOG1P (SOMA 1)

###





#tirar YearBuilt

#num_attribs = ['GrLivArea', 'OverallQual','LotFrontage','LotArea', 'GarageArea', 'TotRmsAbvGrd']



#num_attribs = ['GroundNRoom', 'OverallQual','LotFrontage','LotArea', 'GarageArea']

#num_attribs = ['GLA_LOT', 'OverallQual','YearBuiltDiferenca','GarageArea']

#cat_attribs = ["Neighborhood"]



#num_attribs = list(X_numeros)

#cat_attribs = list(X_cat)



#16/09/2020

#ord_attribs = ['OverallQual']

num_attribs = ['GrLivArea','LotArea','MasVnrArea', 'GarageArea','BedroomAbvGr', 'TotRmsAbvGrd', 'YearBuilt','KitchenAbvGr', 'GarageArea', 'GarageCars','TotalBsmtSF', '1stFlrSF', 'FullBath']

cat_attribs = ["Neighborhood", 'OverallQual', 'OverallCond', 'ExterQual', 'Foundation','KitchenQual']



#num_attribs = ['GrLivArea','LotArea', 'GarageArea', 'TotRmsAbvGrd', 'MasVnrArea']

#cat_attribs = ["Neighborhood", 'OverallQual' ]



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),

        ("cat", cat_pipeline, cat_attribs),

        #("ord", ordinal_pipeline, ord_attribs)

        #("Poly", PolynomialFeatures(interaction_only=True), num_attribs),

        #("cat", OneHotEncoder(), cat_attribs),

    ])



X_completo_processado = full_pipeline.fit_transform(X)
Y_log = np.log(Y)
#CRIANDO O MODELO LINEAR REGRESSION:



lin_reg = LinearRegression()

lin_reg.fit(X_completo_processado, Y)









scores = cross_val_score(lin_reg, X_completo_processado, Y, cv=10, scoring="neg_mean_squared_error")

rmse_scores = np.sqrt(-scores)

rmse_scores.mean()
rmse_scores.std()
Xtest = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
T_completo_processado = full_pipeline.fit_transform(Xtest)
predicted_prices = lin_reg.predict(T_completo_processado)
predicted_prices
my_submission = pd.DataFrame({'Id': Xtest.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('submission.csv', index=False)