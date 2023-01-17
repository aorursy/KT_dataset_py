import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy.stats as stats

from sklearn.model_selection import train_test_split

import os

import warnings

warnings.filterwarnings('ignore')
cardata=pd.read_csv('../input/mtcars/mtcars.csv')
cardata.head()
cardata.rename(columns = {'Unnamed: 0':'model'}, inplace = True) 
cardata.head()
cardata.info()
cardata.isnull().sum()
cardata.describe()
cardata.columns
sns.pairplot(cardata)
plt.figure(figsize=(10,8))

sns.heatmap(cardata.corr(),annot=True)
cardata.corr()
# Mileage has strong negative correlation with cyl,displacemnet,horsepower

cardata.head()
# Find feature and target variable

cardata.drop(columns=['model'],axis=1,inplace=True)
cardata.describe()
# Treating outliers for the target variable



upper_limit= 22.8 + (1.5* (22.8-15.4))

lower_limit= 15.4 - (1.5* (22.8-15.4))

upper_limit,lower_limit
cardata=cardata[cardata.mpg<33.9]

cardata=cardata[cardata.mpg>4.3]

cardata.shape
cardata.head()
X=cardata.drop(columns=['mpg'],axis=1)

X.head()
y=cardata['mpg']
from sklearn.model_selection import train_test_split



X_train, X_test , y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 1)

print(X_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

model = lin_reg.fit(X_train,y_train)

print(f'R^2 score for train: {lin_reg.score(X_train, y_train)}')

print(f'R^2 score for test: {lin_reg.score(X_test, y_test)}')
# Raw OLS model.

import warnings 

warnings.filterwarnings('ignore')

import statsmodels.api as sm



X_constant = sm.add_constant(X)

lin_reg = sm.OLS(y,X_constant).fit()

lin_reg.summary()


import statsmodels.tsa.api as smt



acf = smt.graphics.plot_acf(lin_reg.resid, lags=10 , alpha=0.05)

acf.show()
from scipy import stats

print(stats.jarque_bera(lin_reg.resid))
import seaborn as sns



sns.distplot(lin_reg.resid)
import statsmodels.api as sm

sm.stats.diagnostic.linear_rainbow(res=lin_reg, frac=0.5)

import scipy.stats as stats

import pylab

from statsmodels.graphics.gofplots import ProbPlot

st_residual = lin_reg.get_influence().resid_studentized_internal

stats.probplot(st_residual, dist="norm", plot = pylab)

plt.show()
%matplotlib inline

%config InlineBackend.figure_format ='retina'

import seaborn as sns 

import matplotlib.pyplot as plt

import statsmodels.stats.api as sms

sns.set_style('darkgrid')

sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)



def linearity_test(model, y):

    '''

    Function for visually inspecting the assumption of linearity in a linear regression model.

    It plots observed vs. predicted values and residuals vs. predicted values.

    

    Args:

    * model - fitted OLS model from statsmodels

    * y - observed values

    '''

    fitted_vals = model.predict()

    resids = model.resid



    fig, ax = plt.subplots(1,2)

    

    sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})

    ax[0].set_title('Observed vs. Predicted Values', fontsize=16)

    ax[0].set(xlabel='Predicted', ylabel='Observed')



    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})

    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)

    ax[1].set(xlabel='Predicted', ylabel='Residuals')

    

linearity_test(lin_reg, y)  
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = [variance_inflation_factor(X_constant.values, i) for i in range(X_constant.shape[1])]

pd.DataFrame({'vif': vif[1:]}, index=X.columns).T
cardata.columns


X=cardata[['cyl', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear',

       'carb']]

y=cardata['mpg']
X_train, X_test , y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 1)

print(X_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(X, y)



print(f'Coefficients: {lin_reg.coef_}')

print(f'Intercept: {lin_reg.intercept_}')

print(f'R^2 score: {lin_reg.score(X, y)}')
X_constant = sm.add_constant(X)

lin_reg = sm.OLS(y,X_constant).fit()

lin_reg.summary()
from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 1)

print(X_train.shape)

print(X_test.shape)

print(y_test.shape)
lin_reg = LinearRegression()

model = lin_reg.fit(X_train,y_train)

print(f'R^2 score for train: {lin_reg.score(X_train, y_train)}')

print(f'R^2 score for test: {lin_reg.score(X_test, y_test)}')