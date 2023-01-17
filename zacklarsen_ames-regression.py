# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

%config InlineBackend.figure_format = 'retina'
import os

import time



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import pylab as pl

import plotly.plotly as py

import plotly.graph_objs as go



import numpy as np

np.set_printoptions(precision=2, linewidth=120, suppress=True, edgeitems=4)



import pandas as pd

pd.set_option('display.max_columns', 350)

#pd.set_option('precision', 5)



from sklearn.model_selection import KFold,train_test_split,StratifiedKFold, GridSearchCV

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor,RandomizedLasso

from sklearn.feature_selection import RFE, f_regression

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR

from sklearn.feature_selection import RFE, RFECV
os.chdir('../input/')

!ls

os.getcwd()
train = pd.read_csv('train.csv',sep=',',error_bad_lines=False)

train.head()

len(train.columns)
plt.hist(train['SalePrice'])

plt.xlabel("Price")

plt.ylabel("Frequency")

plt.title("Histogram Plot")
# Using seaborn:

sns.distplot(train.SalePrice,color='green')
# Select our independent variables to feed into the regression model as 'X'

X = train.iloc[:,np.r_[1:80]]

X.head(5)



y = train.iloc[:,80]

y.head(5)
# Train/test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



# Use dummy variables for categoricals so we can standardize the data

one_hot_encoded_training_predictors = pd.get_dummies(X_train)

one_hot_encoded_test_predictors = pd.get_dummies(X_test)



# Ensure that the column names and column order are equivalent in both sets

final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,

                                                                    join='left', 

                                                                    axis=1)
final_train.head(5)
final_train[pd.isnull(final_train).any(axis=1)].head(5)

final_test[pd.isnull(final_test).any(axis=1)].head(5)
## We can safely assume that 'LotFrontage' missing values can be coded as zero 

## because a zero simply indicates an absence of property frontage

final_train['LotFrontage'].fillna(0, inplace=True)

final_test['LotFrontage'].fillna(0, inplace=True)





## Garage year built has many NaN values, and we can safely impute this as the same as the

## year that the house itself was build ('YearBuilt')

final_train['GarageYrBlt'].fillna(final_train['YearBuilt'], inplace=True)

final_test['GarageYrBlt'].fillna(final_test['YearBuilt'], inplace=True)
# Check again for NaN

final_train[pd.isnull(final_train).any(axis=1)]



# MasVnrArea is the only column with NaN

# There are only 2 NaN values left, both for 'MasVnrArea' which is:

# Masonry veneer area in square feet

# It is safe to assume this is zero simply because there is no masonry veneer in the home.

# We can impute this as zero.

final_train['MasVnrArea'].fillna(0, inplace=True)

final_test['MasVnrArea'].fillna(0, inplace=True)



# Check again for NaN again

final_train[pd.isnull(final_train).any(axis=1)]
# Check again for NaN

final_test[pd.isnull(final_test).any(axis=1)].head()
## final_test is missing values in all of the columns below:

# Utilities_NoSeWa

# HeatingQC_Po

# Electrical_FuseP

# Functional_Sev

# GarageQual_Ex

# GarageCond_Ex

# MiscFeature_TenC

# SaleType_CWD

# SaleCondition_AdjLand



# Condition2_Artery

# Condition2_PosN

# Condition2_RRAe

# Condition2_RRAn

# RoofStyle_Shed

# RoofMatl_ClyTile

# RoofMatl_Metal

# RoofMatl_Roll

# Exterior1st_AsphShn

# Exterior1st_CBlock

# Exterior2nd_CBlock

# Exterior2nd_Other

# ExterCond_Po

# Foundation_Wood

# Heating_Floor

# Heating_OthW

# HeatingQC_Po

# Electrical_FuseP

# Functional_Sev

# GarageQual_Ex

# GarageCond_Ex

# MiscFeature_TenC

# SaleType_CWD

# SaleCondition_AdjLand



# Because these are all dummy coded variables for categoricals, we can

# safely assume that imputing the missing values as zero is appropriate



final_test['Utilities_NoSeWa'].fillna(0, inplace=True)

final_test['HeatingQC_Po'].fillna(0, inplace=True)

final_test['Electrical_FuseP'].fillna(0, inplace=True)

final_test['Functional_Sev'].fillna(0, inplace=True)

final_test['GarageQual_Ex'].fillna(0, inplace=True)

final_test['GarageCond_Ex'].fillna(0, inplace=True)

final_test['MiscFeature_TenC'].fillna(0, inplace=True)

final_test['SaleType_CWD'].fillna(0, inplace=True)

final_test['SaleCondition_AdjLand'].fillna(0, inplace=True)



final_test['Condition2_Artery'].fillna(0, inplace=True)

final_test['Condition2_PosN'].fillna(0, inplace=True)

final_test['Condition2_RRAe'].fillna(0, inplace=True)

final_test['Condition2_RRAn'].fillna(0, inplace=True)

final_test['RoofStyle_Shed'].fillna(0, inplace=True)

final_test['RoofMatl_ClyTile'].fillna(0, inplace=True)

final_test['RoofMatl_Metal'].fillna(0, inplace=True)

final_test['RoofMatl_Roll'].fillna(0, inplace=True)

final_test['Exterior1st_AsphShn'].fillna(0, inplace=True)

final_test['Exterior1st_CBlock'].fillna(0, inplace=True)

final_test['Exterior2nd_CBlock'].fillna(0, inplace=True)



final_test['Exterior2nd_Other'].fillna(0, inplace=True)

final_test['ExterCond_Po'].fillna(0, inplace=True)

final_test['Foundation_Wood'].fillna(0, inplace=True)

final_test['Heating_Floor'].fillna(0, inplace=True)

final_test['Heating_OthW'].fillna(0, inplace=True)

final_test['HeatingQC_Po'].fillna(0, inplace=True)

final_test['Electrical_FuseP'].fillna(0, inplace=True)

final_test['Functional_Sev'].fillna(0, inplace=True)

final_test['GarageQual_Ex'].fillna(0, inplace=True)

final_test['GarageCond_Ex'].fillna(0, inplace=True)

final_test['MiscFeature_TenC'].fillna(0, inplace=True)

final_test['SaleType_CWD'].fillna(0, inplace=True)

final_test['SaleCondition_AdjLand'].fillna(0, inplace=True)





# Check again for NaN again

final_test[pd.isnull(final_test).any(axis=1)]
final_train.head()
final_test.head()
plt.matshow(final_train.iloc[:,1:10].corr())
corr = final_train.iloc[:,10:20].corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
from pandas.plotting import scatter_matrix

scatter_matrix(final_train.iloc[:,1:5])

plt.show()
from pylab import rcParams

rcParams['figure.figsize'] = 10, 10

final_train.iloc[:,1:10].hist(bins=50)

plt.show()
final_train.columns
sns.pairplot(final_train.iloc[:,1:10],vars=['LotFrontage','LotArea','MasVnrArea'])

sns.plt.show()
from sklearn import feature_selection

fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=10)

final_train_fs = fs.fit_transform(final_train, y_train)

final_test_fs = fs.fit_transform(final_test,y_test)



np.set_printoptions(suppress=True, precision=2, linewidth=80)

#print (fs.get_support())

#print (fs.scores_)
print(final_train.columns[fs.get_support()].values)



print(final_test.columns[fs.get_support()].values)
for i in range(len(final_train.columns.values)):

    if fs.get_support()[i]:

        print(final_train.columns.values[i],'\t', fs.scores_[i] )
print(final_train_fs)
# This gives us back our dataframe with only the top 10% of features

final_train[final_train.columns[fs.get_support()].values].head()
from sklearn import metrics



def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True):

    y_pred = clf.predict(X)   

    if show_accuracy:

         print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)),"\n")

    if show_classification_report:

        print ("Classification report")

        print (metrics.classification_report(y, y_pred),"\n")

    if show_confusion_matrix:

        print ("Confusion matrix")

        print (metrics.confusion_matrix(y, y_pred),"\n")
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200)

rf.fit(final_train_fs, y_train)





mse = np.mean((y_test - rf.predict(final_test_fs))**2)

mse

# 1421902865.9038565

rmse = np.sqrt(mse)

rmse

# 36502.173390202166

pl.scatter(y_test, rf.predict(final_test_fs))

pl.plot(np.arange(8, 15), np.arange(8, 15))

pl.legend(loc="lower right")

pl.title("RandomForest Regression with scikit-learn")

pl.show()
# standardize the numeric independent predictor variables 

scaler = StandardScaler()



scaler.fit(final_train)

X_train_s = scaler.transform(final_train)



scaler.fit(final_test)

X_test_s = scaler.transform(final_test)
from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=30)

Y_sklearn = sklearn_pca.fit_transform(X_train_s)



print(sklearn_pca.explained_variance_ratio_)
Y_sklearn
#Explained variance

pca = sklearnPCA().fit(X_train_s)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.show()
# Factor analysis to reveal which variables contribute the most to the

# principal components

from sklearn.decomposition import FactorAnalysis

factor = FactorAnalysis(n_components=10, random_state=101).fit(final_train)

factor
print (pd.DataFrame(factor.components_,columns=final_train.columns))
pca = PCA().fit(final_train)

print ('Explained variance by component:', pca.explained_variance_ratio_)
print (pd.DataFrame(pca.components_,columns=final_train.columns))
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline



pca = PCA(n_components=2)

clf = LinearRegression() 



pipe = Pipeline([('pca', pca), ('logistic', clf)])

pipe.fit(X_train_s, y_train)

predictions = pipe.predict(X_test_s)
predictions
# feature elimination

model = RandomForestRegressor()

rfe = RFE(model, 10)

fit = rfe.fit(final_train, y_train)
#print("Num Features:",fit.n_features_)

#print("Selected Features:",fit.support_)

#print("Feature Ranking:",fit.ranking_)
# This is our original training independent variables, but selecting only

# those that have been selected by recursive feature elimination

final_train_rfe = final_train.loc[:,fit.support_]

final_train_rfe.head()
param_grid = {"n_estimators": [200, 500],

    "max_depth": [3, None],

    "max_features": [1, 3, 5, 10],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 3, 10],

    "bootstrap": [True, False]}



model = RandomForestRegressor(random_state=0)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

grid.fit(final_train, y_train)



print(grid.best_score_)

print(grid.best_params_)
# Use parameters selected during grid search above. Then, refit the model

regressor = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=None, max_features=10, 

                                  min_samples_leaf=1, min_samples_split=2, bootstrap=False)



regressor.fit(final_train, y_train)



regressor.score(final_train, y_train)

regressor.score(final_test, y_test)



y_pred = regressor.predict(final_test)



# Plot the predictions vs. actual prices

plt.figure(figsize=(8, 6))

plt.scatter(x=y_test, y=y_pred)

plt.xlim([0,600000])

plt.ylim([0,600000])

plt.plot([0,600000],[0,600000])

plt.show()
# Interpreting the model

feature_import = pd.DataFrame(data=regressor.feature_importances_, index=final_train.columns.values, columns=['values'])

feature_import = feature_import.iloc[0:10,:] # Select only the 10 most important features

feature_import.sort_values(['values'], ascending=False, inplace=True)

feature_import.transpose()
feature_import.index
plt.figure(figsize=(8, 6))

BP = sns.barplot(feature_import.index, feature_import['values'],

            data=feature_import, palette='BuGn')

plt.setp(BP.get_xticklabels(), rotation=90)



plt.show()