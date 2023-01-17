import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

print(os.listdir("../input"))
data_main_train = pd.read_csv('../input/train.csv',index_col='Id')



data_main_test = pd.read_csv('../input/test.csv',index_col='Id')
data_main_train.head()
data_main_test.head()
#so at first I need to separate classification data from linear.
linear_list = ['LotFrontage',

               'LotArea',

               'MasVnrArea',

               'BsmtFinSF1',

               'BsmtFinSF2',

               'BsmtUnfSF',

               'TotalBsmtSF',

               '1stFlrSF',

               '2ndFlrSF',

               'LowQualFinSF',

               'GrLivArea',

               'GarageArea',

               'WoodDeckSF',

               'OpenPorchSF',

               'EnclosedPorch',

               '3SsnPorch',

               'ScreenPorch',

               'PoolArea',

               'MiscVal',

               'SalePrice']



data_main_train_linear = data_main_train[linear_list]



data_main_test_linear = data_main_test[linear_list[:-1]]
data_main_train_linear.head()
data_main_test_linear.head()
#now I make classification columns dataset
class_list =  list()



for x in data_main_train.columns:

    if x not in linear_list:

        class_list.append(x)
data_main_train_class = data_main_train[class_list]



data_main_test_class = data_main_test[class_list]
data_main_train_class.head()
data_main_test_class.head()
#ok now I have to change all NA to 'Non Avaible in classification data
data_main_train_class.fillna(value='NonAvaible',inplace=True)



data_main_test_class.fillna(value='NonAvaible',inplace=True)
data_main_train_class.head()
data_main_test_class.head()
#lets see how many columns will make dummy_variables
len(pd.get_dummies(data_main_train_class,columns=class_list,drop_first=True).columns)
#ok, not perfect but still avaiable to calculate. What if I will delete all Year... columns?
data_main_train_class_noyears = data_main_train_class.drop(data_main_train_class[['YearBuilt','YearRemodAdd','YrSold']],

                                                           axis=1)



data_main_test_class_noyears = data_main_test_class.drop(data_main_test_class[['YearBuilt','YearRemodAdd','YrSold']],

                                                           axis=1)
noyear_cols = data_main_train_class_noyears.columns
len(pd.get_dummies(data_main_train_class_noyears,columns=noyear_cols,drop_first=True).columns)
#looks much better but what if the year built is very important information?

#I am gonna make 'age' column and add this to linear dataframe.
data_main_train_linear['Age'] = data_main_train_class['YrSold']-data_main_train_class['YearBuilt']

data_main_train_linear["Age"].head()
data_main_test_linear['Age'] = data_main_test_class['YrSold']-data_main_test_class['YearBuilt']

data_main_test_linear["Age"].head()
#ok if now we have all selected linear data- lets check the correlation!
plt.figure(figsize=(20,20))

sns.heatmap(data_main_train_linear.corr(),annot=True,fmt='.1f',linewidths=2)
#wow! Age has big correlation to the price! (What a suprise LOL)
#ok now we have two dataframes
data_main_train_class_noyears.head()
data_main_test_class_noyears.head()
data_main_train_linear.head()
data_main_test_linear.head()
#but I need dummy variables to classification data frame and rename both
data_class = pd.get_dummies(data_main_train_class_noyears,columns=noyear_cols,drop_first=True).copy()

data_linear = data_main_train_linear.copy()
data_class_test = pd.get_dummies(data_main_test_class_noyears,columns=noyear_cols,drop_first=True).copy()

data_linear_test = data_main_test_linear.copy()
data_class.head()
data_class_test.head()
data_linear.head()
data_linear_test.head()
#lets see some basic info
data_linear.info()
#there are 259 null data in LotFrontage
len(data_linear[data_linear['LotFrontage']==0])
#no zeros in LotFrontage so I am presuming that NA is just zero
data_linear['LotFrontage'] = data_linear['LotFrontage'].fillna(value=0)



data_linear_test['LotFrontage'] = data_linear_test['LotFrontage'].fillna(value=0)

#ok good, now lets check MasVnrArea
len(data_linear[data_linear['MasVnrArea']==0])
#well... NA is not zero. 

#MasVNrArea is Masonry veneer area in square feet so I cannot just fill by mean. I have to delete this...
data_linear.dropna(axis='rows',inplace=True)

data_linear_test.dropna(axis='rows',inplace=True)
data_linear.info()
#ok just 8 rows-not bad. Lets change dtype for 2 columns
data_linear.LotFrontage = data_linear.LotFrontage.astype(int)

data_linear.MasVnrArea = data_linear.MasVnrArea.astype(int)



data_linear_test.LotFrontage = data_linear_test.LotFrontage.astype(int)

data_linear_test.MasVnrArea = data_linear_test.MasVnrArea.astype(int)
data_linear.info()
data_linear.describe()
#ok lets see how many observations are 'weird' using OneClassSVM

#first scaling
from sklearn.preprocessing import StandardScaler



data_linear_noprice = data_linear.drop('SalePrice',axis=1)

data_linear_scaled = StandardScaler().fit_transform(data_linear_noprice)
from sklearn import svm



out_frac = 0.02

nu_estimate = 0.95*out_frac+0.05

ml = svm.OneClassSVM(kernel='rbf', gamma=1/len(data_linear_scaled), degree=3, nu=nu_estimate)

ml.fit(data_linear_scaled)

detect = ml.predict(data_linear_scaled)
outliners = np.where(detect==-1)

regular = np.where(detect==1)
len(outliners[0])
len(regular[0])
#Now I am choosing only regular values for linear dataframe
data_linear_noprice.head()
data_linear_noprice = data_linear_noprice.loc[regular[0]]
print(len(data_linear_noprice))

print(data_linear_noprice.shape)
#ok now lets simplify the dimensions- I can use classification data!
data_linear_noprice.dropna(axis='rows',inplace=True)

data_linear_noprice = pd.DataFrame(StandardScaler().fit_transform(data_linear_noprice),

                                   columns=data_linear_noprice.columns, index=data_linear_noprice.index)



data_linear_test.dropna(axis='rows',inplace=True)

data_linear_noprice_test = pd.DataFrame(StandardScaler().fit_transform(data_linear_test),

                                   columns=data_linear_test.columns, index=data_linear_test.index)
data_linear_noprice.head()
data_linear_noprice_test.head()
data_to_pca = pd.concat([data_linear_noprice, data_class], axis=1, sort=False).dropna(axis='rows')



data_to_pca_test = pd.concat([data_linear_noprice_test, data_class_test], axis=1, sort=False).dropna(axis='rows')
data_to_pca.head()
data_to_pca_test.head()
from sklearn.decomposition import PCA



pca = PCA(n_components=125)

data_pca = pca.fit_transform(data_to_pca)



data_pca_test = pca.fit_transform(data_to_pca_test)



print(pca.explained_variance_ratio_.sum())

print(data_pca.shape)
#so far so good
data_pca = pd.DataFrame(data = data_pca,index=data_to_pca.index)
data_pca_test = pca.fit_transform(data_pca_test)

data_pca_test = pd.DataFrame(data = data_pca_test,index=data_to_pca_test.index)
data_pca.head()
data_pca_test.head()
data_pca_price = pd.concat([data_pca, data_linear['SalePrice']], axis=1, sort=False).dropna(axis='rows')
data_pca_price.head()
data_pca_price['SalePrice'].mean()
data_pca_price['SalePrice'].std()
from sklearn.model_selection import cross_validate

from sklearn.model_selection import train_test_split





X = data_pca_price.drop('SalePrice',axis=1)

y = data_pca_price['SalePrice']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)





from sklearn.linear_model import LinearRegression



linmodel = LinearRegression()

linear_cv_results = cross_validate(linmodel, X, y, 

                            scoring=('r2', 'neg_mean_absolute_error'), cv=20,return_train_score=True)



print('Default LinearRegression results')

print(min(linear_cv_results['test_neg_mean_absolute_error']))

print(max(linear_cv_results['test_neg_mean_absolute_error']))

print(np.mean(linear_cv_results['test_neg_mean_absolute_error']))
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators=500, n_jobs=-1)

forest_reg_cv_results = cross_validate(forest_reg, X, y, 

                            scoring=('r2', 'neg_mean_absolute_error'), cv=10,return_train_score=True)



print('Default RandomForestRegressor results')

print(min(forest_reg_cv_results['test_neg_mean_absolute_error']))

print(max(forest_reg_cv_results['test_neg_mean_absolute_error']))

print(np.mean(forest_reg_cv_results['test_neg_mean_absolute_error']))
from sklearn.svm import SVR



svr_reg = SVR(gamma=0.01)



svr_reg_cv_results = cross_validate(svr_reg, X, y, 

                            scoring=('r2', 'neg_mean_absolute_error'), cv=10,return_train_score=True)



print('Default SVR results')

print(min(svr_reg_cv_results['test_neg_mean_absolute_error']))

print(max(svr_reg_cv_results['test_neg_mean_absolute_error']))

print(np.mean(svr_reg_cv_results['test_neg_mean_absolute_error']))
search_dict = {'C': [0.01,0.1,1,10,100,200,500,1000],

               'gamma': [0.1,0.01,0.001,0.0001],

               'epsilon': [0.0001,0.001,0.01,0.1,1,10],

               'degree': [1,2,3,4,5]}
from sklearn import model_selection





search_func = model_selection.RandomizedSearchCV(estimator=svr_reg,

                                           param_distributions=search_dict,

                                           n_jobs=-1,

                                           cv=10)

search_func_fit = search_func.fit(X_train,y_train)
print(search_func_fit.best_estimator_)
svr_reg2 = search_func_fit.best_estimator_

svr_reg2_cv_results = cross_validate(svr_reg2, X, y, 

                            scoring=('r2', 'neg_mean_absolute_error'), cv=10,return_train_score=True)



print('Improved SVR results')

print(min(svr_reg2_cv_results['test_neg_mean_absolute_error']))

print(max(svr_reg2_cv_results['test_neg_mean_absolute_error']))

print(np.mean(svr_reg2_cv_results['test_neg_mean_absolute_error']))
n_estimators = [500]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

n_jobs = [-1]





# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap,

               'n_jobs': n_jobs}



new_forest = RandomForestRegressor()

search_func_forest = model_selection.RandomizedSearchCV(estimator=new_forest,

                                           param_distributions=random_grid,

                                           n_jobs=-1,

                                           cv=10)

search_func_forest_fit = search_func_forest.fit(X_train,y_train)
print(search_func_forest_fit.best_estimator_)
forest_reg2 = search_func_forest_fit.best_estimator_

forest_reg2_cv_results = cross_validate(forest_reg2, X, y, 

                            scoring=('r2', 'neg_mean_absolute_error'), cv=10,return_train_score=True)



print('Improved RandomForestRegressor results')

print(min(forest_reg2_cv_results['test_neg_mean_absolute_error']))

print(max(forest_reg2_cv_results['test_neg_mean_absolute_error']))

print(np.mean(forest_reg2_cv_results['test_neg_mean_absolute_error']))
from xgboost import XGBRegressor





model_xgbr = XGBRegressor(colsample_bytree=0.4,

                 gamma=0.01,                 

                 learning_rate=0.07,

                 max_depth=3,

                 min_child_weight=1.5,

                 n_estimators=1000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42) 



model_xgbr_cv_results = cross_validate(model_xgbr, X, y, 

                            scoring=('r2', 'neg_mean_absolute_error'), cv=10,return_train_score=True)



print('Default XGBR results')

print(min(model_xgbr_cv_results['test_neg_mean_absolute_error']))

print(max(model_xgbr_cv_results['test_neg_mean_absolute_error']))

print(np.mean(model_xgbr_cv_results['test_neg_mean_absolute_error']))
parameters = {'nthread':[-1], #when use hyperthread, xgboost may become slower

              'objective':['reg:linear'],

              'learning_rate': [.03, 0.05, .07], #so called `eta` value

              'max_depth': [5, 6, 7],

              'min_child_weight': [4],

              'silent': [1],

              'subsample': [0.7],

              'colsample_bytree': [0.4,0.5,0.7],

              'n_estimators': [100,250,500],

              'gamma': [0.1,0.01,0.001,0.0001],

              'n_jobs': [-1]}





model_xgbr = XGBRegressor()

search_func_xgbr = model_selection.RandomizedSearchCV(estimator=model_xgbr,

                                           param_distributions=parameters,

                                           n_jobs=-1,

                                           cv=10)

search_func_xgbr_fit = search_func_xgbr.fit(X_train,y_train)
print(search_func_xgbr_fit.best_estimator_)
model_xgbr2 = search_func_xgbr_fit.best_estimator_

model_xgbr2_cv_results = cross_validate(model_xgbr2, X, y, 

                            scoring=('r2', 'neg_mean_absolute_error'), cv=10,return_train_score=True)



print('Improved XGBR results')

print(min(model_xgbr2_cv_results['test_neg_mean_absolute_error']))

print(max(model_xgbr2_cv_results['test_neg_mean_absolute_error']))

print(np.mean(model_xgbr2_cv_results['test_neg_mean_absolute_error']))
#fitting best model to test and save

model_xgbr2_fit = model_xgbr2.fit(X,y)

model_xgbr2_pred = model_xgbr2_fit.predict(data_pca_test)

data_save = pd.DataFrame(data=model_xgbr2_pred,index=data_pca_test.index,columns=['SalePrice'])

data_save.to_csv('submission.csv')
pd.read_csv('submission.csv',index_col='Id')