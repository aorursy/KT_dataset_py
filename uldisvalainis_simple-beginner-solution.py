# Importing the libraries and datasets

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_absolute_error



path = '../input/home-data-for-ml-course/'



dataset = pd.read_csv(path+'train.csv')

testset = pd.read_csv(path+'train.csv')

dataset.head(10)
sns.distplot(dataset['SalePrice']) #Pretty cool graph that shows distribution based on price
#Utilities has 1 different value through all columns, probably doesn't affect much! 

dataset['Utilities'].value_counts() 

dataset=dataset.drop(['Utilities'],axis=1)

testset=testset.drop(['Utilities'],axis=1)

dataset=dataset.drop(['Exterior2nd'],axis=1)#Exterior 1st and 2nd is basically the same twice

X1=testset.drop(['Exterior2nd'],axis=1)



print('Exterior2nd and Utilities dropped!')



#Splitting Target and test

X = dataset.drop('SalePrice', axis=1)

y= np.array(dataset['SalePrice']).reshape(-1,1)

#%% #Categorical and numerical columns



categorical_columns = (X.dtypes == 'object')

numerical_columns = (X.dtypes != 'object')



categorical_columns = list(categorical_columns[categorical_columns].index)

numerical_columns = list(numerical_columns[numerical_columns].index)

print('Categorical and numerical columns found!')

#%% # Taking care of missing data

from sklearn.impute import SimpleImputer 

cat_imp = SimpleImputer(missing_values = np.nan, strategy='most_frequent') #categorical

num_imp = SimpleImputer(missing_values = np.nan, strategy='mean') #numerical



#fitting

cat_imp = cat_imp.fit(X[categorical_columns])

num_imp = num_imp.fit(X[numerical_columns])



#Adding the missing values

def transform(X):

    X[categorical_columns] = cat_imp.transform(X[categorical_columns])

    X[numerical_columns] = num_imp.transform(X[numerical_columns])

    print('transformed!')



transform(X)

transform(X1)



# Get number of unique entries in each column with categorical data

dataset[categorical_columns].nunique().sort_values().tail()
#%%# Replacing categorical columns with Dummy variables



X = pd.get_dummies(data=X, columns=categorical_columns, drop_first=True)

X1 = pd.get_dummies(data=X1, columns=categorical_columns, drop_first=True) # Same for testset



#Aligning makes the upper 2 rows ok in any case!

X1, X = X1.align(X, join='inner', axis=1)

print('Got dummy variables and alligned to the test set!')

Xcol = X.columns





#Converting to DataFrame

X = pd.DataFrame(X)

X1 = pd.DataFrame(X1)



print('Missing data changed to most frequent and features and target devided into X and y')



# %% Scaling

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size = 0.15, random_state = 0)

                                                    

#Feature scaling trian and test for training data

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

sc_y = StandardScaler()

y_train = sc_y.fit_transform(y_train.reshape(-1,1))





#Feature scaling the test set

X1 = sc_X.fit_transform(X1)



print('Scaled and splitted into test and train!')
# %%# Fitting Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

regressor1 = RandomForestRegressor(n_estimators= 311,

                                 min_samples_split= 2,

                                 min_samples_leaf=1,

                                 max_features='sqrt',

                                 max_depth= None,

                                 bootstrap=False,

                                 random_state =0)

regressor1.fit(X_train,y_train.ravel())

print('RandomForestRegressor fitted as regressor1!')
#%%Fitting xgboost Regressor

from xgboost import XGBRegressor



regressor2 = XGBRegressor(objective ='reg:squarederror',n_estimators=3000,

                        min_samples_split=10,

                        min_samples_leaf=1,

                        max_depth=2,

                        learning_rate=0.02,

                        random_state =0)

regressor2.fit(X_train, y_train.ravel())

print('XGBRegressor fitted as regressor2!')
# %% Fitting LGBM regressor 

from lightgbm import LGBMRegressor

regressor3 = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       random_state =0)

regressor3.fit(X_train, y_train.ravel())

print('LGBMRegressor fitted as regressor3!')
#%% fitting Gradient Boosting Regressor



from sklearn.ensemble import GradientBoostingRegressor

regressor4= GradientBoostingRegressor(n_estimators=3000, 

                                learning_rate=0.05, 

                                max_depth=4, 

                                max_features='sqrt', 

                                min_samples_leaf=15, 

                                min_samples_split=10, 

                                loss='huber', 

                                random_state =0)

regressor4.fit(X_train, y_train.ravel())

print('GradientBoostingRegressor fitted as regressor4!')
# %%#Predicting, converting to moneys and printing

y_pred1= sc_y.inverse_transform(regressor1.predict(X_test))                             

print('Regressor 1:',mean_absolute_error(y_test,y_pred1))

y_pred2= sc_y.inverse_transform(regressor2.predict(X_test))

print('Regressor 2:',mean_absolute_error(y_test,y_pred2))

y_pred3= sc_y.inverse_transform(regressor3.predict(X_test))

print('Regressor 3:',mean_absolute_error(y_test,y_pred3))

y_pred4= sc_y.inverse_transform(regressor4.predict(X_test))

print('Regressor 4:',mean_absolute_error(y_test,y_pred4))

#Mean error for evaluation

print('Combo: ', mean_absolute_error(y_test, (y_pred1+y_pred2+y_pred3+y_pred4)/4))

#%% #Plot stuff so it looks like i've achived something usefull

#Looks like a rainbow

a = sns.regplot(y_test, y_pred1, color='blue',label='Random Forest' )

a.legend()

b = sns.regplot(y_test, y_pred2, color='red',label='Xboost' )

b.legend()

c = sns.regplot(y_test, y_pred3, color='purple',label='LGBM')

c.legend()

d = sns.regplot(y_test, y_pred4, color='orange',label='Gradient Boosting')

d.legend()

plt.show()

#Prediction graph thing? 

fig = plt.figure(figsize=(25,10))

plt.plot(y_pred1, color = 'Blue')



plt.plot(y_pred2, color = 'red')



plt.plot(y_pred3, color = 'purple')



plt.plot(y_pred4, color = 'orange')



plt.plot(y_test, color = 'black')

plt.show()



#Importance



fig, axs = plt.subplots(2, 2,figsize=(25,25))

important1 = regressor1.feature_importances_

axs[0, 0].bar(range(len(regressor1.feature_importances_)), regressor1.feature_importances_, color ='blue' )

important2 = regressor2.feature_importances_

axs[0, 1].bar(range(len(regressor2.feature_importances_)), regressor2.feature_importances_, color ='red' )

important4 = regressor4.feature_importances_

axs[1, 0].bar(range(len(regressor4.feature_importances_)), regressor4.feature_importances_, color ='orange' )

important3 = regressor3.feature_importances_

axs[1, 1].bar(range(len(regressor3.feature_importances_)), regressor3.feature_importances_, color ='purple' )
#Top25 importance

fig = plt.figure(figsize=(25,25))



plt.subplot(2, 2, 1)

feat_importances1 = pd.Series(important1, index=Xcol)

feat_importances1.nlargest(25).plot(kind='barh', color = 'blue')



plt.subplot(2, 2, 2)

feat_importances2 = pd.Series(important2, index=Xcol)

feat_importances2.nlargest(25).plot(kind='barh', color = 'red')



fig = plt.figure(figsize=(25,25))

plt.subplot(2, 2, 3)

feat_importances3 = pd.Series(important3, index=Xcol)

feat_importances3.nlargest(25).plot(kind='barh', color = 'orange')



plt.subplot(2, 2, 4)

feat_importances4 = pd.Series(important4, index=Xcol)

feat_importances4.nlargest(25).plot(kind='barh', color = 'purple')





plt.show()

#%% #Creating predictions for the submission



y_pred1= sc_y.inverse_transform(regressor1.predict(X1))                             

y_pred2= sc_y.inverse_transform(regressor2.predict(X1))

y_pred3= sc_y.inverse_transform(regressor3.predict(X1))

y_pred4= sc_y.inverse_transform(regressor4.predict(X1))
#%% #Printing out the submission

testset = pd.read_csv(path+'train.csv')

output = pd.DataFrame({'Id': testset.Id,

                       'SalePrice': ((y_pred1+y_pred2+y_pred3+y_pred4)/4)})

output.to_csv('submission.csv', index=False)