import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
your_path="../input/"
train_data=pd.read_csv(your_path+'train.csv')

test_data=pd.read_csv(your_path+'test.csv')

train_data.head()
train_data.describe()
train_data.info()
test_data.info()
train_data.hist(bins=50, figsize=(20,15))

plt.show()
f, ax = plt.subplots(figsize=(20, 15))

sns.heatmap(train_data.corr(),vmax=1.5,vmin=-1.5,cmap="BrBG",square=True,annot=True,fmt='0.1f')
train_data.columns
#SalePrice is strongly correlated with OverallQual, GrLivArea

#YearBuilt is strongly correlated to GarageYrBlt-we will consider YearBuilt as it has no missing values and zeros

#TotalBsmtSF and 1stFlrSF-we will consider 1stFlrSF as it has no missing values and zeros

#GrLivArea and TotRmsAbvGrd-we will consider GrLivArea as it has no missing values and zeros

#2ndFlrSF and GrLivArea-we will consider GrLivArea as it has no missing values and zeros

#BedroomAbvGr and TotRmsAbvGrd-we will consider GrLivArea as it has no missing values and zeros

# GarageCars and GarageArea-we will consider GarageArea as it has good distribution



# we will neglect Alley, Fence, MiscFeature, PoolQC as they have more than 90% missing values 

# we will neglect BsmtFinSF2, 3SsnPorch, BsmtHalfBath, EnclosedPorch, LowQualFinSF, MiscVal, PoolArea, ScreenPorch as they have more than 90%zeros

# we will neglect 2ndFlrSF, WoodDeckSF, BsmtFullBath,HalfBath, MasVnrArea, FireplaceQu, Fireplaces, OpenPorchSF as it has more than 50% zeros or missing values



# LotFrontage has 259 / 17.7% missing values Missing

# BsmtFinSF1 has 467 / 32.0% zeros Zeros

# so we are left with YearBuilt, 1stFlrSF, GrLivArea, GarageArea, 



#Storing test id for final submission

test_id=test_data['Id']

# exclude columns you don't want

train_data_selected=train_data[train_data.columns[~train_data.columns.isin(['Alley', 'Fence', 'MiscFeature', 'PoolQC','BsmtFinSF2',

                                                                            '3SsnPorch', 'BsmtHalfBath', 'EnclosedPorch', 'LowQualFinSF', 

                                                                            'MiscVal', 'PoolArea', 'ScreenPorch','GarageYrBlt','TotalBsmtSF',

                                                                           'TotRmsAbvGrd','BedroomAbvGr','GarageCars','2ndFlrSF','Id',

                                                                           'WoodDeckSF', 'BsmtFullBath', 'HalfBath', 'MasVnrArea', 

                                                                            'FireplaceQu', 'Fireplaces', 'OpenPorchSF','LotFrontage',

                                                                           'BsmtFinSF1'])]]

test_data_selected=test_data[test_data.columns[~test_data.columns.isin(['Alley', 'Fence', 'MiscFeature', 'PoolQC','BsmtFinSF2',

                                                                            '3SsnPorch', 'BsmtHalfBath', 'EnclosedPorch', 'LowQualFinSF', 

                                                                            'MiscVal', 'PoolArea', 'ScreenPorch','GarageYrBlt','TotalBsmtSF',

                                                                           'TotRmsAbvGrd','BedroomAbvGr','GarageCars','2ndFlrSF','Id',

                                                                           'WoodDeckSF', 'BsmtFullBath', 'HalfBath', 'MasVnrArea', 

                                                                            'FireplaceQu', 'Fireplaces', 'OpenPorchSF','LotFrontage',

                                                                           'BsmtFinSF1'])]]

print(train_data_selected.columns)

print(train_data_selected.shape)

print(test_data_selected.shape)

print(test_id.values)
#saleprice correlation matrix

k = 10 #number of variables for heatmap

corrmat=train_data.corr()

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_data[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
train_data_selected.hist(bins=50, figsize=(20,25))

plt.show()
#To select rows whose column value equals a scalar, some_value, use ==:

#df.loc[df['column_name'] == some_value]

for col in test_data_selected.columns:

    print(col)

    print(test_data_selected.loc[test_data_selected[col].isna()])
#is there any difference between nan, NaN, NAN, the answer is no



print(np.nan is np.NaN is np.NAN)

print(np.nan)

print(np.NaN)

print(np.NAN)

test_data_selected.info()
train_data_selected.info()
test_data_selected.mode(dropna=True)
#convert float to int

test_data_selected['BsmtUnfSF'] = test_data_selected['BsmtUnfSF'].fillna(0.0).astype(np.int64)

test_data_selected['GarageArea'] = test_data_selected['GarageArea'].fillna(0.0).astype(np.int64)



test_data_selected.info()
#create a separate DataFrame consisting of only these categorical features

cat_df = train_data_selected.select_dtypes(include=['object']).copy()

cat_df_test = test_data_selected.select_dtypes(include=['object']).copy()

#Let's also check the column-wise distribution of null values:

print(cat_df.isnull().sum())

print(cat_df.shape)

print(cat_df_test.shape)
#create a separate DataFrame consisting of only these continious features

num_df = train_data_selected.select_dtypes(include=['int64']).copy()

num_df_test = test_data_selected.select_dtypes(include=['int64']).copy()

#Let's also check the column-wise distribution of null values:

print(num_df.isnull().sum())

print(num_df_test.isnull().sum())

print(num_df.shape)

print(num_df_test.shape)
#method .value_counts() returns the frequency distribution of each category in the feature

#Then selecting the top category, which is the mode, with the .index attribute.

def fillna(col):

    col.fillna(col.value_counts().index[0], inplace=True)

    return col



cat_df_fillna=cat_df.apply(lambda col:fillna(col))



cat_df_test_fillna=cat_df_test.apply(lambda col:fillna(col))



num_df_fillna=num_df.apply(lambda col:fillna(col))



num_df_test_fillna=num_df_test.apply(lambda col:fillna(col))



cat_df_fillna.info()

print(cat_df_test_fillna.info())

print(cat_df_test_fillna.shape)

print(cat_df_fillna.shape)

print(cat_df_fillna.describe())

print(cat_df_test_fillna.describe())
#Note the change in label of columns MSZoning , HouseStyle , Electrical , KitchenQual , GarageQual ,SaleType 
for col in cat_df_fillna:

    print(cat_df_fillna[col].unique())

print("#################")

for col in cat_df_test_fillna:

    print(cat_df_test_fillna[col].unique())
for col in cat_df_fillna:

    sns.set()

    fig,ax=plt.subplots()

    sns.countplot(x=col, data=cat_df_fillna)

    sns.despine(offset=5,trim=True)

    fig.set_size_inches(5,5)
for col in num_df:

    fig,ax=plt.subplots()

    ax=sns.barplot(x=col,y=col,data=num_df,estimator=lambda x:len(x)/len(num_df)*100)

    ax.set(ylabel='Percent')
#from above plot it is clear that street, utilities, condition2, heating, functional columns are having less variation in data 

#so we will drop them from categorical list

cat_df_fillna=cat_df_fillna[cat_df_fillna.columns[~cat_df_fillna.columns.isin(['Street', 'Utilities', 'Condition2', 'Heating', 'Functional'])]]

cat_df_test_fillna=cat_df_test_fillna[cat_df_test_fillna.columns[~cat_df_test_fillna.columns.isin(['Street', 'Utilities', 'Condition2', 'Heating', 'Functional'])]]

#neglect BsmtUnfSF since having low variation in data

num_df=num_df[num_df.columns[~num_df.columns.isin(['BsmtUnfSF'])]]

num_df_test=num_df_test[num_df_test.columns[~num_df_test.columns.isin(['BsmtUnfSF'])]]

num_df.info()

print(cat_df_fillna.shape)

print(cat_df_test_fillna.shape)

print(num_df.shape)

print(num_df_test.shape)
#How to solve mismatch in train and test set after categorical dummy encoding?



#Assume you have identical feature's names in train and test dataset. 

#You can generate concatenated dataset from train and test, get dummies from concatenated dataset and split it to train and test back.

train_objs_num = len(cat_df_fillna)

dataset = pd.concat(objs=[cat_df_fillna, cat_df_test_fillna], axis=0)

dataset_preprocessed = pd.get_dummies(dataset,drop_first=True)

cat_cols = dataset_preprocessed[:train_objs_num]

cat_cols_test = dataset_preprocessed[train_objs_num:]
#cat_cols = pd.get_dummies(cat_df_fillna, drop_first=True)

#cat_cols_test = pd.get_dummies(cat_df_test_fillna, drop_first=True)

print(cat_cols.shape)

print(cat_cols_test.shape)

cat_cols.head()
dataset = pd.concat([num_df,cat_cols], axis=1 )

dataset_test = pd.concat([num_df_test,cat_cols_test], axis=1 )

print(dataset.info())

print(dataset.shape)

print(dataset_test.shape)



set(dataset.columns).intersection(set(dataset_test.columns))
y = dataset["SalePrice"].values

X = dataset.drop(['SalePrice'], axis=1)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
# Importing the packages for Decision Tree Classifier



from sklearn import tree

my_tree_one = tree.DecisionTreeRegressor(criterion="mse",splitter="best",min_samples_split=3, max_depth=5, random_state=101)

my_tree_one
# Fit the decision tree model on your features and label



my_tree_one = my_tree_one.fit(X_train, y_train)
# The feature_importances_ attribute make it simple to interpret the significance of the predictors you include



list(zip(X_train.columns,my_tree_one.feature_importances_))
# The accuracy of the model on Train data



print(my_tree_one.score(X_train, y_train))



# The accuracy of the model on Test data



print(my_tree_one.score(X_test, y_test))
# Visualize the decision tree graph



with open('tree.dot','w') as dotfile:

    tree.export_graphviz(my_tree_one, out_file=dotfile, feature_names=X_train.columns, filled=True)

    dotfile.close()

    

# You may have to install graphviz package using 

# conda install graphviz

# conda install python-graphviz

from graphviz import Source



with open('tree.dot','r') as f:

    text=f.read()

    plot=Source(text)

plot 
y_pred = my_tree_one.predict(X_test)
from sklearn.metrics import mean_squared_error,r2_score





#define root mean square log error-we take log of predicted and actual values, so basically what changes is 

#variance that you are measuring, RMSE is used when both predicted and actual values are huge number

#if both predicted and actual values are small number: RMSE and RMSLE is same

def RMSLE(y,pred):

    return np.sqrt(mean_squared_error(np.log1p(y+1),np.log1p(pred+1)))



def rmsle(y,y_pred):

    assert len(y)==len(y_pred)

    terms_to_sum=[(np.log1p(y_pred[i]+1)-np.log1p(y[i]+1))**2 for i,pred in enumerate(y_pred)]

    return (sum(terms_to_sum)/len(y))**0.5



y_train_pred=my_tree_one.predict(X_train)

rsquare=r2_score(y_train,y_train_pred)

colcount=dataset.columns.size

n=len(X_train)

train_rmse=np.sqrt(mean_squared_error(y_train, y_train_pred))

test_rmse=np.sqrt(mean_squared_error(y_test, y_pred))



print("testing on training data")

print("RSquare: "+(str)(rsquare))

print("Adjusted RSquare: "+(str)(1-(1-rsquare)*(n-1)/(n-colcount-1)))

print("RMSE: "+str(train_rmse))

print("RMSLE: ",RMSLE(y_train, y_train_pred))

print("rmse: ",rmsle(y_train, y_train_pred))



print("testing on test data")

print("RMSE: "+str(test_rmse))

print("RMSLE: ",RMSLE(y_test, y_pred))



if rmsle(y_train, y_train_pred)*1.1 < RMSLE(y_test, y_pred):

    print("since test rmse is more than 10% of train rmse so model is overfitted")

else:

     print("since test rmse is less than 10% of train rmse so model is ok")

        

#lets check the goodness of fit with the prediction visualized as line

fig,ax=plt.subplots()

ax.scatter(y_test,y_pred,edgecolors=(0,0,0))

ax.plot([y_test.min(),y_test.max()],[y_pred.min(),y_pred.max()],'k--',lw=4)

ax.set_xlabel('Actual')

ax.set_ylabel('Predicted')



# Different parameters we want to test



max_depth = [3,5,7,10,12,15] 

splitter=['best','random']

min_samples_split=[2,3,4,5,6]

min_samples_leaf = [1, 2, 4]



# Importing GridSearch



from sklearn.model_selection import GridSearchCV



# Building the model



my_tree_three = tree.DecisionTreeRegressor(random_state=101)



# Cross-validation tells how well a model performs on a dataset using multiple samples of train data

grid = GridSearchCV(estimator = my_tree_three, cv=5, 

                    param_grid = dict(max_depth = max_depth, splitter=splitter,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf))



grid.fit(X_train,y_train)



# Best accuracy score



print(grid.best_score_)



# Best parameters for the model



print(grid.best_params_)
# Building the model based on new parameters



my_tree_three = tree.DecisionTreeRegressor(criterion="mse",splitter="best",min_samples_split=6, max_depth=10,min_samples_leaf=2, random_state=101)

my_tree_three.fit(X_train,y_train)

# Accuracy Score for new model



my_tree_three.score(X_train,y_train)
y_pred = my_tree_three.predict(X_test)

# The accuracy of the model on Test data



print(my_tree_three.score(X_test, y_test))
# Building and fitting Random Forest



from sklearn.ensemble import RandomForestRegressor



forest = RandomForestRegressor(criterion = 'mse',  n_estimators = 100, max_depth = 10, random_state = 101)
# Fitting the model on Train Data



my_forest = forest.fit(X_train, y_train)
# Print the accuracy score of the fitted random forest



print(my_forest.score(X_train, y_train))
# Making predictions



y_pred = my_forest.predict(X_test)



# The accuracy of the model on Test data



print(my_forest.score(X_test, y_test))
list(zip(X_train.columns,my_forest.feature_importances_))
from sklearn.metrics import mean_squared_error,r2_score



y_train_pred=my_forest.predict(X_train)

rsquare=r2_score(y_train,y_train_pred)

colcount=dataset.columns.size

n=len(X_train)

train_rmse=np.sqrt(mean_squared_error(y_train, y_train_pred))

test_rmse=np.sqrt(mean_squared_error(y_test, y_pred))



print("testing on training data")

print("RSquare: "+(str)(rsquare))

print("Adjusted RSquare: "+(str)(1-(1-rsquare)*(n-1)/(n-colcount-1)))

print("RMSE: "+str(train_rmse))

print("RMSLE: ",RMSLE(y_train, y_train_pred))

print("rmse: ",rmsle(y_train, y_train_pred))



print("testing on test data")

print("RMSE: "+str(test_rmse))

print("RMSLE: ",RMSLE(y_test, y_pred))



if rmsle(y_train, y_train_pred)*1.1 < RMSLE(y_test, y_pred):

    print("since test rmse is more than 10% of train rmse so model is overfitted")

else:

     print("since test rmse is less than 10% of train rmse so model is ok")

        

#lets check the goodness of fit with the prediction visualized as line

fig,ax=plt.subplots()

ax.scatter(y_test,y_pred,edgecolors=(0,0,0))

ax.plot([y_test.min(),y_test.max()],[y_pred.min(),y_pred.max()],'k--',lw=4)

ax.set_xlabel('Actual')

ax.set_ylabel('Predicted')



# Different parameters we want to test



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 50)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]



random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



# Importing RandomizedSearchCV



from sklearn.model_selection import RandomizedSearchCV



forest_two = RandomForestRegressor()



# Fitting 3 folds for each of 100 candidates, totalling 300 fits

rf_random = RandomizedSearchCV(estimator = forest_two, param_distributions = random_grid, 

                               n_iter = 10, cv = 3, verbose=2, random_state=42)



rf_random.fit(X_train,y_train)



print(rf_random.best_params_)



print(rf_random.best_score_)
# Building the model based on new parameters



forest_tree = RandomForestRegressor(criterion = 'mse',  n_estimators = 934, max_depth = 60, min_samples_split=2 , min_samples_leaf=1 , max_features='sqrt', bootstrap=False, random_state = 101)

forest_tree.fit(X_train,y_train)

# Accuracy Score for new model



forest_tree.score(X_train,y_train)
y_pred = forest_tree.predict(X_test)

# The accuracy of the model on Test data



print(forest_tree.score(X_test, y_test))
#It is kind of acceptable model

#prediction on test_data

y_pred_test = forest_tree.predict(dataset_test)



submission = pd.DataFrame(columns=['Id', 'SalePrice'])

submission.Id = test_id

submission.SalePrice = y_pred_test

submission.to_csv('HouseResult.csv', index=False)

#Preprocessing- scaling the feature is important factor for KNN

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))



x_train_scaled=scaler.fit_transform(X_train)

X_train=pd.DataFrame(x_train_scaled)



x_test_scaled=scaler.fit_transform(X_test)

X_test=pd.DataFrame(x_test_scaled)



#let us have a look at error rate for diffrent k values



from sklearn import neighbors



def rmse_knn(k):

    if(k==0):

        k=1

    model=neighbors.KNeighborsRegressor(n_neighbors=k)

    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    error=np.sqrt(mean_squared_error(y_test,pred))

    return error



rmse_val=[]

for k in range(50):

    rmse_val.append(rmse_knn(k))

    

curve=pd.DataFrame(rmse_val)

curve.plot()


#let us also implement gridsearchcv

algorithm =['auto', 'ball_tree', 'kd_tree', 'brute']

weights=['uniform','distance']



params={'n_neighbors':[2,3,4,5,6,7,8,9],'algorithm':algorithm,'weights':weights}

knn=neighbors.KNeighborsRegressor()

model=GridSearchCV(knn,params,cv=5)

model.fit(X_train,y_train)

model.best_params_

model=neighbors.KNeighborsRegressor(n_neighbors=9,algorithm='auto',weights='distance')

model.fit(X_train,y_train)

# Accuracy Score for new model

print(model.score(X_train,y_train))



y_pred=model.predict(X_test)

# The accuracy of the model on Test data

print(model.score(X_test, y_test))
# Bagged Decision Trees for Regression



from sklearn.ensemble import BaggingRegressor

from sklearn import model_selection



cart = tree.DecisionTreeRegressor()

cart.fit(X_train,y_train)

print(cart.score(X_test,y_test))

print(cart.score(X_train,y_train))

num_trees = 100

#max_sample=0.5 define that each bag contain 50% of training data

model = BaggingRegressor(base_estimator=cart,max_samples=0.5, n_estimators=num_trees, random_state=7)

results = model_selection.cross_val_score(model, X_train, y_train, cv=5)

model.fit(X_train,y_train)

print(model.score(X_test,y_test))

print(model.score(X_train,y_train))

print(results)
# AdaBoost Regression



from sklearn.ensemble import AdaBoostRegressor

seed = 7

num_trees = 70

model = AdaBoostRegressor(n_estimators=num_trees, random_state=seed)

results = model_selection.cross_val_score(model, X_train, y_train, cv=5)

model.fit(X_train,y_train)

print(model.score(X_test,y_test))

print(model.score(X_train,y_train))

print(results.mean())
# GradientBoost Regression



from sklearn.ensemble import GradientBoostingRegressor

seed = 7

params={'n_estimators':100,'max_depth':4,'min_samples_split':2,'learning_rate':0.01,'loss':'ls','random_state':seed}

model1= GradientBoostingRegressor(max_depth=4,n_estimators=1,min_samples_split=2,learning_rate=0.01,loss='ls',random_state=seed)

model2 = GradientBoostingRegressor(**params)

results = model_selection.cross_val_score(model2, X_train, y_train, cv=5)

model1.fit(X_train,y_train)

model2.fit(X_train,y_train)

print(model2.score(X_test,y_test))

print(model2.score(X_train,y_train))

print(results.mean())



#predictions

y_pred1=model1.predict(X_test)

y_pred2=model2.predict(X_test)



#Plot Training and Test Deviance

test_score=np.zeros((params['n_estimators'],),dtype=np.float64)

for i,y_pred in enumerate(model2.staged_predict(X_test)):

    test_score[i]=model2.loss_(y_test,y_pred)

    

plt.plot(np.arange(params['n_estimators'])+1,model2.train_score_,'b-',label='Training set Deviance')

plt.plot(np.arange(params['n_estimators'])+1,test_score,'r-',label='Test set Deviance')

plt.legend(loc='upper right')

plt.xlabel('Boosting Iterations')

plt.ylabel('Deviance')



#after boosting iteration 20 the loss of test data is more than loss of training data and model started overfitting