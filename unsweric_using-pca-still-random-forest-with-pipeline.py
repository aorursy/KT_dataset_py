import numpy as np

import pandas as pd

data_train=pd.read_csv('train.csv')

target_data=data_train['SalePrice']

data_train=data_train.drop('SalePrice', axis = 1)





data_test=pd.read_csv('test.csv')

combine=[data_train,data_test]

data = pd.concat(combine)
data.head()
data.info()
data.describe(include=['O'])
raw_data=data.drop('Id', axis = 1)
#drop "Alley" as it has only 91 entries

raw_data=raw_data.drop('Alley', axis = 1)
# replace miss values of 'LotFrontage' with its median

raw_data.loc[ raw_data.LotFrontage.isnull(),'LotFrontage'] =raw_data['LotFrontage'].median()

     
# replace miss values of 'MasVnrType' with its mode

print(raw_data['MasVnrType'].mode())

raw_data.loc[ raw_data.MasVnrType.isnull(),'MasVnrType'] ='None'
# replace miss values of 'MasVnrArea' with its mode

print(raw_data['MasVnrArea'].mode())

raw_data.loc[ raw_data.MasVnrArea.isnull(),'MasVnrArea'] =0.0

# replace miss values of 'BsmtQual' with its mode

print(raw_data['BsmtQual'].mode())

raw_data.loc[ raw_data.BsmtQual.isnull(),'BsmtQual'] ='TA'
# replace miss values of 'BsmtCond' with its mode

print(raw_data['BsmtCond'].mode())

raw_data.loc[ raw_data.BsmtCond.isnull(),'BsmtCond'] ='TA'
# replace miss values of 'BsmtExposure' with its mode

print(raw_data['BsmtExposure'].mode())

raw_data.loc[ raw_data.BsmtExposure.isnull(),'BsmtExposure'] ='No'
# replace miss values of 'BsmtFinType1' with its mode

print(raw_data['BsmtFinType1'].mode())

raw_data.loc[ raw_data.BsmtFinType1.isnull(),'BsmtFinType1'] ='Unf'
# replace miss values of 'BsmtFinType2' with its mode

print(raw_data['BsmtFinType2'].mode())

raw_data.loc[ raw_data.BsmtFinType2.isnull(),'BsmtFinType2'] ='Unf'
# replace miss values of 'BElectrical' with its mode

print(raw_data['Electrical'].mode())

raw_data.loc[ raw_data.Electrical.isnull(),'Electrical'] ='SBrkr'
#drop "FireplaceQu" as it has only 770 entries

raw_data=raw_data.drop('FireplaceQu', axis = 1)
# replace miss values of 'GarageType' with its mode

print(raw_data['GarageType'].mode())

raw_data.loc[ raw_data.GarageType.isnull(),'GarageType'] ='Attchd'
# replace miss values of 'GarageYrBlt' with its mode

print(raw_data['GarageYrBlt'].mode())

raw_data.loc[ raw_data.GarageYrBlt.isnull(),'GarageYrBlt'] =2005.0
# replace miss values of 'GarageQual' with its mode

print(raw_data['GarageQual'].mode())

raw_data.loc[ raw_data.GarageQual.isnull(),'GarageQual'] ='TA'
# replace miss values of 'GarageFinish' with its mode

print(raw_data['GarageFinish'].mode())

raw_data.loc[ raw_data.GarageFinish.isnull(),'GarageFinish'] ='Unf'
# replace miss values of 'GarageCond' with its mode

print(raw_data['GarageCond'].mode())

raw_data.loc[ raw_data.GarageCond.isnull(),'GarageCond'] ='TA'
#drop "PoolQC " as it has only 7 entries

raw_data=raw_data.drop('PoolQC', axis = 1)
#drop "Fence " as it has only 281 entries

raw_data=raw_data.drop('Fence', axis = 1)
#drop "MiscFeature  " as it has only 54 entries

raw_data=raw_data.drop('MiscFeature', axis = 1)
# replace miss values of 'BsmtFinSF1' with its mode

print(raw_data['BsmtFinSF1'].mode())

raw_data.loc[ raw_data.BsmtFinSF1.isnull(),'BsmtFinSF1'] =0.0
# replace miss values of 'BsmtFinSF2' with its mode

print(raw_data['BsmtFinSF2'].mode())

raw_data.loc[ raw_data.BsmtFinSF2.isnull(),'BsmtFinSF2'] =0.0
# replace miss values of 'BsmtFullBath' with its mode

print(raw_data['BsmtFullBath'].mode())

raw_data.loc[ raw_data.BsmtFullBath.isnull(),'BsmtFullBath'] =0.0
# replace miss values of 'BsmtHalfBath' with its mode

print(raw_data['BsmtHalfBath'].mode())

raw_data.loc[ raw_data.BsmtHalfBath.isnull(),'BsmtHalfBath'] =0.0
# replace miss values of 'BsmtUnfSF' with its mode

print(raw_data['BsmtUnfSF'].mode())

raw_data.loc[ raw_data.BsmtUnfSF.isnull(),'BsmtUnfSF'] =0.0
# replace miss values of 'Exterior1st' with its mode

print(raw_data['Exterior1st'].mode())

raw_data.loc[ raw_data.Exterior1st.isnull(),'Exterior1st'] ='VinylSd'
# replace miss values of 'Exterior2nd ' with its mode

print(raw_data['Exterior2nd'].mode())

raw_data.loc[ raw_data.Exterior2nd .isnull(),'Exterior2nd'] ='VinylSd'
# replace miss values of 'Functional' with its mode

print(raw_data['Functional'].mode())

raw_data.loc[ raw_data.Functional.isnull(),'Functional'] ='Typ'
# replace miss values of 'GarageArea' with its mode

print(raw_data['GarageArea'].mode())

raw_data.loc[ raw_data.GarageArea.isnull(),'GarageArea'] =0.0
# replace miss values of 'GarageCars' with its mode

print(raw_data['GarageCars'].mode())

raw_data.loc[ raw_data.GarageCars.isnull(),'GarageCars'] =2.0
# replace miss values of 'GarageCars' with its mode

print(raw_data['GarageCars'].mode())

raw_data.loc[ raw_data.GarageCars.isnull(),'GarageCars'] =2.0
# replace miss values of 'KitchenQual' with its mode

print(raw_data['KitchenQual'].mode())

raw_data.loc[ raw_data.KitchenQual.isnull(),'KitchenQual'] ='TA'
# replace miss values of 'MSZoning' with its mode

print(raw_data['MSZoning'].mode())

raw_data.loc[ raw_data.MSZoning.isnull(),'MSZoning'] ='RL'
# replace miss values of 'SaleType' with its mode

print(raw_data['SaleType'].mode())

raw_data.loc[ raw_data.SaleType.isnull(),'SaleType'] ='WD'
# replace miss values of 'TotalBsmtSF' with its mode

print(raw_data['TotalBsmtSF'].mode())

raw_data.loc[ raw_data.TotalBsmtSF.isnull(),'TotalBsmtSF'] =0.0
# replace miss values of 'Utilities' with its mode

print(raw_data['Utilities'].mode())

raw_data.loc[ raw_data.Utilities.isnull(),'Utilities'] ='AllPub'
raw_data.info()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

numerical = list(raw_data.select_dtypes(include=['int64']).columns.values)+list(raw_data.select_dtypes(include=['float64']).columns.values)

print(numerical)

raw_data[numerical] = scaler.fit_transform(raw_data[numerical])
raw_data.head()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



categorical=list(raw_data.select_dtypes(include=['object']).columns.values)

print(categorical)

for value in categorical:

    le.fit(raw_data[value])

    raw_data[value]=le.transform(raw_data[value])

raw_data.head()

print(raw_data.shape)
from sklearn.decomposition import PCA



pca = PCA(n_components=74)

pca.fit(raw_data)

ratio=pca.explained_variance_ratio_

print(ratio)



a=0

for i in range(74):

    a=a+ratio[i]

    if a>0.99:

        print(i)

        break
pca = PCA(n_components=40)

pca.fit(raw_data)

raw_data_pca = pca.transform(raw_data)
from sklearn.cross_validation import train_test_split

feature_train_all=raw_data_pca[:1460]

feature_test=raw_data_pca[1460:]

X_train, X_test, y_train, y_test = train_test_split(feature_train_all,target_data, test_size = 0.2, random_state = 0)

# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score

from sklearn.metrics import r2_score

import numpy as np

import pandas as pd



def train_predict(learner, X_train, y_train, X_test, y_test): 

    '''

    inputs:

       - learner: the learning algorithm to be trained and predicted on

       - sample_size: the size of samples (number) to be drawn from training set

       - X_train: features training set

       - y_train: income training set

       - X_test: features testing set

       - y_test: income testing set

    '''

    

    results = {}

    learner = learner.fit(X_train,y_train)

    predictions_test = learner.predict(X_test) 

    # TODO: Compute accuracy on test set

    results['R2_test'] = r2_score(y_test,predictions_test)

        

    # Return the results

    return results
from sklearn import linear_model

from sklearn import svm

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import RandomForestRegressor



# TODO: Initialize the three models

clf_A = linear_model.Ridge (random_state=1)

clf_B = svm.SVR(kernel='linear')

clf_C = svm.SVR(kernel='rbf')

clf_D=AdaBoostRegressor(random_state=1)

clf_E=RandomForestRegressor(random_state=1)





# Collect results on the learners

results = {}

for clf in [clf_A, clf_B, clf_C, clf_D, clf_E]:

    clf_name = clf.__class__.__name__

    results[clf_name] = train_predict(clf, X_train, y_train, X_test, y_test)

    
print(results)
from sklearn.metrics import make_scorer

from sklearn.ensemble import RandomForestRegressor

from sklearn import grid_search

from sklearn.metrics import r2_score

from sklearn.model_selection import ShuffleSplit



def fit_model(X, y):

    """ Performs grid search over the 'max_depth' parameter for a 

        decision tree regressor trained on the input data [X, y]. """

    

    # Create cross-validation sets from the training data

    # For Balance Data

    #cv_sets = ShuffleSplit(X.shape[0], n_splits = 10, test_size = 0.20, random_state = 0)

    

    

    

    # TODO: Create a decision tree regressor object

    regressor =  RandomForestRegressor(random_state=0)



    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10

    params = {'n_estimators':range(10,200,10)}



    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 

    scoring_fnc = make_scorer(r2_score)



    # TODO: Create the grid search object

    grid =grid_search.GridSearchCV(regressor, params,scoring_fnc)

    

    # Fit the grid search object to the data to compute the optimal model

    grid = grid.fit(X, y)



    # Return the optimal model after fitting the data

    return grid.best_estimator_
model=fit_model(X_train, y_train,)
print(model.get_params()['n_estimators'])
y_predict=model.predict(X_train)
print(r2_score(y_train,y_predict))
Y_pred=model.predict(feature_test)
submission = pd.DataFrame({

        "Id": data_test["Id"],

        "SalePrice": Y_pred

    })

submission.to_csv('submission.csv', index=False)