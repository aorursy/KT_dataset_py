#Libraries

import pandas as pd

import numpy as np

import scipy

import matplotlib.pyplot as mp

import seaborn as sns
#Function to calculate Percentage_error

def percentage_error(actual, predicted):

    """

    Function to calculate the percentage error if the actual value is zero so that the final MAPE will be infinity

    

    Parameters Taken: Actual test labels

    

    Return the the np array of residues

    """

    res = np.empty(actual.shape)

    for j in range(actual.shape[0]):

        if actual[j] != 0:

            res[j] = (actual[j] - predicted[j]) / actual[j]

        else:

            res[j] = predicted[j] / np.mean(actual)

    return res
#Function to Calculate RMSE

def rmse_calculator(actual, predicted):

    """

    Function to calculate the Root Mean Square Error for regression problem

    """

    from math import sqrt

    squared_error = 0

    for i in range(actual.shape[0]):

        squared_error += (actual[i] - predicted[i])**2

    rmse = sqrt(squared_error/actual.shape[0])

    return rmse
#Function to evaluate metrics

def evaluate(model, test_features, test_labels):

    """

    Function to calculate accuracy

    

    Parameters Taken: Trained_model, Test features and test labels

    

    Returns

    RMSE, MAPE values

    """

    from sklearn import metrics

    predictions = model.predict(test_features)

    mape = 100 * np.mean(np.abs(percentage_error(np.asarray(test_labels), np.asarray(predictions))))

    RMSE = rmse_calculator(np.asarray(test_labels), np.asarray(predictions))

    print('Model Performance')

    print('RMSE: {}'.format(round(RMSE, 2)))

    print('MAPE = {}'.format(round(mape, 2)))

    print('r2: {}'.format(round(metrics.r2_score(test_labels, predictions), 2)))
#Loading Data

train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
#total number of rows and columns

train.shape
#Availability of Data

train.columns
#Desciption about mean,max and min value

train.describe()
#Datatype of Columns in Pandas Data Frame

train.info()
#Missing Value Investigation

null_columns = train.columns[train.isnull().any()]

train[null_columns].isnull().sum().sort_values(ascending=False)
#Dropping columns

train=train.drop(columns=['PoolQC','MiscFeature','Fence','Alley'])
#Filling the Missing Values

Fill_na = ['BsmtQual', 'FireplaceQu', 'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'MasVnrType', 'MasVnrArea',

         'BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','Electrical']



for item in Fill_na:

    train[item] = train[item].fillna(train[item].mode()[0])
#Filling the Missing values with mean

train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())

train['GarageYrBlt']=train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())
#Checking for Missing values

null_columns = train.columns[train.isnull().any()]

train[null_columns].isnull().sum().sort_values(ascending=False)
train.info()
#Number of Unique Elements [nunique]

for col in train.columns:

    print(col, train[col].nunique())
#Checking for Duplicate Rows

duplicate_rows=train[train.duplicated()]

print(duplicate_rows.shape)
#Finding Outliers using scatter plot

mp.scatter(train['GrLivArea'],train['SalePrice'])
#Finding outliers using boxplot

sns.boxplot(x=train['SalePrice'])
#Removing outliers using IQR

def fun(a):

    q1=float(train['SalePrice'].quantile([0.25]))

    q3=float(train['SalePrice'].quantile([0.75]))

    iqr=q3-q1

    print(q3)

    lower_range = q1-1.5*iqr

    upper_range = q3+1.5*iqr

    for i in a:

        if(i<q1-1.5*iqr or i>q3+1.5*iqr):

            train.drop(train[ (train.SalePrice >= upper_range) | (train.SalePrice <= lower_range) ].index , inplace=True)

            

fun(train['SalePrice']);
#After Removing Outliers 

sns.boxplot(x=train['SalePrice'])

sns.set(rc={'figure.figsize':(10,5)})

mp.scatter(train['GrLivArea'],train['SalePrice'])
train.shape
#Printing columns of object Datatype

for col in train.columns:

    if train[col].dtype == 'object':

        print(col)
#Printing columns of integer Datatype

for col in train.columns:

    if train[col].dtype == 'int64':

        print(col)
features=[['Id','MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

           'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',

           'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch',

           'PoolArea','MiscVal','MoSold','YrSold']]

target=['SalePrice']

# Separating out the features

x = train[['Id','MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

           'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',

           'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch',

           'PoolArea','MiscVal','MoSold','YrSold']]

# Separating out the target

y = train[['SalePrice']]

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

# create a base classifier used to evaluate a subset of attributes

model = LogisticRegression()

# create the RFE model and select 10 attributes

rfe = RFE(model, 10)

rfe = rfe.fit(x,y)

# summarize the selection of the attributes

print(rfe.support_)

print(rfe.ranking_)
#Dropping columns with respect to result of feature engineering

train=train.drop(columns=['MSSubClass','OverallQual','OverallCond','BsmtFinSF2','LowQualFinSF','BsmtFullBath','BsmtHalfBath',

                         'FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','1stFlrSF',

                         'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold'])
train.shape


train['Street'] = train['Street'].replace({"Grvl":0,"Pave":1})

train['CentralAir']=train['CentralAir'].replace({"N":0,"Y":1})

train['PavedDrive'] = train['PavedDrive'].replace({"Y":0,"P":1,"N":2})

train['LandSlope'] = train['LandSlope'].replace({"Gtl":0,"Mod":1,"Sev":2})

train['LotShape'] = train['LotShape'].replace({"Reg":0,"IR1":1,"IR2":2,"IR3":3})

train['GarageFinish']= train['GarageFinish'].replace({"Fin":0,"RFn":1,"Unf":2,"NA":3})

train['LandContour'] = train['LandContour'].replace({"Lvl":0,"Bnk":1,"HLS":2,"Low":3})

train['Utilities'] = train['Utilities'].replace({"AllPub":0,"NoSewr":1,"NoSeWa":2,"ELO":3})

train['BsmtExposure']=train['BsmtExposure'].replace({"Gd":0,"Av":1,"Mn":2,"No":3,"NA":4})

train['LotConfig']=train['LotConfig'].replace({"Inside":0,"Corner":1,"CulDSac":2,"FR2":3,"FR3":4})

train['BldgType']=train['BldgType'].replace({"1Fam":0,"2fmCon":1,"Duplex":2,"TwnhsE":3,"Twnhs":4})

train['Electrical'] = train['Electrical'].replace({"SBrkr":0,"FuseA":1,"FuseF":2,"FuseP":3,"Mix":4})

train['Heating'] = train['Heating'].replace({"Floor":0,"GasA":1,"GasW":2,"Grav":3,"OthW":3,"Wall":4})

train['MasVnrType']=train['MasVnrType'].replace({"BrkCmn":0,"BrkFace":1,"CBlock":2,"None":3,"Stone":4})

train['RoofStyle']= train['RoofStyle'].replace({"Flat":0,"Gable":1,"Gambrel":2,"Hip":3,"Mansard":4,"Shed":5})

train['SaleCondition'] = train['SaleCondition'].replace({"Normal":0,"Abnorml":1,"AdjLand":2,"Alloca":3,"Family":4,"Partial":5})

train['Foundation'] = train['Foundation'].replace({"BrkTil":0,"CBlock":1,"PConc":3,"Slab":4,"Stone":5,"Wood":6})

train['GarageType']=train['GarageType'].replace({"2Types":0,"Attchd":1,"Basment":2,"BuiltIn":3,"CarPort":4,"Detchd":5,"NA":6})

train['MSZoning'] = train['MSZoning'].replace({"A":0,"C (all)":1,"FV":2,"I":3,"RH":4,"RL":5,"RP":6,"RM":7})

train['Functional']= train['Functional'].replace({"Typ":0,"Min1":1,"Min2":2,"Mod":3,"Maj1":4,"Maj2":5,"Sev":6,"Sal":7})

train['RoofMatl'] = train['RoofMatl'].replace({"ClyTile":0,"CompShg":1,"Membran":2,"Metal":3,"Roll":4,"Tar&Grv":5,"WdShake":6,

                                               "WdShngl":7})

train['HouseStyle'] = train['HouseStyle'].replace({ "1Story":0,"1.5Fin":1,"1.5Unf":2,"2Story":3,"2.5Fin":4,

                                                   "2.5Unf":5,"SFoyer":6,"SLvl":7})

train['SaleType']=train['SaleType'].replace({"WD":0,"CWD":1,"VWD":2,"New":3,"COD":4,

                                             "Con":5,"ConLw":6,"ConLI":7,"ConLD":8,"Oth":9})

train['Exterior1st'] = train['Exterior1st'].replace({"AsbShng":0,"AsphShn":1,"BrkComm":2,"BrkFace":3,"CBlock":4,"CemntBd":5,

                                "HdBoard":6,"ImStucc":7,"MetalSd":8,"Other":9,"Plywood":10,"PreCast":11,"Stone":12,

                                "Stucco":13,"VinylSd":14,"Wd Sdng":15,"WdShing":16})

train['Exterior2nd'] = train['Exterior2nd'].replace({"AsbShng":0,"AsphShn":1,"Brk Cmn":2,"BrkFace":3,"CBlock":4,"CmentBd":5,

                                "HdBoard":6,"ImStucc":7,"MetalSd":8,"Other":9,"Plywood":10,"PreCast":11,"Stone":12,

                                "Stucco":13,"VinylSd":14,"Wd Sdng":15,"WdShing":16,"Wd Shng":17})

train['Neighborhood']=train['Neighborhood'].replace({"Blmngtn":0,"Blueste":1,"BrDale":2,"BrkSide":3,"ClearCr":4,"CollgCr":5,

                                                     "Crawfor":6,"Edwards":7,"Gilbert":8,"IDOTRR":9,"MeadowV":10,"Mitchel":11,

                                                     "Names":12,"NoRidge":13,"NPkVill":14,"NridgHt":15,"NWAmes":16,"OldTown":17,

                                                     "SWISU":18,"Sawyer":19,"SawyerW":20,"Somerst":21,"StoneBr":22,"Timber":23,

                                                     "Veenker":25,"NAmes":26})
for col in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual",

            "FireplaceQu","GarageQual","GarageCond"]:

    train[col]= train[col].map({"Gd": 4 , "TA": 3, "Ex": 5, "Fa":2, "Po":1})
for col in ["Condition1","Condition2"]:

    train[col]= train[col].map({"Artery":0,"Feedr":1,"Norm":2,"RRNn":3,"RRAn":4,"PosN":5,"PosA":6,"RRNe":7,"RRAe":8})
for col in ["BsmtFinType1","BsmtFinType2"]:

    train[col]= train[col].map({"GLQ":0,"ALQ":1,"BLQ":2,"Rec":3,"LwQ":4,"Unf":5,"NA":6})
for col in train.columns:

    if train[col].dtype == 'object':

        print(col)

        
features=[["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual",

           "FireplaceQu","GarageQual","GarageCond","Condition1","Condition2","BsmtFinType1","BsmtFinType2",

           "RoofStyle","SaleCondition","Foundation","GarageType","MSZoning","Functional","RoofMatl","HouseStyle",

           "SaleType","Neighborhood","LandContour","Utilities","BsmtExposure","LotConfig","BldgType","Electrical","Heating","MasVnrType","LotShape","GarageFinish",

           "PavedDrive","LandSlope","Street","CentralAir","Exterior1st","Exterior2nd"]]

target=['SalePrice']

# Separating out the features

x = train[["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual",

           "FireplaceQu","GarageQual","GarageCond","Condition1","Condition2","BsmtFinType1","BsmtFinType2",

           "RoofStyle","SaleCondition","Foundation","GarageType","MSZoning","Functional","RoofMatl","HouseStyle",

           "SaleType","Neighborhood","LandContour","Utilities","BsmtExposure","LotConfig","BldgType","Electrical","Heating","MasVnrType","LotShape","GarageFinish",

           "PavedDrive","LandSlope","Street","CentralAir","Exterior1st","Exterior2nd"]]

# Separating out the target

y = train[['SalePrice']]

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

# create a base classifier used to evaluate a subset of attributes

model = LogisticRegression()

# create the RFE model and select 10 attributes

rfe = RFE(model, 10)

rfe = rfe.fit(x,y)

# summarize the selection of the attributes

print(rfe.support_)

print(rfe.ranking_)
#Dropping columns with reference to feature engineering

train=train.drop(columns=["ExterQual", "ExterCond","BsmtCond", "HeatingQC","FireplaceQu","GarageQual","GarageCond","Condition1","Condition2","BsmtFinType2","SaleCondition",

                         "GarageType","Functional","RoofMatl","HouseStyle","SaleType","Neighborhood","LandContour","Utilities","LotConfig","BldgType","Electrical","Heating",

                         "PavedDrive","LandSlope","Street","CentralAir","Exterior1st","Exterior2nd"])


train.shape
train.info()
#Independent Variables

x=train.iloc[:,0:23]

#Dependent Variable

y=train.iloc[:,23]

#Train and Test Split 

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.7, random_state=42)

print("Train x shape: ", train_x.shape)

print("Test x shape: ", test_x.shape)

print("Train y shape: ", train_y.shape)

print("Test y shape: ", test_y.shape)
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression

regr = RandomForestRegressor(n_estimators = 1000,random_state=42)

regr.fit(x, y)

y_pred = regr.predict(test_x)
evaluate(regr, test_x, test_y)
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#Dropping columns

test=test.drop(columns=['PoolQC','MiscFeature','Fence','Alley'])


Fill_na = ['BsmtQual', 'FireplaceQu', 'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'MasVnrType', 'MasVnrArea',

         'BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','Electrical']



for item in Fill_na:

    test[item] = test[item].fillna(test[item].mode()[0])
test['LotFrontage']=test['LotFrontage'].fillna(test['LotFrontage'].mean())
test['GarageYrBlt']=test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean())
#Checking for Missing Values

null_columns = train.columns[train.isnull().any()]

train[null_columns].isnull().sum().sort_values(ascending=False)
#Checking for Duplicate Rows

duplicate_rows=test[test.duplicated()]

print(duplicate_rows.shape)
#Dropping columns with reference to feature engineering

test=test.drop(columns=['MSSubClass','OverallQual','OverallCond','BsmtFinSF2','LowQualFinSF','BsmtFullBath','BsmtHalfBath',

                         'FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','1stFlrSF',

                         'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold'])
test=test.drop(columns=["ExterQual", "ExterCond","BsmtCond", "HeatingQC","FireplaceQu","GarageQual","GarageCond","Condition1","Condition2","BsmtFinType2","SaleCondition",

                         "GarageType","Functional","RoofMatl","HouseStyle","SaleType","Neighborhood","LandContour","Utilities","LotConfig","BldgType","Electrical","Heating",

                         "PavedDrive","LandSlope","Street","CentralAir","Exterior1st","Exterior2nd"])
test.info()
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])

test['MSZoning'] = test['MSZoning'].replace({"A":0,"C (all)":1,"FV":2,"I":3,"RH":4,"RL":5,"RP":6,"RM":7}).astype(int)

test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])

test['KitchenQual']=test['KitchenQual'].replace({"Gd": 4 , "TA": 3, "Ex": 5, "Fa":2, "Po":1}).astype(int)

test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mode()[0])

test['BsmtFinSF1']=test['BsmtFinSF1'].astype(int)

test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mode()[0])

test['BsmtUnfSF']=test['BsmtUnfSF'].astype(int)

test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mode()[0])

test['TotalBsmtSF']=test['TotalBsmtSF'].astype(int)

test['BsmtQual']=test['BsmtQual'].replace({"Gd": 4 , "TA": 3, "Ex": 5, "Fa":2, "Po":1}).astype(int)
test['LotShape'] = test['LotShape'].replace({"Reg":0,"IR1":1,"IR2":2,"IR3":3})

test['MasVnrType']=test['MasVnrType'].replace({"BrkCmn":0,"BrkFace":1,"CBlock":2,"None":3,"Stone":4})

test['RoofStyle']= test['RoofStyle'].replace({"Flat":0,"Gable":1,"Gambrel":2,"Hip":3,"Mansard":4,"Shed":5})

test['Foundation'] = test['Foundation'].replace({"BrkTil":0,"CBlock":1,"PConc":3,"Slab":4,"Stone":5,"Wood":6})

test['BsmtExposure']=test['BsmtExposure'].replace({"Gd":0,"Av":1,"Mn":2,"No":3,"NA":4})

test['GarageFinish']= test['GarageFinish'].replace({"Fin":0,"RFn":1,"Unf":2,"NA":3})

test['BsmtFinType1'] = test['BsmtFinType1'].replace({"GLQ":0,"ALQ":1,"BLQ":2,"Rec":3,"LwQ":4,"Unf":5,"NA":6})
test.info()
y_pred_test = regr.predict(test)
y_pred_test