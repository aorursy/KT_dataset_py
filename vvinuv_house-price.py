import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
print(os.listdir("../input"))
#Reading the databases
house_train = pd.read_csv('../input/train.csv', keep_default_na=False)
house_test = pd.read_csv('../input/test.csv', keep_default_na=False)
#Giving numerical numbers to the string values. After I read the scikit-learn
#documentation it is not the best practice to give the numbers to string 
#https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html

mszoning = {'A': 0,'C (all)': 1, 'FV': 2, 'I': 3, 'RH': 4, 'RL': 5, 'RP': 6, 
            'RM': 7, 'NA': 8} 
street = {'Grvl': 0, 'Pave': 1}
alley = {'Grvl': 0, 'Pave': 1, 'NA': 2}
lotshape = {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}
landcontour = {'Lvl': 0, 'Bnk': 1, 'HLS': 2, 'Low': 3}
utilities = {'AllPub': 0, 'NoSewr': 1, 'NoSeWa': 2, 'ELO': 3, 'NA': 4}
lotconfig = {'Inside': 0, 'Corner': 1, 'CulDSac': 2, 'FR2': 3, 'FR3': 4} 
landslope = {'Gtl': 0, 'Mod': 1, 'Sev': 2}
neighborhood = {"Blmngtn": 0, "Blueste": 1, "BrDale": 2, "BrkSide": 3, "ClearCr": 4, 
                "CollgCr": 5, "Crawfor": 6, "Edwards": 7, "Gilbert": 8, "IDOTRR": 9, 
                "MeadowV": 10, "Mitchel": 11, "NAmes": 12, "NoRidge": 13, 
                "NPkVill": 14, "NridgHt": 15, "NWAmes": 16, "OldTown": 17, "SWISU": 18, 
                "Sawyer": 19, "SawyerW": 20, "Somerst": 21, "StoneBr": 22, 
                "Timber": 23, "Veenker": 24}
condition1 = {'Artery': 0, 'Feedr': 1, 'Norm': 2, 'RRNn': 3, 'RRAn': 4, 'PosN': 5, 
             'PosA': 6, 'RRNe': 7, 'RRAe': 8}
condition2 = {'Artery': 0, 'Feedr': 1, 'Norm': 2, 'RRNn': 3, 'RRAn': 4, 'PosN': 5, 
             'PosA': 6, 'RRNe': 7, 'RRAe': 8}
bldgtype = {"1Fam": 0, "2fmCon": 1, "Duplex": 2, "TwnhsE": 3, "TwnhsI": 4, "Twnhs": 5}
housestyle = {"1Story": 0, "1.5Fin": 1, "1.5Unf": 2, "2Story": 3, "2.5Fin": 4, 
              "2.5Unf": 5, "SFoyer": 6, "SLvl": 7}
roofstyle = {"Flat": 0, "Gable": 1, "Gambrel": 2, "Hip": 3, "Mansard": 4, 
             "Shed": 5}
roofmatl = {"ClyTile": 0, "CompShg": 1, "Membran": 2, "Metal": 3, "Roll": 4, "Tar&Grv": 5, 
            "WdShake": 6, "WdShngl": 7}
exterior1st = {"AsbShng": 0, "AsphShn": 1, "BrkComm": 2, "BrkFace": 3, "CBlock": 4, "CemntBd": 5, 
               "HdBoard": 6, "ImStucc": 7, "MetalSd": 8, "Other": 9, "Plywood": 10, "PreCast": 11, 
               "Stone": 12, "Stucco": 13, "VinylSd": 14, "Wd Sdng": 15, "WdShing": 16, "NA": 17}
exterior2nd = {"AsbShng": 0, "AsphShn": 1, "Brk Cmn": 2, "BrkFace": 3, "CBlock": 4, "CmentBd": 5, 
               "HdBoard": 6, "ImStucc": 7, "MetalSd": 8, "Other": 9, "Plywood": 10, "PreCast": 11, 
               "Stone": 12, "Stucco": 13, "VinylSd": 14, "Wd Sdng": 15, "Wd Shng": 16, "NA": 17}
masvnrtype = {"BrkCmn": 0, "BrkFace": 1, "NA": 2, "None": 3, "Stone": 4}
exterqual = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4}
extercond = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4}
foundation = {"BrkTil": 0, "CBlock": 1, "PConc": 2, "Slab": 3, "Stone": 4, "Wood": 5}
bsmtqual = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}
bsmtcond = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}
bsmtexposure = {"Gd": 0, "Av": 1, "Mn": 2, "No": 3, "NA": 4}
bsmtfintype1 = {"GLQ": 0, "ALQ": 1, "BLQ": 2, "Rec": 3, "LwQ": 4, "Unf": 5, "NA": 6}
bsmtfintype2 = {"GLQ": 0, "ALQ": 1, "BLQ": 2, "Rec": 3, "LwQ": 4, "Unf": 5, "NA": 6}
heating = {"Floor": 0, "GasA": 1, "GasW": 2, "Grav": 3, "OthW": 4, "Wall": 5}
heatingqc = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4}
centralair = {"N": 0, "Y": 1}
electrical = {"SBrkr": 0, "FuseA": 1, "FuseF": 2, "FuseP": 3, "Mix": 4, "NA":5}
kitchenqual = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}
functional = {"Typ": 0, "Min1": 1, "Min2": 2, "Mod": 3, "Maj1": 4, "Maj2": 5, 
              "Sev": 6, "Sal": 7, "NA": 8}
fireplacequ = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}
garagetype = {"2Types": 0, "Attchd": 1, "Basment": 2, "BuiltIn": 3, "CarPort": 4, 
              "Detchd": 5, "NA": 6}
garagefinish = {"Fin": 0, "RFn": 1, "Unf": 2, "NA": 3}
garagequal = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}
garagecond = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}
paveddrive = {"Y": 0, "P": 1, "N": 2}
poolqc = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "NA": 4}
fence = {"GdPrv": 0, "MnPrv": 1, "GdWo": 2, "MnWw": 3, "NA": 4}
miscfeature = {"Elev": 0, "Gar2": 1, "Othr": 2, "Shed": 3, "TenC": 4, "NA": 5}
saletype = {"WD": 0, "CWD": 1, "VWD": 2, "New": 3, "COD": 4, "Con": 5, 
            "ConLw": 6, "ConLI": 7, "ConLD": 8, "Oth": 9, "NA": 10}
salecondition = {'Normal': 0, 'Abnorml': 1, 'AdjLand': 2, 'Alloca': 3, 'Family':4, 'Partial':5}
#Fill the numerical value of train sample
house_train.MSZoning = [mszoning[item] for item in house_train.MSZoning]
house_train.Street = [street[item] for item in house_train.Street]
house_train.Alley = [alley[item] for item in house_train.Alley]
house_train.LotShape = [lotshape[item] for item in house_train.LotShape]
house_train.LandContour = [landcontour[item] for item in house_train.LandContour]
house_train.Utilities = [utilities[item] for item in house_train.Utilities]
house_train.LotConfig = [lotconfig[item] for item in house_train.LotConfig]
house_train.LandSlope = [landslope[item] for item in house_train.LandSlope]
house_train.Neighborhood = [neighborhood[item] for item in house_train.Neighborhood]
house_train.Condition1 = [condition1[item] for item in house_train.Condition1]
house_train.Condition2 = [condition2[item] for item in house_train.Condition2]
house_train.BldgType = [bldgtype[item] for item in house_train.BldgType]
house_train.HouseStyle = [housestyle[item] for item in house_train.HouseStyle]
house_train.RoofStyle = [roofstyle[item] for item in house_train.RoofStyle]
house_train.RoofMatl = [roofmatl[item] for item in house_train.RoofMatl]
house_train.Exterior1st = [exterior1st[item] for item in house_train.Exterior1st]
house_train.Exterior2nd = [exterior2nd[item] for item in house_train.Exterior2nd]
house_train.MasVnrType = [masvnrtype[item] for item in house_train.MasVnrType]
house_train.ExterQual = [exterqual[item] for item in house_train.ExterQual]
house_train.ExterCond = [extercond[item] for item in house_train.ExterCond]
house_train.Foundation = [foundation[item] for item in house_train.Foundation]
house_train.BsmtQual = [bsmtqual[item] for item in house_train.BsmtQual]
house_train.BsmtCond = [bsmtcond[item] for item in house_train.BsmtCond]
house_train.BsmtExposure = [bsmtexposure[item] for item in house_train.BsmtExposure]
house_train.BsmtFinType1 = [bsmtfintype1[item] for item in house_train.BsmtFinType1]
house_train.BsmtFinType2 = [bsmtfintype2[item] for item in house_train.BsmtFinType2]
house_train.Heating = [heating[item] for item in house_train.Heating]
house_train.HeatingQC = [heatingqc[item] for item in house_train.HeatingQC]
house_train.CentralAir = [centralair[item] for item in house_train.CentralAir]
house_train.Electrical = [electrical[item] for item in house_train.Electrical]
house_train.KitchenQual = [kitchenqual[item] for item in house_train.KitchenQual]
house_train.Functional = [functional[item] for item in house_train.Functional]
house_train.FireplaceQu = [fireplacequ[item] for item in house_train.FireplaceQu]
house_train.GarageType = [garagetype[item] for item in house_train.GarageType]
house_train.GarageFinish = [garagefinish[item] for item in house_train.GarageFinish]
house_train.GarageQual = [garagequal[item] for item in house_train.GarageQual]
house_train.GarageCond = [garagecond[item] for item in house_train.GarageCond]
house_train.PavedDrive = [paveddrive[item] for item in house_train.PavedDrive]
house_train.PoolQC = [poolqc[item] for item in house_train.PoolQC]
house_train.Fence = [fence[item] for item in house_train.Fence]
house_train.MiscFeature = [miscfeature[item] for item in house_train.MiscFeature]
house_train.SaleType = [saletype[item] for item in house_train.SaleType]
house_train.SaleCondition = [salecondition[item] for item in house_train.SaleCondition]
#Fill the numerical value of test sample

house_test.MSZoning = [mszoning[item] for item in house_test.MSZoning]
house_test.Street = [street[item] for item in house_test.Street]
house_test.Alley = [alley[item] for item in house_test.Alley]
house_test.LotShape = [lotshape[item] for item in house_test.LotShape]
house_test.LandContour = [landcontour[item] for item in house_test.LandContour]
house_test.Utilities = [utilities[item] for item in house_test.Utilities]
house_test.LotConfig = [lotconfig[item] for item in house_test.LotConfig]
house_test.LandSlope = [landslope[item] for item in house_test.LandSlope]
house_test.Neighborhood = [neighborhood[item] for item in house_test.Neighborhood]
house_test.Condition1 = [condition1[item] for item in house_test.Condition1]
house_test.Condition2 = [condition2[item] for item in house_test.Condition2]
house_test.BldgType = [bldgtype[item] for item in house_test.BldgType]
house_test.HouseStyle = [housestyle[item] for item in house_test.HouseStyle]
house_test.RoofStyle = [roofstyle[item] for item in house_test.RoofStyle]
house_test.RoofMatl = [roofmatl[item] for item in house_test.RoofMatl]
house_test.Exterior1st = [exterior1st[item] for item in house_test.Exterior1st]
house_test.Exterior2nd = [exterior2nd[item] for item in house_test.Exterior2nd]
house_test.MasVnrType = [masvnrtype[item] for item in house_test.MasVnrType]
house_test.ExterQual = [exterqual[item] for item in house_test.ExterQual]
house_test.ExterCond = [extercond[item] for item in house_test.ExterCond]
house_test.Foundation = [foundation[item] for item in house_test.Foundation]
house_test.BsmtQual = [bsmtqual[item] for item in house_test.BsmtQual]
house_test.BsmtCond = [bsmtcond[item] for item in house_test.BsmtCond]
house_test.BsmtExposure = [bsmtexposure[item] for item in house_test.BsmtExposure]
house_test.BsmtFinType1 = [bsmtfintype1[item] for item in house_test.BsmtFinType1]
house_test.BsmtFinType2 = [bsmtfintype2[item] for item in house_test.BsmtFinType2]
house_test.Heating = [heating[item] for item in house_test.Heating]
house_test.HeatingQC = [heatingqc[item] for item in house_test.HeatingQC]
house_test.CentralAir = [centralair[item] for item in house_test.CentralAir]
house_test.Electrical = [electrical[item] for item in house_test.Electrical]
house_test.KitchenQual = [kitchenqual[item] for item in house_test.KitchenQual]
house_test.Functional = [functional[item] for item in house_test.Functional]
house_test.FireplaceQu = [fireplacequ[item] for item in house_test.FireplaceQu]
house_test.GarageType = [garagetype[item] for item in house_test.GarageType]
house_test.GarageFinish = [garagefinish[item] for item in house_test.GarageFinish]
house_test.GarageQual = [garagequal[item] for item in house_test.GarageQual]
house_test.GarageCond = [garagecond[item] for item in house_test.GarageCond]
house_test.PavedDrive = [paveddrive[item] for item in house_test.PavedDrive]
house_test.PoolQC = [poolqc[item] for item in house_test.PoolQC]
house_test.Fence = [fence[item] for item in house_test.Fence]
house_test.MiscFeature = [miscfeature[item] for item in house_test.MiscFeature]
house_test.SaleType = [saletype[item] for item in house_test.SaleType]
house_test.SaleCondition = [salecondition[item] for item in house_test.SaleCondition]
for i in range(4):
    sns.distplot(np.log10(house_train.SalePrice)[house_train.FullBath==i], label='%d'%i)
plt.legend()
#There is a string NA in some columns. I replaced that with just zero, 
#which is very simple thing
house_train = house_train.replace('NA', 0).astype(float)
house_test = house_test.replace('NA', 0).astype(float)
#I am going to plot sale_price for unique values in the column.
#This is to find out how the sale_price is depend on individual parameters.
#If I find unique values greater than 28 I simply plot scatter plot between
#sale_price and that column. I need to find what are the import parameters 
#affecting the sale_price
n = 1
for column in house_train:
    uvalue = house_train[column].unique()
    if column == 'Id':
        continue
    plt.figure(n)
    if uvalue.shape[0] > 28:
        xcol = house_train[column]
        if xcol.max() > 500:
            plt.scatter(np.log10(house_train.SalePrice), np.log10(xcol))
        else:
            plt.scatter(np.log10(house_train.SalePrice), xcol)
        plt.xscale('log')
        plt.yscale('log')
    else:
        for i in uvalue:
            sns.distplot(np.log10(house_train.SalePrice)[house_train[column] == i], 
                         label='%d'%i)
    plt.ylabel(column)
    plt.legend()
    n += 1

#After looking at the above figures I think these are the parameters which
#mostly affecting the sale_price
columns = ["LotFrontage", "LotArea", "LotShape", "Neighborhood", "Condition1", 
           "HouseStyle", "OverallQual", "YearBuilt", "YearRemodAdd", "RoofMatl", 
           "MasVnrType", "ExterQual", "BsmtQual", "BsmtCond", "BsmtExposure", 
           "BsmtFinType1", "TotalBsmtSF", "Heating", "CentralAir", "Electrical", 
           "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "FullBath", 
           "KitchenAbvGr", "KitchenQual", "Fireplaces", "FireplaceQu", "GarageType", 
           "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageCond", 
           "3SsnPorch", "PoolArea", "PoolQC", "SaleType", "SaleCondition"]
imp_house_train = house_train[columns]
imp_house_test = house_test[columns]
imp_house_train.columns
sale_price = house_train.SalePrice
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.linear_model import ElasticNetCV, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
#I divided the columns using the important parameters
X_train, X_test, y_train, y_test = train_test_split(imp_house_train, 
                                                    sale_price, test_size=0.3)
#For fun I am using a linear regression, 
linear_reg = LinearRegression()
linear_reg_model = linear_reg.fit(X_train, y_train)
predictions = linear_reg_model.predict(X_test)
print('Score > ', linear_reg_model.score(X_test, y_test))
#Seriously, I am using cross validation for linear regression 
linear_scores = cross_val_score(linear_reg, imp_house_train, sale_price, cv=5)
linear_prediction = cross_val_predict(linear_reg, imp_house_train, sale_price, 
                                      cv=5)
#Plotting true and predicted values
plt.scatter(sale_price, linear_prediction)
plt.plot([0, 700000], [0, 700000], c='r')
plt.xlabel('Original')
plt.ylabel('Predicted')
#Now I try ridge regression
ridge_reg = RidgeCV(alphas=np.array([ 0.01, 0.1, 1, 10]), fit_intercept=True, 
                    normalize=False, scoring=None, 
                    cv=None, gcv_mode=None, 
                    store_cv_values=False)
ridge_reg.fit(X_train, y_train)
ridge_prediction = ridge_reg.predict(X_test)
plt.scatter(y_test, ridge_prediction)
plt.plot([0, 600000], [0, 400000], c='r')
plt.xlabel('Original')
plt.ylabel('Predicted')
#Now I try lasso regression
lasso_reg = LassoCV(eps=0.001, n_alphas=100, cv=10)
lasso_reg.fit(X_train, y_train)
lasso_prediction = lasso_reg.predict(X_test)
plt.scatter(y_test, lasso_prediction)
plt.plot([0, 600000], [0, 400000], c='r')
plt.xlabel('Original')
plt.ylabel('Predicted')
#Since the decision tree is not giving me nice output I use the AdaBoostRegressor
#for the regresion to improve the result. AdaBoostRegressor boost 
#the weak learners in trees
ada_deci_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10, 
                                                       min_samples_split=2),
                                 n_estimators=300, learning_rate=0.2, 
                                 loss='linear', random_state=992)
ada_deci_reg.fit(X_train, y_train)
ada_deci_prediction = ada_deci_reg.predict(X_test)
plt.scatter(y_test, ada_deci_prediction)
plt.plot([0, 700000], [0, 700000], c='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Original')
plt.ylabel('Predicted')
#The same for random forest 
ada_rand_reg = AdaBoostRegressor(RandomForestRegressor(n_estimators=10, 
                                                       criterion='mse', 
                                                       max_depth=10, 
                                                       min_samples_split=2, 
                                                       min_samples_leaf=1, 
                                                       bootstrap=True, 
                                                       #n_jobs=,
                                                       random_state=932),
                                 n_estimators=300, random_state=992)
ada_rand_reg.fit(X_train, y_train)
ada_rand_prediction = ada_rand_reg.predict(X_test)
plt.scatter(y_test, ada_rand_prediction)
plt.plot([0, 700000], [0, 700000], c='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Original')
plt.ylabel('Predicted')
#How well the adaboostregressor doing by ploting the decision and random trees
plt.scatter(ada_deci_prediction, ada_rand_prediction)
plt.plot([0, 700000], [0, 700000], c='r')
plt.xlabel('Decision tree')
plt.ylabel('Random tree')
#Just using random forest regressor 
rand_reg = RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=10, 
                                 min_samples_split=2, min_samples_leaf=1, 
                                 bootstrap=True, random_state=932) 
rand_reg.fit(X_train, y_train)
rand_prediction = rand_reg.predict(X_test)
plt.scatter(y_test, rand_prediction)
plt.plot([0, 700000], [0, 700000], c='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Original')
plt.ylabel('Predicted')
log_columns = ["LotFrontage", "LotArea", "LotShape", "Neighborhood", "Condition1", 
           "HouseStyle", "OverallQual", "YearBuilt", "YearRemodAdd", "RoofMatl", 
           "MasVnrType", "ExterQual", "BsmtQual", "BsmtCond", "BsmtExposure", 
           "BsmtFinType1", "TotalBsmtSF", "Heating", "CentralAir", "Electrical", 
           "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "FullBath", 
           "KitchenAbvGr", "KitchenQual", "Fireplaces", "FireplaceQu", "GarageType", 
           "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageCond", 
           "3SsnPorch", "PoolArea", "PoolQC", "SaleType", "SaleCondition", "SalePrice"]
log_imp_house_train = house_train[log_columns]
log_imp_house_test = house_test[columns]
log_columns = ['LotFrontage', 'LotArea', 'TotalBsmtSF', '1stFlrSF', 
                 '2ndFlrSF', 'GarageArea', 'SalePrice']
columns = ['LotFrontage', 'LotArea', 'TotalBsmtSF', '1stFlrSF', 
                 '2ndFlrSF', 'GarageArea']
log_imp_house_train = log_imp_house_train.drop(log_columns, axis=1)
log_imp_house_test = log_imp_house_test.drop(columns, axis=1)
log_imp_house_train['LotFrontage'] = np.log10(house_train.LotFrontage)
log_imp_house_train['LotArea'] = np.log10(house_train.LotArea)
log_imp_house_train['TotalBsmtSF'] = np.log10(house_train.TotalBsmtSF)
log_imp_house_train['1stFlrSF'] = np.log10(house_train['1stFlrSF'])
log_imp_house_train['2ndFlrSF'] = np.log10(house_train['2ndFlrSF'])
log_imp_house_train['GarageArea'] = np.log10(house_train.GarageArea)
log_imp_house_train['SalePrice'] = house_train.SalePrice
log_imp_house_test['LotFrontage'] = np.log10(house_test.LotFrontage)
log_imp_house_test['LotArea'] = np.log10(house_test.LotArea)
log_imp_house_test['TotalBsmtSF'] = np.log10(house_test.TotalBsmtSF)
log_imp_house_test['1stFlrSF'] = np.log10(house_test['1stFlrSF'])
log_imp_house_test['2ndFlrSF'] = np.log10(house_test['2ndFlrSF'])
log_imp_house_test['GarageArea'] = np.log10(house_test.GarageArea)
#Removing all the nan values for all columns which give ~450 rows
log_imp_house_train = log_imp_house_train.replace([np.inf, -np.inf], np.nan).dropna()
log_imp_house_test = log_imp_house_test.replace([np.inf, -np.inf], np.nan).dropna()
#You must understand log_sale_price is not log(sale_price). This is for 
#the conveiniance
log_sale_price = log_imp_house_train['SalePrice']
log_imp_house_train = log_imp_house_train.drop('SalePrice', axis=1)
X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(
    log_imp_house_train, log_sale_price, test_size=0.3)
#Using only adaboost regressor with decision tree
ada_deci_log_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10, 
                                                       min_samples_split=2),
                                 n_estimators=300, learning_rate=0.2, 
                                 loss='linear', random_state=992)
ada_deci_log_reg.fit(X_log_train, y_log_train)
ada_deci_log_prediction = ada_deci_log_reg.predict(X_log_test)
plt.scatter(y_log_test, ada_deci_log_prediction)
plt.plot([5e4, 700000], [5e4, 700000], c='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Original')
plt.ylabel('Predicted')
#Copy training dataframe to scaling dataframe 
scale_house_train = house_train.copy().astype(float)
#Imaportant columns
columns = ["LotFrontage", "LotArea", "LotShape", "Neighborhood", "Condition1",
           "HouseStyle", "OverallQual", "YearBuilt", "YearRemodAdd", "RoofMatl", 
           "MasVnrType", "ExterQual", "BsmtQual", "BsmtCond", "BsmtExposure",
           "BsmtFinType1", "TotalBsmtSF", "Heating", "CentralAir", "Electrical", 
           "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "FullBath",
           "KitchenAbvGr", "KitchenQual", "Fireplaces", "FireplaceQu", "GarageType", 
           "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageCond", 
           "3SsnPorch", "PoolArea", "PoolQC", "SaleType", "SaleCondition"]
scale_imp_house_train = scale_house_train[columns]
scale_sale_price = scale_house_train.SalePrice
#Scaling columns
scale_columns = ['LotFrontage', 'LotArea', 'TotalBsmtSF', '1stFlrSF', 
                 '2ndFlrSF', 'GarageArea']
temp_features = scale_imp_house_train[scale_columns]
#Scaler for columns
scaler = StandardScaler().fit(temp_features.values)
#Tranform from original to scaling value
features = scaler.transform(temp_features.values)
#Removing all the columns from the important dataframe
scale_imp_house_train = scale_imp_house_train.drop(scale_columns, axis=1)
scale_imp_house_train.columns
#Reinsert scaled columns
scale_imp_house_train['LotFrontage'] = features[:, 0]
scale_imp_house_train['LotArea'] = features[:, 1]
scale_imp_house_train['TotalBsmtSF'] = features[:, 2]
scale_imp_house_train['1stFlrSF'] = features[:, 3]
scale_imp_house_train['2ndFlrSF'] = features[:, 4]
scale_imp_house_train['GarageArea'] = features[:, 5]
#Train and test dataframes
X_scale_train, X_scale_test, y_scale_train, y_scale_test = train_test_split(
    scale_imp_house_train, scale_sale_price, test_size=0.3)
#Using adaboost with decision tree
ada_deci_scale_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=20, 
                                                       min_samples_split=6),
                                 n_estimators=300, learning_rate=0.8, 
                                 loss='linear', random_state=992)
ada_deci_scale_reg.fit(X_scale_train, y_scale_train)
ada_deci_scale_prediction = ada_deci_scale_reg.predict(X_scale_test)
plt.scatter(y_scale_test, ada_deci_scale_prediction)
plt.plot([5e4, 700000], [5e4, 700000], c='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Original')
plt.ylabel('Predicted')
house_all_train = house_train.copy()
house_all_train = house_all_train.drop('Id', axis=1)
house_all_train = house_all_train.drop('SalePrice', axis=1)
sale_all_price = house_train.SalePrice
house_all_test = house_test.drop('Id', axis=1)
X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(house_all_train, 
                                                    sale_all_price, test_size=0.3)
X_all_train.shape
#Decision tree with adaboost
ada_deci_all_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10, 
                                                       min_samples_split=2),
                                 n_estimators=300, learning_rate=0.2, 
                                 loss='linear', random_state=992)
ada_deci_all_reg.fit(X_all_train, y_all_train)
ada_deci_all_prediction = ada_deci_all_reg.predict(X_all_test)
plt.scatter(y_all_test, ada_deci_all_prediction)
plt.plot([5e4, 700000], [5e4, 700000], c='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Original')
plt.ylabel('Predicted')
plt.scatter(ada_deci_prediction, 
            (ada_deci_all_prediction - ada_deci_prediction) / ada_deci_prediction) 
#plt.plot([5e4, 700000], [5e4, 700000], c='r')
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Original')
plt.ylabel('Predicted')
#Random forest with adaboost
ada_rand_all_reg = AdaBoostRegressor(RandomForestRegressor(n_estimators=10, 
                                                       criterion='mse', 
                                                       max_depth=10, 
                                                       min_samples_split=2, 
                                                       min_samples_leaf=1, 
                                                       bootstrap=True, 
                                                       #n_jobs=,
                                                       random_state=932),
                                 n_estimators=300, random_state=992)
ada_rand_all_reg.fit(X_all_train, y_all_train)
ada_rand_all_prediction = ada_rand_all_reg.predict(X_all_test)
plt.scatter(y_all_test, ada_rand_all_prediction)
plt.plot([5e4, 700000], [5e4, 700000], c='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Original')
plt.ylabel('Predicted')
plt.scatter(ada_rand_prediction, 
            (ada_rand_all_prediction - ada_rand_prediction) / ada_rand_prediction) 
#plt.plot([5e4, 700000], [5e4, 700000], c='r')
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Original')
plt.ylabel('Predicted')
#Scale all the columns with unique value greater than 28
scale_all_house_train = house_train.copy().astype(float)
scale_all_sale_price = scale_all_house_train.SalePrice
scale_all_house_train = scale_all_house_train.drop('Id', axis=1)
scale_all_house_train = scale_all_house_train.drop('SalePrice', axis=1)

scale_all_columns = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 
                     'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
                     '2ndFlrSF', 'GrLivArea',  'GarageArea', 'WoodDeckSF', 
                     'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']

temp_features = scale_all_house_train[scale_all_columns]
scaler = StandardScaler().fit(temp_features.values)
features = scaler.transform(temp_features.values)
scale_all_house_train = scale_all_house_train.drop(scale_all_columns, axis=1)
scale_all_house_train['LotFrontage'] = features[:, 0]
scale_all_house_train['LotArea'] = features[:, 1]
scale_all_house_train['MasVnrArea'] = features[:, 2]
scale_all_house_train['BsmtFinSF1'] = features[:, 3]
scale_all_house_train['BsmtFinSF2'] = features[:, 4]
scale_all_house_train['BsmtUnfSF'] = features[:, 5]
scale_all_house_train['TotalBsmtSF'] = features[:, 6]
scale_all_house_train['1stFlrSF'] = features[:, 7]
scale_all_house_train['2ndFlrSF'] = features[:, 8]
scale_all_house_train['GrLivArea'] = features[:, 9]
scale_all_house_train['GarageArea'] = features[:, 10]
scale_all_house_train['WoodDeckSF'] = features[:, 11]
scale_all_house_train['OpenPorchSF'] = features[:, 12]
scale_all_house_train['EnclosedPorch'] = features[:, 13]
scale_all_house_train['ScreenPorch'] = features[:, 14]
X_scale_all_train, X_scale_all_test, y_scale_all_train, y_scale_all_test = train_test_split(scale_all_house_train, 
                                                       scale_all_sale_price, 
                                                       test_size=0.3)
#Using adaboost with decision tree using all scaled values
ada_deci_scale_all_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=20,
                                                       min_samples_split=6),
                                 n_estimators=300, learning_rate=0.8,
                                 loss='linear', random_state=992)
ada_deci_scale_all_reg.fit(X_scale_all_train, y_scale_all_train)
ada_deci_scale_all_prediction = ada_deci_scale_all_reg.predict(X_scale_all_test)
plt.scatter(y_scale_all_test, ada_deci_scale_all_prediction)
plt.plot([5e4, 700000], [5e4, 700000], c='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Original')
plt.ylabel('Predicted')
#Using adaboost with decision tree using all scaled values
ada_rand_scale_all_reg = AdaBoostRegressor(RandomForestRegressor(n_estimators=10, 
                                                       criterion='mse', 
                                                       max_depth=10, 
                                                       min_samples_split=2, 
                                                       min_samples_leaf=1, 
                                                       bootstrap=True, 
                                                       #n_jobs=,
                                                       random_state=932),
                                 n_estimators=300, learning_rate=0.8,
                                 loss='linear', random_state=992)
ada_rand_scale_all_reg.fit(X_scale_all_train, y_scale_all_train)
ada_rand_scale_all_prediction = ada_rand_scale_all_reg.predict(X_scale_all_test)
plt.scatter(y_scale_all_test, ada_rand_scale_all_prediction)
plt.plot([5e4, 700000], [5e4, 700000], c='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Original')
plt.ylabel('Predicted')
kaggle_ada_deci_prediction = ada_deci_reg.predict(imp_house_test)
kaggle_ada_rand_prediction = ada_rand_reg.predict(imp_house_test)
kaggle_ada_deci_all_prediction = ada_deci_all_reg.predict(house_all_test)
kaggle_ada_rand_all_prediction = ada_rand_all_reg.predict(house_all_test)
kaggle_ada_deci_scale_all_prediction = ada_deci_scale_all_reg.predict(house_all_test)
kaggle_ada_rand_scale_all_prediction = ada_rand_scale_all_reg.predict(house_all_test)
kaggle_df = pd.DataFrame(np.array([kaggle_ada_deci_prediction, 
                                   kaggle_ada_rand_prediction,
                                   kaggle_ada_deci_all_prediction,
                                   kaggle_ada_rand_all_prediction,
                                   kaggle_ada_deci_scale_all_prediction,
                                   kaggle_ada_rand_scale_all_prediction
                                  ]).T, 
                         columns=['kaggle_ada_deci_prediction', 
                                  'kaggle_ada_rand_prediction',
                                  'kaggle_ada_deci_all_prediction',
                                  'kaggle_ada_rand_all_prediction',
                                  'kaggle_ada_deci_scale_all_prediction',
                                  'kaggle_ada_rand_scale_all_prediction'])
sns.pairplot(kaggle_df)
plt.figure(1)
plt.scatter(kaggle_ada_deci_prediction, kaggle_ada_rand_prediction)
plt.plot([0, 700000], [0, 700000], c='r')
plt.xlabel('AdaDeci')
plt.ylabel('AdaRand')
plt.figure(2)
plt.scatter(kaggle_ada_deci_prediction, kaggle_ada_deci_all_prediction)
plt.plot([0, 700000], [0, 700000], c='r')
plt.xlabel('AdaDeci')
plt.ylabel('AdaDeciAll')
plt.figure(3)
plt.scatter(kaggle_ada_deci_prediction, kaggle_ada_rand_all_prediction)
plt.plot([0, 700000], [0, 700000], c='r')
plt.xlabel('AdaDeci')
plt.ylabel('AdaRandAll')
plt.figure(4)
plt.scatter(kaggle_ada_rand_prediction, kaggle_ada_deci_all_prediction)
plt.plot([0, 700000], [0, 700000], c='r')
plt.xlabel('AdaRand')
plt.ylabel('AdaDeciAll')
plt.figure(5)
plt.scatter(kaggle_ada_rand_prediction, kaggle_ada_rand_all_prediction)
plt.plot([0, 700000], [0, 700000], c='r')
plt.xlabel('AdaRand')
plt.ylabel('AdaRandAll')
plt.figure(6)
plt.scatter(kaggle_ada_deci_all_prediction, kaggle_ada_rand_all_prediction)
plt.plot([0, 700000], [0, 700000], c='r')
plt.xlabel('AdaDeciAll')
plt.ylabel('AdaRandAll')