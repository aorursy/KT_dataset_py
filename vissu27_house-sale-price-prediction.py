# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline
data_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

data_test =  pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")



print (data_train.shape)

print (data_test.shape)
data_train.head()
target = data_train[['SalePrice']]

data_train = data_train.iloc[:,0:40]



data_test = data_test.iloc[:,0:40]



data_train.head()
print(data_train.columns)
def preprocess_input_data(data_train):

    data_train['Exterior2nd'].fillna(18, inplace=True)

    data_train.Exterior2nd = [{'AsbShng':1,'AsphShn':2,'Brk Cmn':3,'BrkFace':4,'CBlock':5,'CmentBd':6,'HdBoard':7,'ImStucc':8,'MetalSd':9,'Other':10,'Plywood':11,'PreCast':12,'Stone':13,'Stucco':14,'VinylSd':15,'Wd Sdng':16,'Wd Shng':17,18:18}[item] for item in data_train.Exterior2nd]

    

    data_train['MSZoning'].fillna(1, inplace=True)

    data_train.MSZoning = [{"A":1,"C (all)":2,"FV":3,"I":4,"RH":5,"RL":6,"RP":7,"RM":8,1:1}[item] for item in data_train.MSZoning]



    data_train['BsmtFinType2'].fillna(1, inplace=True)

    data_train.BsmtFinType2 = [{'Unf': 2, 'LwQ': 3, 'Rec': 4,'BLQ': 5, 'ALQ': 6, 'GLQ':3, 1:1 }[item] for item in data_train.BsmtFinType2] 



    data_train['BsmtFinType1'].fillna(1, inplace=True)

    data_train.BsmtFinType1 = [{'Unf': 2, 'LwQ': 3, 'Rec': 4,'BLQ': 5, 'ALQ': 6, 'GLQ':3, 1:1 }[item] for item in data_train.BsmtFinType1] 



    data_train.Heating = [{'Floor':1,'GasA':2,'GasW':3,'Grav':4,'OthW':5,'Wall':6}[item] for item in data_train.Heating] 



    data_train['BsmtExposure'].fillna(1, inplace=True)

    data_train.BsmtExposure = [{'Gd':4,'Av':3,'Mn':3,'No':2,1:1}[item] for item in data_train.BsmtExposure] 



    data_train['BsmtCond'].fillna(1, inplace=True)       

    data_train.BsmtCond = [{'Ex':6,'Gd':5,'TA':4,'Fa':3,'Po':2,1:1}[item] for item in data_train.BsmtCond] 



    data_train['BsmtQual'].fillna(1, inplace=True)       

    data_train.BsmtQual = [{'Ex':6,'Gd':5,'TA':4,'Fa':3,'Po':2,1:1}[item] for item in data_train.BsmtQual] 



    data_train.Foundation = [{'BrkTil':6,'CBlock':5,'PConc':4,'Slab':3,'Stone':2,'Wood':1}[item] for item in data_train.Foundation] 



    data_train.ExterCond = [{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}[item] for item in data_train.ExterCond] 

    data_train.ExterQual = [{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}[item] for item in data_train.ExterQual] 



    data_train['MasVnrType'].fillna(1, inplace=True)

    data_train.MasVnrType = [{'BrkCmn':5,'BrkFace':4,'CBlock':3,'Stone':2,'None':6,1:1}[item] for item in data_train.MasVnrType]



    data_train['Exterior1st'].fillna(18, inplace=True)

    data_train.Exterior1st = [{'AsbShng':1,'AsphShn':2,'BrkComm':3,'BrkFace':4,'CBlock':5,'CemntBd':6,'HdBoard':7,'ImStucc':8,'MetalSd':9,'Other':10,'Plywood':11,'PreCast':12,'Stone':13,'Stucco':14,'VinylSd':15,'Wd Sdng':16,'WdShing':17,18:18}[item] for item in data_train.Exterior1st]



    data_train.RoofMatl = [{'ClyTile':1,'CompShg':2,'Membran':3,'Metal':4,'Roll':5,'Tar&Grv':6,'WdShake':7,'WdShngl':8}[item] for item in data_train.RoofMatl]



    data_train.RoofStyle =[{'Flat':1,'Gable':2,'Gambrel':3,'Hip':4,'Mansard':5,'Shed':6}[item] for item in data_train.RoofStyle]



    data_train.HouseStyle = [{'1Story':1,'1.5Fin':2,'1.5Unf':3,'2Story':4,'2.5Fin':5,'2.5Unf':6,'SFoyer':7,'SLvl':8}[item] for item in data_train.HouseStyle]       



    data_train.BldgType = [{'1Fam':1,'2fmCon':2,'Duplex':3,'TwnhsE':4,'Twnhs':5}[item] for item in data_train.BldgType]



    data_train.Street = [{"Grvl":1,"Pave":2}[item] for item in data_train.Street]



    data_train['Alley'].fillna(1, inplace=True)

    data_train.Alley = [{"Grvl":2,"Pave":3,1:1}[item] for item in data_train.Alley]



    data_train.LotShape = [{"Reg":1,"IR1":2,"IR2":3,"IR3":4}[item] for item in data_train.LotShape]



    data_train.LandContour = [{"Lvl":1,"Bnk":2,"HLS":3,"Low":4}[item] for item in data_train.LandContour]



    data_train['Utilities'].fillna(5, inplace=True)

    data_train.Utilities = [{"AllPub":1,"NoSewr":2,"NoSeWa":3,"ELO":4,5:5}[item] for item in data_train.Utilities]



    data_train.LandSlope = [{"Gtl":1,"Mod":2,"Sev":3}[item] for item in data_train.LandSlope]



    data_train.Neighborhood = [{"Blmngtn":1,"Blueste":2,"BrDale":3,"BrkSide":4,"ClearCr":5,"CollgCr":6,"Crawfor":7,"Edwards":8,"Gilbert":9,"IDOTRR":10,"MeadowV":11,"Mitchel":12,"NAmes":13,"NoRidge":14,"NPkVill":15,"NridgHt":16,"NWAmes":17,"OldTown":18,"WISU":19,"Sawyer":20,"SawyerW":21,"Somerst":22,"StoneBr":23,"Timber":24,"Veenker":25,"SWISU":26}[item] for item in data_train.Neighborhood]



    data_train.Condition1 = [{'Artery':1,'Feedr':2,'Norm':3,'RRNn':4,'RRAn':5,'PosN':6,'PosA':7,'RRNe':8,'RRAe':9}[item] for item in data_train.Condition1]



    data_train.Condition2 = [{'Artery':1,'Feedr':2,'Norm':3,'RRNn':4,'RRAn':5,'PosN':6,'PosA':7,'RRNe':8,'RRAe':9}[item] for item in data_train.Condition2]



    data_train.LotConfig = [{"Inside":1,"Corner":2,"CulDSac":3,"FR2":4,"FR3":5}[item] for item in data_train.LotConfig]

    

    return data_train
data_train = preprocess_input_data(data_train)
data_test = preprocess_input_data(data_test)
data_train = data_train.drop(['Id'], axis = 1)



test_feature_set = data_test.drop(['Id'], axis = 1)
data_train['LotFrontage'].fillna(data_train['LotFrontage'].mean(), inplace=True)

data_train['MasVnrArea'].fillna(data_train['MasVnrArea'].mean(), inplace=True)
test_feature_set['LotFrontage'].fillna(data_train['LotFrontage'].mean(), inplace=True)

test_feature_set['MasVnrArea'].fillna(data_train['MasVnrArea'].mean(), inplace=True)

test_feature_set['BsmtFinSF1'].fillna(data_train['BsmtFinSF1'].mean(), inplace=True)

test_feature_set['BsmtFinSF2'].fillna(data_train['BsmtFinSF2'].mean(), inplace=True)

test_feature_set['BsmtUnfSF'].fillna(data_train['BsmtUnfSF'].mean(), inplace=True)

test_feature_set['TotalBsmtSF'].fillna(data_train['TotalBsmtSF'].mean(), inplace=True)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_train, target, test_size=0.2, random_state=42)
from sklearn import linear_model

#reg = linear_model.Ridge(alpha=.5)

reg = linear_model.LinearRegression()

#reg = svm.SVC()

reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)

y_test_pred = reg.predict(test_feature_set)
from sklearn.metrics import mean_squared_error, r2_score

from matplotlib import style 

# The coefficients

#print('Coefficients: \n', reg.coef_)

# The mean squared error

print('Mean squared error: %.2f'% mean_squared_error(y_test, y_pred))



# The coefficient of determination: 1 is perfect prediction

print('Coefficient of determination: %.2f'% r2_score(y_test, y_pred))



x_test_axis = [item for item in range((int)(y_pred.min()),(int)(y_pred.max()),(int)((y_pred.max()-y_pred.min())/len(y_pred)))]

del x_test_axis[-1] 

#x_test_axis = np.arange(y_pred.min(), y_pred.max(), ).tolist()

#plt.scatter(y_test, y_pred,  color='red')

#plt.scatter(x_test,y_test,  color='black')

style.use('ggplot')

plt.plot(x_test_axis,y_pred, color='blue', linewidth=3)

plt.plot(x_test_axis,y_test, color='red', linewidth=1)



plt.xticks(())

plt.yticks(())



plt.show()



plt.figure(figsize=(20,8))

plt.subplots_adjust(hspace=.5)

plt.subplot(2,2,1)

plt.title('Predicted Results')

plt.plot(x_test_axis,y_pred)

plt.subplot(2,2,2)

plt.title('Original Results')

plt.plot(x_test_axis,y_test)
diff = y_test - y_pred

plt.plot(x_test_axis,diff, color='red', linewidth=1)
pred_list = reg.predict(test_feature_set)

y_pred_dataframe = pd.DataFrame(pred_list,columns=['SalePrice'])

y_pred_dataframe
predictions = pd.concat([data_test['Id'],y_pred_dataframe], axis=1)

predictions.head()