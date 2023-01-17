# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#raw_train = pd.read_csv("../input/train.csv")

#print(raw_train)

# Any results you write to the current directory are saved as output.
#Divide columns

CONTINUOUS_COLUMNS = ["LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","MasVnrArea","ExterQual","ExterCond",

	"BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","HeatingQC",

	"1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr",

	"KitchenQual","TotRmsAbvGrd","Functional","Fireplaces","FireplaceQu","GarageYrBlt","GarageFinish","GarageCars",

	"GarageArea","GarageQual","GarageCond","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","PoolQC",

	"MiscVal","MoSold","YrSold"]

CATEGORICAL_COLUMNS = ["MSSubClass","MSZoning","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood",

	"Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType",

	"Foundation","Heating","CentralAir","Electrical","GarageType","PavedDrive","Fence","MiscFeature","SaleType","SaleCondition"]

LABEL_COLUMN = "SalePrice"

COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS + [LABEL_COLUMN]

print("Continuous columns {0:2d}".format(len(CONTINUOUS_COLUMNS)))

print("Categorical columns {0:2d}".format(len(CATEGORICAL_COLUMNS)))

print("Total useful columns {0:2d}".format(len(COLUMNS)))
#Fillin NA and convert some Categorical columns into Continuous columns

def input_clean(raw_data):

    data = raw_data.drop('Id', 1).copy()

    data = data.replace({                            

                            'MSSubClass':{

                                            20:'class20',

                                            30:'class30',

                                            40:'class40',

                                            45:'class45',

                                            50:'class50',

                                            60:'class60',

                                            70:'class70',

                                            75:'class75',

                                            80:'class80',

                                            85:'class85',

                                            90:'class90',

                                            120:'class120',

                                            150:'class150',

                                            160:'class160',

                                            180:'class180',

                                            190:'class190'

                                            },

                            'MSZoning': {'C (all)': 'C'

                                            },

                            'ExterQual': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1

                                            },

                            'ExterCond': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1

                                            },

                            'BsmtQual':{ 

                                            'Ex':5,

                                            'Gd':4,

                                            'TA':3,

                                            'Fa':2,

                                            'Po':1,

                                            'NoBsmt': 0},

                            'BsmtCond': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'NoBsmt': 0},

                            'BsmtExposure': {'Gd':4,

                                            'Av':3,

                                            'Mn':2,

                                            'No':1,

                                            'NoBsmt':0},

                            'BsmtFinType1':{'GLQ':6,

                                            'ALQ':5,

                                            'BLQ':4,

                                            'Rec':3,

                                            'LwQ':2,

                                            'Unf':1,

                                            'NoBsmt':0},

                            'BsmtFinType2':{

                                            'GLQ':6,

                                            'ALQ':5,

                                            'BLQ':4,

                                            'Rec':3,

                                            'LwQ':2,

                                            'Unf':1,

                                            'NoBsmt':0},

                            'HeatingQC': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1

                                            },

                            'KitchenQual': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'NoKit':0},

                            'Functional': {'Typ': 8,

                                            'Min1': 7,

                                            'Min2': 6,

                                            'Mod': 5,

                                            'Maj1': 4,

                                            'Maj2': 3,

                                            'Sev': 2,

                                            'Sal': 1,

                                            'NoFunc':0},

                            'FireplaceQu': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'NoFireplace': 0 

                                            },

                            'GarageFinish': {

                                            'Fin':3,

                                            'RFn':2,

                                            'Unf':1,

                                            'NoGarage':0

                                             },

                            'GarageQual': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'NoGarage': 0},

                            'GarageCond': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'NoGarage': 0},

                            'PoolQC': {'Ex':4,

                                            'Gd':3,

                                            'TA':2,

                                            'Fa':1,

                                            'NoPool':0

                                       }

                            })

    #fill NaN

    #CONTINUOUS_COLUMNS

    data['LotFrontage']=data['LotFrontage'].fillna(0)

    data['MasVnrArea']=data['MasVnrArea'].fillna(0)

    data['GarageYrBlt']=data['GarageYrBlt'].fillna(1899)

    data['BsmtFinSF1']=data['BsmtFinSF1'].fillna(0)

    data['BsmtFinSF2']=data['BsmtFinSF2'].fillna(0)

    data['BsmtUnfSF']=data['BsmtUnfSF'].fillna(0)

    data['TotalBsmtSF']=data['TotalBsmtSF'].fillna(0)

    data['BsmtFullBath']=data['BsmtFullBath'].fillna(0)

    data['BsmtHalfBath']=data['BsmtHalfBath'].fillna(0)

    data['GarageCars']=data['GarageCars'].fillna(0)

    data['GarageArea']=data['GarageArea'].fillna(0)



    #CONTINUOUS_COLUMNS==>CATEGORICAL_COLUMNS

    data['BsmtQual']=data['BsmtQual'].fillna(0)

    data['BsmtCond']=data['BsmtCond'].fillna(0)

    data['BsmtExposure']=data['BsmtExposure'].fillna(0)

    data['BsmtFinType1']=data['BsmtFinType1'].fillna(0)

    data['BsmtFinType2']=data['BsmtFinType1'].fillna(0)

    data['FireplaceQu']=data['FireplaceQu'].fillna(0)

    data['GarageFinish']=data['GarageFinish'].fillna(0)

    data['GarageQual']=data['GarageQual'].fillna(0)

    data['GarageCond']=data['GarageCond'].fillna(0)

    data['PoolQC']=data['PoolQC'].fillna(0)

    data['KitchenQual']=data['KitchenQual'].fillna(0)

    data['Functional']=data['Functional'].fillna(0)

    

    #CATEGORICAL_COLUMNS

    data['MSZoning'] = data['MSZoning'].fillna('None')

    data['Utilities'] = data['Utilities'].fillna('None')

    data['Exterior1st'] = data['Exterior1st'].fillna('None')

    data['Exterior2nd'] = data['Exterior2nd'].fillna('None')

    data['SaleType'] = data['SaleType'].fillna('None')

    data['Alley']=data['Alley'].fillna('NoAlley')

    data['MasVnrType']=data['MasVnrType'].fillna('NoMasVnr')

    data['GarageType']=data['GarageType'].fillna('NoGarage')

    data['Electrical']=data['Electrical'].fillna('NoElec')

    data['Fence']=data['Fence'].fillna('NoFc')

    data['MiscFeature']=data['MiscFeature'].fillna('NoFtr')

    

    return data
folder = '../input/'

raw_train = pd.read_csv(folder+"train.csv")

train = input_clean(raw_train)

print(train.describe())
#Print the options of each categorical columns

categorical_options = {}

for l in CATEGORICAL_COLUMNS:

    categorical_options[l]=train.loc[:,l].value_counts().index.values.tolist()

print(categorical_options)