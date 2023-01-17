# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

train.head()
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

TEST_ID = test['Id']

test.head()
## 目的変数との相関関係をみる

(train.corr()**2)["SalePrice"].sort_values(ascending = False)[1:] # 2乗してマイナスもプラスにした版



train.corr()['SalePrice'].sort_values(ascending = False) # 普通に相関をした版



## corr()メソッドではデータ型がobject（文字列）の列は除外
# GrLivArea

train.loc[train.GrLivArea > 4000].loc[train.SalePrice <= 200000]
# TotalBsmtSF

train.loc[train.TotalBsmtSF > 6000]
# 1stFlrSF

train.loc[train['1stFlrSF'] > 4000]
# YearRemodAdd

train.loc[1990 < train.YearRemodAdd].loc[train.YearRemodAdd < 2000].loc[train.SalePrice > 600000]
# 上記の外れ値を削除　index指定でdrop()

train.drop(train.index[[523, 691, 1169, 1182, 1298]])
all_data = pd.concat((train, test)).reset_index(drop = True)
# 数値変数　→欠損を中央値で　(objectを除外)

for feature in all_data.select_dtypes(exclude=['object']).columns:

        all_data[feature].fillna(all_data[feature].median(), inplace = True)



# カテゴリ変数 →欠損を最頻値で (objectを抽出)

for feature in all_data.select_dtypes(include=['object']).columns: 

        all_data[feature].fillna(all_data[feature].value_counts().idxmax(), inplace = True)
from scipy.stats import skew



# 数値変数

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# 歪度計算

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))



skewed_feats = skewed_feats[skewed_feats > 0.75] # 絞る

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
f_drop_column = [

    'BsmtFinSF1',

    'LotFrontage',

    'WoodDeckSF',

    '2ndFlrSF',

    'OpenPorchSF',

    'HalfBath',

    'LotArea',

    'BsmtFullBath',

    'BsmtUnfSF',

    'BedroomAbvGr',

    'ScreenPorch',

    'PoolArea',

    'MoSold',

    '3SsnPorch',

    'BsmtFinSF2',

    'BsmtHalfBath',

    'MiscVal',

    'Id',

    'LowQualFinSF',

    'YrSold',

    'OverallCond',

    'MSSubClass',

    'EnclosedPorch',

    'KitchenAbvGr'

]
all_data.drop(f_drop_column, axis=1, inplace=True)
MSZoning_map = {'RL':1, 'RM':0, 'C (all)':0, 'FV':1, 'RH':0}

Street_map = {"Pave":1, "Grvl":1}

Alley_map = {"Grvl":1, "Pave":1}

LotShape_map = {"Reg":0, "IR1":1, "IR2":1, "IR3":1}

LandContour_map = {"Lv1":0, "Bnk":0, "Low":1, "HLS":1}

Utilities_map = {"AllPub":1, "NoSeWa":1}

LotConfig_map = {"Inside":0, "FR2":0, "Corner":0, "CulDSac":1, "FR3":1}

LandSlope_map = {"Gtl":1, "Mod":1, "Sev":1}

Neighborhood_map = {

    'CollgCr':1, 'Veenker':1, 'Crawfor':1, 'NoRidge':2, 'Mitchel':0, 'Somerst':1, 'NAmes':1, 'OldTown':0, 'BrkSide':0,

    'Sawyer':0, 'NridgHt':2, 'NWAmes':0, 'SawyerW':1, 'IDOTRR':0, 'MeadowV':0, 'Edwards':0, 'Timber':1, 'Gilbert':1,

    'StoneBr':2, 'ClearCr':1, 'NPkVill':0, 'Blmngtn':1, 'BrDale':0, 'SWISU':0, 'Blueste':0,

}

Condition1_map = {

    "Norm":0, "Feedr":0, "PosN":1, "Artery":0, "RRAe":0, "RRNn":1, "RRAn":0, "PosA":1, "RRNe":0

}

Condition2_map = {

    "Norm":0, "Artery":0, "RRNn":0, "Feedr":0, "PosN":1, "PosA":1, "RRAn":0, "RRAe":0,

}

BldgType_map = {

    "1Fam":1, "2fmCon":1, "Duplex":1, "TwnhsE":1, "Twnhs":1

}

HouseStyle_map = {

    '2Story':0, '1Story':0, '1.5Fin':0, '1.5Unf':0, 'SFoyer':0, 'SLvl':0, '2.5Unf':0, '2.5Fin':1

}

RoofStyle_map = {

    'Gable':1, 'Hip':1, 'Gambrel':1, 'Mansard':1, 'Flat':1, 'Shed':1

}

RoofMatl_map = {

    'CompShg':0, 'WdShngl':1, 'Metal':0, 'WdShake':0, 'Membran':0, 'Tar&Grv':0, 'Roll':0, 'ClyTile':0

}

Exterior1st_map = {

    'VinylSd':1, 'MetalSd':0, 'Wd Sdng':0, 'HdBoard':0, 'BrkFace':1, 'WdShing':0,

    'CemntBd':1, 'Plywood':0, 'AsbShng':0, 'Stucco':0, 'BrkComm':0, 'AsphShn':0,

    'Stone':2, 'ImStucc':2, 'CBlock':0

}

Exterior2nd_map = {

    'VinylSd':1, 'MetalSd':0, 'Wd Shng':0, 'HdBoard':0, 'Plywood':0, 'Wd Sdng':0,

    'CmentBd':1, 'BrkFace':0, 'Stucco':0, 'AsbShng':0, 'Brk Cmn':0, 'ImStucc':1,

    'AsphShn':0, 'Stone':0, 'Other':0, 'CBlock':0

}

MasVnrType_map = {

    'BrkFace':0, 'None':0, 'Stone':1, 'BrkCmn':0

}

ExterQual_map = {

    'Gd':0, 'TA':0, 'Ex':1, 'Fa':0

}

ExterCond_map = {

    'TA':0, 'Gd':0, 'Fa':0, 'Po':0, 'Ex':1

}

Foundation_map = {

    'PConc':1, 'CBlock':0, 'BrkTil':0, 'Wood':1, 'Slab':0, 'Stone':1

}

BsmtQual_map = {

    'Gd':0, 'TA':0, 'Ex':1, 'Fa':0

}

BsmtCond_map = {

    'TA':1, 'Gd':1, 'Fa':0, 'Po':0

}

BsmtExposure_map = {

    'No':0, 'Gd':1, 'Mn':0, 'Av':0

}

BsmtFinType1_map = {

    'GLQ':1, 'ALQ':0, 'Unf':0, 'Rec':0, 'BLQ':0, 'LwQ':0

}

BsmtFinType2_map = {

    'Unf':0, 'BLQ':0, 'ALQ':1, 'Rec':0, 'LwQ':0, 'GLQ':1

}

Heating_map = {

    'GasA':1, 'GasW':1, 'Grav':0, 'Wall':0, 'OthW':0, 'Floor':0

}

HeatingQC_map = {

    'Ex':1, 'Gd':0, 'TA':0, 'Fa':0, 'Po':0

}

CentralAir_map = {

    'Y':1, 'N':0

}

Electrical_map = {

    'SBrkr':1, 'FuseF':0, 'FuseA':0, 'FuseP':0, 'Mix':0

}

KitchenQual_map = {

    'Gd':0, 'TA':0, 'Ex':1, 'Fa':0

}

Functional_map = {

    'Typ':1, 'Min1':1, 'Maj1':1, 'Min2':1, 'Mod':1, 'Maj2':0, 'Sev':1

}

FireplaceQu_map = {

    'Gd':0, 'TA':0, 'Fa':0, 'Ex':1, 'Po':0

}

GarageType_map = {

    'Attchd':1, 'Detchd':0, 'BuiltIn':1, 'CarPort':0, 'Basment':0, '2Types':0

}

GarageFinish_map = {

    'RFn':0, 'Unf':0, 'Fin':1

}

GarageQual_map = {

    'TA':0, 'Fa':0, 'Gd':1, 'Ex':1, 'Po':0

}

GarageCond_map = {

    'TA':1, 'Fa':0, 'Gd':0, 'Po':0, 'Ex':0

}

PavedDrive_map = {

    'Y':1, 'N':1, 'P':1

}

PoolQC_map = {

    'Ex':1, 'Fa':0, 'Gd':0

}

Fence_map = {

    'MnPrv':1, 'GdWo':1, 'GdPrv':1, 'MnWw':1

}

MiscFeature_map = {

    'Shed':1, 'Gar2':1, 'Othr':0, 'TenC':1

}

SaleType_map = {

    'WD':0, 'New':1, 'COD':0, 'ConLD':0, 'ConLI':0, 'CWD':0, 'ConLw':0, 'Con':1, 'Oth':0

}

SaleCondition_map = {

    'Normal':0, 'Abnorml':0, 'Partial':1, 'AdjLand':0, 'Alloca':0, 'Family':0

}
all_data["MSZoning"] = all_data.MSZoning.replace(MSZoning_map)

all_data["Street"] = all_data.Street.replace(Street_map)



all_data["Alley"] = all_data.Alley.replace(Alley_map)

all_data["LotShape"] = all_data.LotShape.replace(LotShape_map)

all_data["LandContour"] = all_data.LandContour.replace(LandContour_map)

all_data["Utilities"] = all_data.Utilities.replace(Utilities_map)

all_data["LotConfig"] = all_data.LotConfig.replace(LotConfig_map)

all_data["LandSlope"] = all_data.LandSlope.replace(LandSlope_map)

all_data["Neighborhood"] = all_data.Neighborhood.replace(Neighborhood_map)

all_data["Condition1"] = all_data.Condition1.replace(Condition1_map)

all_data["Condition2"] = all_data.Condition2.replace(Condition2_map)

all_data["BldgType"] = all_data.BldgType.replace(BldgType_map)

all_data["HouseStyle"] = all_data.HouseStyle.replace(HouseStyle_map)



all_data["RoofStyle"] = all_data.RoofStyle.replace(RoofStyle_map)

all_data["RoofMatl"] = all_data.RoofMatl.replace(RoofMatl_map)

all_data["Exterior1st"] = all_data.Exterior1st.replace(Exterior1st_map)

all_data["Exterior2nd"] = all_data.Exterior2nd.replace(Exterior2nd_map)

all_data["MasVnrType"] = all_data.MasVnrType.replace(MasVnrType_map)

all_data["ExterQual"] = all_data.ExterQual.replace(ExterQual_map)



all_data["ExterCond"] = all_data.ExterCond.replace(ExterCond_map)

all_data["Foundation"] = all_data.Foundation.replace(Foundation_map)

all_data["BsmtQual"] = all_data.BsmtQual.replace(BsmtQual_map)

all_data["BsmtCond"] = all_data.BsmtCond.replace(BsmtCond_map)



all_data["BsmtExposure"] = all_data.BsmtExposure.replace(BsmtExposure_map)

all_data["BsmtFinType1"] = all_data.BsmtFinType1.replace(BsmtFinType1_map)

all_data["BsmtFinType2"] = all_data.BsmtFinType2.replace(BsmtFinType2_map)

all_data["Heating"] = all_data.Heating.replace(Heating_map)

all_data["HeatingQC"] = all_data.HeatingQC.replace(HeatingQC_map)

all_data["CentralAir"] = all_data.CentralAir.replace(CentralAir_map)

all_data["Electrical"] = all_data.Electrical.replace(Electrical_map)

all_data["KitchenQual"] = all_data.KitchenQual.replace(KitchenQual_map)

all_data["Functional"] = all_data.Functional.replace(Functional_map)

all_data["FireplaceQu"] = all_data.FireplaceQu.replace(FireplaceQu_map)

all_data["GarageType"] = all_data.GarageType.replace(GarageType_map)

all_data["GarageFinish"] = all_data.GarageFinish.replace(GarageFinish_map)

all_data["GarageQual"] = all_data.GarageQual.replace(GarageQual_map)

all_data["GarageCond"] = all_data.GarageCond.replace(GarageCond_map)

all_data["PavedDrive"] = all_data.PavedDrive.replace(PavedDrive_map)

all_data["PoolQC"] = all_data.PoolQC.replace(PoolQC_map)

all_data["Fence"] = all_data.Fence.replace(Fence_map)



all_data["MiscFeature"] = all_data.MiscFeature.replace(MiscFeature_map)

all_data["SaleType"] = all_data.SaleType.replace(SaleType_map)

all_data["SaleCondition"] = all_data.SaleCondition.replace(SaleCondition_map)
# drop_columns = [

#     'GarageYrBlt', 'TotRmsAbvGrd', 'GarageArea'

# ]



# all_data.drop(drop_columns, axis=1, inplace=True)
# log transform the target:

train['SalePrice'] = np.log1p(train['SalePrice'])

y = train['SalePrice']
all_data = pd.get_dummies(all_data)

all_data.shape
drop_columns = [

    'LandSlope',

    'Street',

    'FireplaceQu',

    'Alley',

    'LandContour_0',

    'Utilities',

    'MiscFeature',

    'PavedDrive',

    'SaleCondition',

    'Fence',

    'HouseStyle',

    'BldgType',

    'Condition2',

    'RoofStyle', # ここまでは0だったやつ

#     'Exterior2nd',

#     'Condition1',

#     'LandContour_Lvl',

#     'Foundation',

#     'MasVnrArea',

#     'GarageYrBlt',

#     'TotRmsAbvGrd',

#     'Electrical', # ここまではマイナスのやつ

    'LandContour_Lvl',

    'Foundation',

    'MasVnrArea',

    'GarageYrBlt',

    'TotRmsAbvGrd',

    'Electrical',

    'GarageArea',

    'YearBuilt',

    'GarageType',

    'YearRemodAdd',

    'LandContour_1', # ここまでは+-0.01以下

#     'Condition1',

#     'LandContour_Lvl',

#     'Foundation',

#     'MasVnrArea',

#     'GarageYrBlt',

#     'TotRmsAbvGrd',

#     'Electrical',

#     'GarageArea',

#     'YearBuilt',

#     'GarageType',

#     'YearRemodAdd',

#     'LandContour_1',

#     'MasVnrType',

#     'ExterCond',

#     'ExterQual',

#     'BsmtFinType2',

#     'LotShape',

#     'GarageCars' # ここまでは+-0.02以下

]



all_data.drop(drop_columns, axis=1, inplace=True)
all_data.drop('SalePrice', axis=1, inplace=True)
train = all_data[:len(train)]

test = all_data[len(train):] # これで予測値を測りたい

y = y
# カラムの数が一緒か、確認

print(train.shape)

print(test.shape)
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold



# Define Model

ENet = ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=0)
ENet.fit(train, y)
# submit用のtestを渡すことに注意

def predict_cv(model, train_x, train_y, test_x):

    preds = []

    preds_test = []

    va_indexes = []

    kf = KFold(n_splits=4, shuffle=True, random_state=1234) # train_test_split(train, y, test_size=0.3, random_state=3) 2個分割

    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する

    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)): # enumerate 配列要素とインデックスも同時に取得する

        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]

        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        model.fit(tr_x, tr_y)

        tr_pred = model.predict(tr_x)

        pred = model.predict(va_x)

        preds.append(pred)

        pred_test = model.predict(test_x)

        preds_test.append(pred_test)

        va_indexes.append(va_idx)

        print('  score Train : {:.6f}' .format(np.sqrt(mean_squared_error(tr_y, tr_pred))), 

              '  score Valid : {:.6f}' .format(np.sqrt(mean_squared_error(va_y, pred)))) 

    # バリデーションデータに対する予測値を連結し、その後元の順番に並べなおす

    va_indexes = np.concatenate(va_indexes)

    preds = np.concatenate(preds, axis=0)

    order = np.argsort(va_indexes)

    pred_train = pd.DataFrame(preds[order])

    # テストデータに対する予測値の平均をとる

    preds_test = pd.DataFrame(np.mean(preds_test, axis=0))

    print('Score : {:.6f}' .format(np.sqrt(mean_squared_error(train_y, pred_train))))

    return pred_train, preds_test, model
pred_train, preds_test, model = predict_cv(ENet, train, y, test)

# preds_test submit用のデータの予測値が入っている。
# クロスバリデーション内の関数で、すでにモデルを使ってtestのyを予測してくれている

preds_test
# 提出用にする 指数関数 (E)x の値 eには、ネイピア数 (自然対数の底) の近似値 

y_test = np.expm1(preds_test[0])
sub = pd.DataFrame()

sub['Id'] = TEST_ID

sub['SalePrice'] = y_test

sub.to_csv('submission.csv',index=False)