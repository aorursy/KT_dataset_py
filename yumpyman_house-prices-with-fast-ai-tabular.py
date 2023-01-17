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
from fastai.tabular import *
path = Config().data_path()
train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
len(train_data), train_data.head()
test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test_data = test_data.fillna(0)
len(test_data), test_data.head()
#remove outliers

train_data = train_data.drop(
    train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)].index
)
all_data=pd.concat((train_data, test_data), axis=0)
# New features

all_data['TotalSF']=all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['No2ndFlr']=(all_data['2ndFlrSF']==0)
all_data['NoBsmt']=(all_data['TotalBsmtSF']==0)


all_data['TotalBath']     = all_data['BsmtFullBath'] + all_data['FullBath'] + (all_data['BsmtHalfBath']/2) + (all_data['HalfBath']/2)
all_data['YrBltAndRemod'] = all_data['YearBuilt']+all_data['YearRemodAdd']

# Basement 

Basement = ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtUnfSF',\
            'TotalBsmtSF']
Bsmt=all_data[Basement]

Bsmt=Bsmt.replace(to_replace='Po', value=1)
Bsmt=Bsmt.replace(to_replace='Fa', value=2)
Bsmt=Bsmt.replace(to_replace='TA', value=3)
Bsmt=Bsmt.replace(to_replace='Gd', value=4)
Bsmt=Bsmt.replace(to_replace='Ex', value=5)
Bsmt=Bsmt.replace(to_replace='None', value=0)

Bsmt=Bsmt.replace(to_replace='No', value=1)
Bsmt=Bsmt.replace(to_replace='Mn', value=2)
Bsmt=Bsmt.replace(to_replace='Av', value=3)
Bsmt=Bsmt.replace(to_replace='Gd', value=4)

Bsmt=Bsmt.replace(to_replace='Unf', value=1)
Bsmt=Bsmt.replace(to_replace='LwQ', value=2)
Bsmt=Bsmt.replace(to_replace='Rec', value=3)
Bsmt=Bsmt.replace(to_replace='BLQ', value=4)
Bsmt=Bsmt.replace(to_replace='ALQ', value=5)
Bsmt=Bsmt.replace(to_replace='GLQ', value=6)

Bsmt['BsmtScore']= Bsmt['BsmtQual']  * Bsmt['BsmtCond'] * Bsmt['TotalBsmtSF']
all_data['BsmtScore']=Bsmt['BsmtScore']

Bsmt['BsmtFin'] = (Bsmt['BsmtFinSF1'] * Bsmt['BsmtFinType1']) + (Bsmt['BsmtFinSF2'] * Bsmt['BsmtFinType2'])
all_data['BsmtFinScore']=Bsmt['BsmtFin']
all_data['BsmtDNF']=(all_data['BsmtFinScore']==0)

# Lot

lot=['LotFrontage', 'LotArea','LotConfig','LotShape']
Lot=all_data[lot]

Lot['LotScore'] = np.log((Lot['LotFrontage'] * Lot['LotArea'])+1)
all_data['LotScore']=Lot['LotScore']

# Garage 

garage=['GarageArea','GarageCars','GarageCond','GarageFinish','GarageQual','GarageType','GarageYrBlt']
Garage=all_data[garage]
all_data['NoGarage']=(all_data['GarageArea']==0)

Garage=Garage.replace(to_replace='Po', value=1)
Garage=Garage.replace(to_replace='Fa', value=2)
Garage=Garage.replace(to_replace='TA', value=3)
Garage=Garage.replace(to_replace='Gd', value=4)
Garage=Garage.replace(to_replace='Ex', value=5)
Garage=Garage.replace(to_replace='None', value=0)

Garage=Garage.replace(to_replace='Unf', value=1)
Garage=Garage.replace(to_replace='RFn', value=2)
Garage=Garage.replace(to_replace='Fin', value=3)

Garage=Garage.replace(to_replace='CarPort', value=1)
Garage=Garage.replace(to_replace='Basment', value=4)
Garage=Garage.replace(to_replace='Detchd', value=2)
Garage=Garage.replace(to_replace='2Types', value=3)
Garage=Garage.replace(to_replace='Basement', value=5)
Garage=Garage.replace(to_replace='Attchd', value=6)
Garage=Garage.replace(to_replace='BuiltIn', value=7)

Garage['GarageScore']=(Garage['GarageArea']) * (Garage['GarageCars']) * (Garage['GarageFinish'])*(Garage['GarageQual']) * \
(Garage['GarageType'])
all_data['GarageScore']=Garage['GarageScore']

# other 

all_data['NoLowQual']=(all_data['LowQualFinSF']==0)
all_data['NoOpenPorch']=(all_data['OpenPorchSF']==0)
all_data['NoWoodDeck']=(all_data['WoodDeckSF']==0)


# Drop

all_data=all_data.drop(columns=['PoolArea','PoolQC'])
all_data=all_data.drop(columns=['MiscVal','MiscFeature'])
all_data=all_data.drop(columns=['Street','Utilities','Condition2','RoofMatl','Heating'])

dep_var = 'SalePrice'

target = train_df[dep_var].copy()
# unskew values

numeric=all_data.select_dtypes(np.number)

def log_transform(col_list):
    transformed_col=[]
    while len(col_list)!=0:
        col=col_list.pop(0)
        if all_data[col].skew() > 0.5:
            print(f"Unskewing {col}")
            all_data[col]=np.log(all_data[col]+1)
            transformed_col.append(col)
        else:
            pass
    print(f"{len(transformed_col)} features had been transformed")
    print(all_data.shape)
    
log_transform(list(numeric))
all_data[dep_var] = target
# all_data[dep_var].head()
cat_vars = ['MSSubClass', 'MSZoning',
       'Alley', 'LotShape', 'LandContour', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
       'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical', 
       'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageQual',
       'GarageCond', 'PavedDrive',
       'Fence', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition',
       'NoLowQual', 'NoOpenPorch', 'NoWoodDeck', 'BsmtDNF', 'No2ndFlr', 'NoBsmt', 'YrBltAndRemod']

cont_vars =  [ 'LotFrontage', 'LotArea','MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 
       'WoodDeckSF', 'OpenPorchSF','EnclosedPorch', '3SsnPorch', 'ScreenPorch',
       'GarageScore', 'BsmtFinScore', 'LotScore', 'TotalSF', 'TotalBath']
# cat_vars = ['MSSubClass', 'MSZoning','Street',
#        'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
#        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
#        'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
#        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
#        'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
#        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
#        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
#        'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
#        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
#        'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
#        'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageQual',
#        'GarageCond', 'PavedDrive', 'PoolQC',
#        'Fence', 'MiscFeature', 'MoSold', 'YrSold', 'SaleType',
#        'SaleCondition']

# cont_vars =  [ 'LotFrontage', 'LotArea','MasVnrArea', 'BsmtFinSF1',
#        'BsmtFinSF2', 'BsmtUnfSF', 'To talBsmtSF',
#        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 
#        'WoodDeckSF', 'OpenPorchSF','EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']


train_df=all_data[:len(train_data)]
test_df=all_data[len(train_data):]
dep_var = 'SalePrice'
df = train_df[cat_vars + cont_vars + [dep_var]].copy()



procs=[FillMissing, Categorify, Normalize]
valid_idx = range(len(df)-290, len(df))

data = (TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs,)
                .split_by_idx(valid_idx)
                .label_from_df(cols=dep_var, label_cls=FloatList, log=True)
                .add_test(TabularList.from_df(test_df, path=path, cat_names=cat_vars, cont_names=cont_vars))
                .databunch())
# max_y = (np.max(train_df['SalePrice'])*1.05)
# y_range = torch.tensor([0, max_y], device=defaults.device)

max_log_y = np.log(np.max(train_df['SalePrice'])*1.1)
y_range = torch.tensor([0, max_log_y], device=defaults.device)

max_log_y, y_range
learn = tabular_learner(
    data, 
    layers=[1000,500],
    ps=[0.001,0.01],
    emb_drop=0.04,             
    y_range=y_range, 
    metrics=[exp_rmspe]
)

learn.model
len(data.train_ds.cont_names)
learn.lr_find()
learn.recorder.plot()
# learn.fit_one_cycle(100, 1e-3)

from fastai.callbacks import *

# learn.fit_one_cycle(250, 1e-3)

learn.fit_one_cycle(
    500, 
    1e-3, 
    callbacks=[
      ShowGraph(learn),
      SaveModelCallback(learn,monitor='exp_rmspe',mode='min')
    ]
)
#learn.recorder.plot_losses()
learn.recorder.plot_metrics(skip_start=100)
learn.load('bestmodel')
# diff_total = 0

# for x in range(len(train_df)):
#   target = train_df.iloc[x].SalePrice
#   pred = np.exp(learn.predict(train_df.iloc[x])[1][0])
#   diff =  abs(target - int(pred.data))
#   diff_total = diff_total + diff

# len(train_df), diff_total , diff_total / len(train_df)
# RMSE score

kaggle_df = train_df.drop(columns=['SalePrice'])

k_predictions = [0] * len(train_df)
k_targets     = [0] * len(train_df)


for x in range(len(kaggle_df)):
  k_predictions[x] = learn.predict(kaggle_df.iloc[x])[1][0].item()
  k_targets[x]     = np.log(train_df.SalePrice.iloc[x])
    
root_mean_squared_error(tensor(k_predictions), tensor(k_targets))    
reference_data = pd.read_csv("/kaggle/input/house-price-reference/reference.csv")
reference_data = reference_data.fillna(0)
len(reference_data), reference_data.head()
# kaggle score (against reference data)
    
root_mean_squared_error(tensor(k_predictions), tensor(np.log(reference_data.SalePrice))) 
# i = 3
# k_predictions[i], k_targets[i], k_predictions[i]/k_targets[i]
predictions = [0] * len(test_df)

for x in range(len(test_df)):
  predictions[x] = np.exp(learn.predict(test_df.iloc[x])[1][0].item())

len(predictions), predictions
output = pd.DataFrame({'Id': test_df.Id, 'SalePrice': predictions})

output.to_csv('house_price_cuijamm1_submission.csv', index=False)
