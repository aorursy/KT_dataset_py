# https://github.com/kaggle/docker-python

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR

from sklearn.utils import shuffle

from subprocess import check_output

import time
INPUT_PATH = '../input/'

ENCODING = 'utf8'

OUTPUT_PATH = '../working/submission.csv'



TRAIN_PATH = INPUT_PATH + '/train.csv'

TEST_PATH = INPUT_PATH + '/test.csv'



ONE_HOT = [

    'Alley',

    'BldgType',

    'Condition1',

    'Condition2',

    'Electrical',

    'Exterior1st',

    'Exterior2nd',

    'Foundation',

    'GarageFinish',

    'GarageType',

    'Heating',

    'HouseStyle',

    'LandContour',

    'LandSlope',

    'LotConfig',

    'LotShape',

    'MasVnrType',

    'MiscFeature',

    'MSZoning',

    'Neighborhood',

    'PavedDrive',

    'RoofMatl',

    'RoofStyle',

    'SaleCondition',

    'SaleType',

    'Street',

    'Utilities',

    'MSSubClass'

]



TARGET = 'SalePrice'
print(check_output(["ls", INPUT_PATH]).decode(ENCODING))
train = pd.read_csv(TRAIN_PATH)

train.head()
train.describe()
train.shape
size = (10, 8)

fig, ax = plt.subplots(figsize=size)

sns.heatmap(train.corr(), ax=ax)
def one_hot_encode(data):

    for item in ONE_HOT:

        one_hot = pd.get_dummies(data[item])

        one_hot = one_hot.add_prefix(item)

        data = data.drop(item, axis=1)

        data = data.join(one_hot)

    

    return data
train = one_hot_encode(train)

train.shape
def check_null(data):

    for item in data.columns:

        is_null = data[item].isnull().values.any()



        if is_null:

            print("%s: %s" % (item, is_null))
check_null(train)
def impute_null(data):

    data['BsmtCond'].fillna('NA', inplace=True)

    data['BsmtExposure'].fillna('NA', inplace=True)

    data['BsmtFinSF1'].fillna(data['BsmtFinSF1'].mean(), inplace=True)

    data['BsmtFinSF2'].fillna(data['BsmtFinSF2'].mean(), inplace=True)

    data['BsmtFinType1'].fillna('NA', inplace=True)

    data['BsmtFinType2'].fillna('NA', inplace=True)

    data['BsmtFullBath'].fillna(data['BsmtFullBath'].mean(), inplace=True)

    data['BsmtHalfBath'].fillna(data['BsmtHalfBath'].mean(), inplace=True)

    data['BsmtQual'].fillna('NA', inplace=True)

    data['BsmtUnfSF'].fillna(data['BsmtUnfSF'].mean(), inplace=True)

    data['Fence'].fillna('NA', inplace=True)

    data['FireplaceQu'].fillna('NA', inplace=True)

    data['Functional'].fillna('Typ', inplace=True)

    data['GarageArea'].fillna(data['GarageArea'].mean(), inplace=True)

    data['GarageCars'].fillna(data['GarageCars'].mean(), inplace=True)

    data['GarageCond'].fillna('NA', inplace=True)

    data['GarageQual'].fillna('NA', inplace=True)

    data['GarageYrBlt'].fillna(0, inplace=True)

    data['KitchenQual'].fillna('TA', inplace=True)

    data['LotFrontage'].fillna(data['LotFrontage'].mean(), inplace=True)

    data['MasVnrArea'].fillna(data['MasVnrArea'].mean(), inplace=True)

    data['PoolQC'].fillna('NA', inplace=True)

    data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].mean(), inplace=True)
impute_null(train)
check_null(train)
def rank_exterior_quality(data):

    rank = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

    return data.replace({'ExterQual': rank})
def rank_exterior_condition(data):

    rank = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

    return data.replace({'ExterCond': rank})
def rank_basement_quality(data):

    rank = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

    return data.replace({'BsmtQual': rank})
def rank_basement_condition(data):

    rank = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

    return data.replace({'BsmtCond': rank})
def rank_basement_exposure(data):

    rank = {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}

    return data.replace({'BsmtExposure': rank})
def rank_basement_finish_type_one(data):

    rank = {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 4, 'GLQ': 5}

    return data.replace({'BsmtFinType1': rank})
def rank_basement_finish_type_two(data):

    rank = {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 4, 'GLQ': 5}

    return data.replace({'BsmtFinType2': rank})
def rank_heating_quality_condition(data):

    rank = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

    return data.replace({'HeatingQC': rank})
def rank_kitchen_quality(data):

    rank = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

    return data.replace({'KitchenQual': rank})
def rank_functionality(data):

    rank = {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}

    return data.replace({'Functional': rank})
def rank_fireplace_quality(data):

    rank = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

    return data.replace({'FireplaceQu': rank})
def rank_garage_quality(data):

    rank = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

    return data.replace({'GarageQual': rank})
def rank_garage_condition(data):

    rank = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

    return data.replace({'GarageCond': rank})
def rank_pool_quality(data):

    rank = {'NA': 0, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

    return data.replace({'PoolQC': rank})
def rank_fence_quality(data):

    rank = {'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}

    return data.replace({'Fence': rank})
def ordinal_to_rank(data):

    data = rank_exterior_quality(data)

    data = rank_exterior_condition(data)

    data = rank_basement_quality(data)

    data = rank_basement_condition(data)

    data = rank_basement_exposure(data)

    data = rank_basement_finish_type_one(data)

    data = rank_basement_finish_type_two(data)

    data = rank_heating_quality_condition(data)

    data = rank_kitchen_quality(data)

    data = rank_functionality(data)

    data = rank_fireplace_quality(data)

    data = rank_garage_quality(data)

    data = rank_garage_condition(data)

    data = rank_pool_quality(data)

    return rank_fence_quality(data)
train = ordinal_to_rank(train)
def central_air_to_binary(data):

    central_air = {'N': 0, 'Y': 1}

    return data.replace({'CentralAir': central_air})
train = central_air_to_binary(train)
for item in train.columns:

    print('%s: %s' % (item, train[item].dtype))
train = shuffle(train)
train_x = train.drop(['Id', TARGET], axis=1)

train_y = train[TARGET]
test = pd.read_csv(TEST_PATH)

identifier = test['Id']

test = one_hot_encode(test)

impute_null(test)

test = ordinal_to_rank(test)

test = central_air_to_binary(test)

test_x = test.drop('Id', axis=1)
check_null(test)
train_x.shape[1] == test_x.shape[1]
for item in list(set(train_x.columns) - set(test_x.columns)):

    test_x[item] = 0
for item in list(set(test_x.columns) - set(train_x.columns)):

    train_x[item] = 0
train_x.shape[1] == test_x.shape[1]
train_x = np.array(train_x)

train_y = np.array(train_y)

test_x = np.array(test_x)
train_x = preprocessing.scale(train_x)

test_x = preprocessing.scale(test_x)
classifier = RandomForestRegressor()

cross_val_score(

    classifier,

    train_x,

    train_y

).mean()
classifier = SVR()

cross_val_score(

    classifier,

    train_x,

    train_y

).mean()
classifier = MLPRegressor()

cross_val_score(

    classifier,

    train_x,

    train_y

).mean()
parameters = [

    {'n_estimators': [10, 20, 40, 80, 160, 320, 640, 1280]}

]



classifier = GridSearchCV(RandomForestRegressor(), parameters)

classifier.fit(train_x, train_y)

classifier.best_params_
classifier.cv_results_
price = classifier.predict(test_x)
submission = pd.DataFrame(identifier)

price = pd.DataFrame(price)

submission = submission.join(price)

submission.columns = ['Id', 'SalePrice']

submission_path = '../working/submission_' + time.strftime("%Y%m%d-%H%M%S") + '.csv'

submission.to_csv(submission_path, index=False)