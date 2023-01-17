# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import evogression



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
keep_cols = ['SalePrice', 'MSSubClass', 'LotFrontage']

keep_cols += ['LotArea', 'LotConfig', 'HouseStyle', 'OverallQual', 'OverallCond']

keep_cols += ['YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF']

keep_cols += ['GrLivArea', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr']

keep_cols += ['TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF']

keep_cols += ['MoSold', 'YrSold']



def lotconfig(x):

    if x == 'CulDSac':

        return 3

    elif x == 'Inside':

        return 2

    elif x == 'Corner':

        return 1

    else:

        return 2



def housestyle(x):

    if '2.5' in x:

        return 2.5

    elif '2' in x:

        return 2

    elif '1.5' in x:

        return 1.5

    else:

        return 1



converters = {'LotFrontage': lambda x: 0 if 'NA' else x,

              'LotConfig': lotconfig,

              'HouseStyle': housestyle,}

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train = train[keep_cols]

for col in keep_cols:

    if col in keep_cols:

        if col in converters:

            train[col] = train[col].map(converters[col])
model = evogression.Evolution('SalePrice', train, num_creatures=1000, num_cycles=10, use_multip=False, optimize=10)
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_id = test['Id'].tolist()

test = test[keep_cols[1:]]

for col in keep_cols:

    if col in converters:

        test[col] = test[col].map(converters[col])

        

test_predicted = model.predict(test, 'SalePrice')

test_predicted['SalePrice'] = test_predicted['SalePrice'].map(lambda x: x if x > 0 else 0)

test_predicted['Id'] = test_id

test_predicted[['Id', 'SalePrice']].to_csv('submission.csv', index=False)