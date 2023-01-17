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
from sklearn import model_selection, linear_model, metrics

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler
zeshan_csv=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

zeshan_labels=zeshan_csv['SalePrice']
zero_count=zeshan_csv.isnull().sum(axis = 0)



#remove columns without zero

zero_count=zero_count[zero_count > 0]

zero_count.sort_values()
numerical_columns=['BsmtFinSF2','TotalBsmtSF' , 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF','LotArea' , '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal','BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']

numbers_data=zeshan_csv[numerical_columns]

numbers_data=numbers_data.fillna(0)

fig, axs = plt.subplots (7,7)

fig.set_size_inches(30,20)



fig.suptitle('numeric features')

for i,feature in enumerate(numbers_data.columns):

    axs[i//5, i%5].scatter(numbers_data[feature], zeshan_csv['SalePrice'], color='red')

    axs[i//5, i%5].set_title(feature)
corr_labels=numbers_data.corrwith(zeshan_labels)

corr_labels.sort_values()
numerical_columns=['LotArea', 'OpenPorchSF', '2ndFlrSF', 'WoodDeckSF', 'BsmtFinSF1', 'Fireplaces', 'MasVnrArea', 'TotRmsAbvGrd', 'FullBath', '1stFlrSF', 'TotalBsmtSF', 'GarageArea', 'GarageCars', 'GrLivArea']

numbers_data=numbers_data[numerical_columns]
correlation_vectors=numbers_data.corr(method ='pearson')

correlation_vectors
for Value_C, content in correlation_vectors.items():

    for value_R, score in content.items():

        if score>0.5 and value_R>Value_C:

            print(value_R, Value_C, score)
numbers_data.drop(['1stFlrSF','2ndFlrSF','TotalBsmtSF', 'GarageArea', 'BsmtFinSF1','GrLivArea', 'FullBath', 'GarageCars','TotRmsAbvGrd'], axis = 1, inplace = True)
numbers_data.skew()

numbers_data.kurt()
rank_columns = ['OverallQual', 'OverallCond','ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond','BsmtFinType1', 'BsmtFinType2','Functional']

rank_data=zeshan_csv[rank_columns]
def linear_regression():

    Y_zeshan=zeshan_csv['SalePrice']

    Y_zeshan = np.log1p(Y_zeshan)

    X_zeshan=numbers_data(zeshan_csv)

    regressor=linear_model.SGDRegressor()

    

    # SGDRegressor

    regressor.fit(X_zeshan,Y_zeshan)

    

    #perform cross-validation

    scorer=metrics.make_scorer(metrics.mean_squared_error)

    linear_scoring = model_selection.cross_val_score(regressor, X_zeshan, Y_zeshan, scoring=scorer, cv = 5)

    print('mean: {}, std: {}'.format(linear_scoring.mean(), linear_scoring.std()))

    

    #predict results

    X_test=numbers_data(test_csv)

    ridge_predictions = regressor.predict(X_test)

    ridge_predictions_restored=np.expm1(ridge_predictions)

    

    return ridge_predictions_restored
test_csv=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
ridge_predictions_restored=linear_regression()