# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
houseprice=pd.read_csv('../input/train.csv')

houseprice.head()
# To check how many columns have missing values - this can be repeated to see the progress made

def show_missing():

    missing = houseprice.columns[houseprice.isnull().any()].tolist()

    return missing
houseprice[show_missing()].isnull().sum()
# Looking at categorical values

def cat_exploration(column):

    return houseprice[column].value_counts()
# Imputing the missing values

def cat_imputation(column, value):

    houseprice.loc[houseprice[column].isnull(),column] = value
# check correlation with LotArea

houseprice['LotFrontage'].corr(houseprice['LotArea'])
# improvement - and good enough for now

houseprice['SqrtLotArea']=np.sqrt(houseprice['LotArea'])

houseprice['LotFrontage'].corr(houseprice['SqrtLotArea'])
import seaborn as sns

%pylab inline

sns.pairplot(houseprice[['LotFrontage','SqrtLotArea']].dropna())
cond = houseprice['LotFrontage'].isnull()

houseprice.LotFrontage[cond]=houseprice.SqrtLotArea[cond]
# This column is not needed anymore

del houseprice['SqrtLotArea']
cat_exploration('Alley')
# I assume empty fields here mean no alley access

cat_imputation('Alley','None')
houseprice[['MasVnrType','MasVnrArea']][houseprice['MasVnrType'].isnull()==True]
cat_exploration('MasVnrType')
cat_imputation('MasVnrType', 'None')

cat_imputation('MasVnrArea', 0.0)
basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']

houseprice[basement_cols][houseprice['BsmtQual'].isnull()==True]
for cols in basement_cols:

    if 'FinSF'not in cols:

        cat_imputation(cols,'None')
cat_exploration('Electrical')
# Impute most frequent value

cat_imputation('Electrical','SBrkr')
cat_exploration('FireplaceQu')
houseprice['Fireplaces'][houseprice['FireplaceQu'].isnull()==True].describe()
cat_imputation('FireplaceQu','None')
pd.crosstab(houseprice.Fireplaces, houseprice.FireplaceQu)
garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']

houseprice[garage_cols][houseprice['GarageType'].isnull()==True]
#Garage Imputation

for cols in garage_cols:

    if houseprice[cols].dtype==np.object:

        cat_imputation(cols,'None')

    else:

        cat_imputation(cols, 0)
cat_exploration('PoolQC')
houseprice['PoolArea'][houseprice['PoolQC'].isnull()==True].describe()
cat_imputation('PoolQC', 'None')
cat_imputation('Fence', 'None')
cat_imputation('MiscFeature', 'None')
houseprice[show_missing()].isnull().sum()
houseprice
from xgboost.sklearn import XGBRegressor
clf = XGBRegressor()
from sklearn.model_selection import cross_val_score
x_train = pd.get_dummies(houseprice.drop(['Id', 'SalePrice'], axis=1))
y_train = houseprice['SalePrice']
cross_val_score(clf, x_train, y_train)
clf.fit(x_train, y_train)
houseprice=pd.read_csv('../input/test.csv')