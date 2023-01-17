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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats

import pandas_profiling



train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head(3)

test.head(3)
train.shape, test.shape
train.info()
train.describe()
# lets understand our target variable sale price

# descrptive stats of price



train ['SalePrice'].describe()
sns.distplot(train['SalePrice']);
# what is the relationship between the price and the rest of the features

# Is there a correlation

# Perform correlation



corr = train.corr()

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

print(corr.SalePrice)

#scatterplot of the above identified top corr features



sns.set()

cols = [ 'SalePrice','OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'TotRmsAbvGrd', 'YearBuilt']

sns.pairplot(train[cols], height = 2.5)

plt.show();



# after viewing the scatter plots, I have adjusted the code to remove FullBath and GarageCars - they seem not to give much info
# We have identified missing values in various columns in our previous codes

# Lets see missing values in an assending order

# adjust count to only show values with missing data

# are the features among the most corr or least corr



total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/1460).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)

# drop the null observation in Electrical 

train = train.drop(train.loc[train['Electrical'].isnull()].index)
# drop all columns with null

# Note for better performance, we will review this section



train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train.shape
# 2. check for duplicates entries, in Id

# If duplicates. . delete

# drop Id column since we will not need to use it in our analysis



id_unique = len(set(train.Id))

id_total = train.shape[0]

id_dup = id_total - id_unique

print(id_dup)



#drop id column

train.drop(['Id'],axis =1,inplace=True)



# no duplicates :-)



# check the shape of the train dataset at this point

train.shape
# Outliers

# data values 1.5 times the interquartile range above the third quartile or below the first quartile - IQR rule



# outliers in price

stat = train.SalePrice.describe()



IQR = stat ['75%'] - stat ['25%']

upper = stat ['75%'] + 1.5 * IQR

lower = stat ['25%'] - 1.5 * IQR

print ('The lower and upper bound for suspected outliers in SalePrice are {} and {},'. format (upper, lower))
train[train.SalePrice == 3875.0]
train[train.SalePrice > 340075]
train.duplicated().sum()
#get dummies for categorical data

train = pd.get_dummies(train)
# Display the first 5 rows of the last 12 columns to confirm that categorical features been converted to numerical 0 & 1

train.iloc[:,5:].head(5)
train.shape
# define X and Y axis



X_train = train.drop(['SalePrice'], axis=1)

Y_train = train['SalePrice']
#Use numpy to convert to array

Xtrain = np.array(X_train)

Ytrain = np.array(Y_train)
# use decision tree



from sklearn.tree import DecisionTreeRegressor

decision_tree = DecisionTreeRegressor()

decision_tree.fit(Xtrain, Ytrain)

acc_decision_tree = round(decision_tree.score(Xtrain, Ytrain) * 100, 2)

acc_decision_tree
# use random forest

# Import the model we are using

from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data

rf.fit(Xtrain, Ytrain);

rf
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# drop all features dropped during training



test = test.drop (['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage','GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual','MasVnrArea','MasVnrType'], axis =1)
test.shape
#get dummies for categorical data

test = pd.get_dummies(test)

test.shape