# For example, here's several helpful packages to load in 



import seaborn as sns

import seaborn.matrix as smatrix

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import seaborn as sns
#data imported

train = pd.read_csv ('../input/train.csv')

train.head()
# dtypes: float64(3), int64(35), object(43) = 79 features and 1460 samples

train.info()
#have been selected only the int and the float

Train_numeric = train [['SalePrice','MSSubClass', 'LotArea','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd']]

Train_numeric.head()
Train_fl_int = train [['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond' , 'YearBuilt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath' , 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces','GarageCars', 'GarageArea', 'WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SalePrice']]

Train_fl_int.head()
Train_numeric.describe()
sns.set()



sns.pairplot(Train_numeric, hue= 'OverallQual')
sns.set(style="white")

corr = Train_numeric.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True







# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.99, square=True, linewidths=.5, cbar_kws={"shrink": .5}, )
corr
sns.stripplot(x="OverallQual", y= 'SalePrice', data=Train_numeric);
sns.jointplot(x="OverallQual", y="SalePrice", data=Train_numeric, kind="hex", );
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn import neighbors, datasets





# import some data to play 

X = train [['MSSubClass', 'LotArea','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd']]  # we only take the first two features. We could

# avoid this ugly slicing by using a two-dim dataset

y = train ['SalePrice']
Train_fl_int = train [['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond' , 'YearBuilt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath' , 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces','GarageCars', 'GarageArea', 'WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SalePrice']]

Train_fl_int.head()
sns.set(style="white")

corr = Train_fl_int.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.99, square=True, linewidths=.5, cbar_kws={"shrink": .5}, )
from sklearn.cross_validation import train_test_split

from sklearn import preprocessing



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=33)



print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)
from sklearn import linear_model

reg = linear_model.LinearRegression()

reg = reg.fit (X_train, y_train)

from sklearn import metrics

print (reg.score (X_train, y_train))

print (reg.score (X_test, y_test))
from sklearn.linear_model import ridge

rid = linear_model.Ridge()

rid = rid.fit (X_train, y_train)

print (rid.score (X_train, y_train))

print (rid.score (X_test, y_test))
Lasso = linear_model.Lasso(alpha = 0.1)

Lasso = reg.fit (X_train, y_train)

print (Lasso.score (X_train, y_train))

print (Lasso.score (X_test, y_test))
Lasso = linear_model.Lasso(alpha = 0.1)

Lasso = reg.fit (X_train, y_train)

print (Lasso.score (X_train, y_train))

print (Lasso.score (X_test, y_test))

from sklearn.linear_model import Lasso

lasso00001 = Lasso( alpha = 0.1). fit( X_train, y_train) 

print (lasso00001.score( X_train, y_train)) 

print (lasso00001.score( X_test, y_test))

print (np.sum( lasso00001.coef_ != 0))

Lars = linear_model.LassoLars (alpha = 0.1)

Lars = Lasso.fit (X_train, y_train)

print (Lars.score (X_train, y_train))

print (Lars.score (X_test, y_test))
BR = linear_model.BayesianRidge()

BR = BR.fit (X_train, y_train)

print (BR.score (X_train, y_train))

print (BR.score (X_test, y_test))
from sklearn import ensemble

rfr = ensemble.RandomForestRegressor()

rfr = rfr.fit (X_train, y_train)

print (rfr.score (X_train, y_train))

print (rfr.score (X_test, y_test))
from sklearn import tree

DTR = tree.DecisionTreeRegressor()

DTR = DTR.fit(X_train, y_train)

print (DTR.score (X_train, y_train))

print (DTR.score (X_test, y_test))
# import some data to play 

X = train [['MSSubClass', 'LotArea','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd']]  # we only take the first two features. We could

# avoid this ugly slicing by using a two-dim dataset

y = train ['SalePrice']
X_1 = Train_fl_int[['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond' , 'YearBuilt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath' , 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces','GarageCars', 'GarageArea', 'WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']]



y_1 = Train_fl_int [['SalePrice']]
X_1.head()
from sklearn.cross_validation import train_test_split

from sklearn import preprocessing



X1_train, X1_test, y1_train, y1_test = train_test_split(X_1, y_1, test_size=0.20, random_state=33)



print (X1_train.shape, y_train.shape)

print (X1_test.shape, y_test.shape)
from sklearn import ensemble

rfr = ensemble.RandomForestRegressor()

rfr = rfr.fit (X1_train, y1_train)

print (rfr.score (X1_train, y1_train))

print (rfr.score (X1_test, y1_test))
test_1_row = X_1.loc[[1000], :]
rfr.predict(test_1_row)
y_1.loc[1000]
from sklearn.metrics import classification_report