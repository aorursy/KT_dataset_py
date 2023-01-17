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
# Read datasets

diamonds = pd.read_csv("../input/diamonds.csv", index_col = 0)
# Print breaker

def pb():

    print('---------------------------------------------------------')
# import extra modules

import matplotlib.pyplot as plt

import seaborn as sns
# First look at the dataset



print(diamonds.head(10))

print(diamonds.describe())

print(diamonds.cut.unique())

print(diamonds.color.unique())

print(diamonds.clarity.unique())
# Define the Encode Function

def OrdererdListEncoder(ord_list):

    """

    ord_list: a python list with predetermined order

    return a dictionary that maps values in ord_list to a ranking

    """

    return {ord_list[i]: len(ord_list) - i for i in range(len(ord_list))}
# Encode Carat, Color and Cut

cut_rank = OrdererdListEncoder(['Ideal','Premium','Very Good', 'Good', 'Fair'])

color_rank = OrdererdListEncoder(list('DEFGHIJ'))

clarity_rank = OrdererdListEncoder(['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1'])



diamonds['cut'] = diamonds['cut'].apply(lambda x: cut_rank[x])

diamonds['color'] = diamonds['color'].apply(lambda x: color_rank[x])

diamonds['clarity'] = diamonds['clarity'].apply(lambda x: clarity_rank[x])



diamonds.head()
# Making some plots

sns.set(color_codes = True)

sns.regplot(x = 'carat', y = 'price', data=diamonds)

sns.jointplot(x = 'depth', y = 'table', data = diamonds, kind = 'kde')
# Tranform the data

# Add Carat Square and Carat Expotential Columns - since Size is the key factor of diamond values



diamonds['carat_sqrd'] = np.square(diamonds['carat'])

diamonds['carat_exp'] = np.exp(diamonds['carat'])

diamonds = diamonds.drop(['x','y','z'], axis = 1)

diamonds.head()
sns.regplot(x = 'carat', y = 'price', data=diamonds.loc[(diamonds['clarity']>=4) & (diamonds.cut >=4) & (diamonds.clarity >= 4), :])
# Preprocessing

from sklearn.model_selection import train_test_split



# Y

y = diamonds.price



# X

X = diamonds.loc[:,diamonds.columns != 'price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1)



X_train_reg = X_train.drop(['carat_sqrd', 'carat_exp'], axis = 1)

X_train_sqrd = X_train.drop(['carat', 'carat_exp'], axis = 1)

X_train_exp = X_train.drop(['carat_sqrd', 'carat'], axis = 1)

# Print Perfromance

import statsmodels.api as sm

mod_reg = sm.OLS(y_train, X_train_reg)

mod_sqrd = sm.OLS(y_train, X_train_sqrd)

mod_exp = sm.OLS(y_train, X_train_exp)



res1 = mod_reg.fit()

res2 = mod_sqrd.fit()

res3 = mod_exp.fit()



print(res1.summary())

pb()

print(res2.summary())

pb()

print(res3.summary())
# Train and Test

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_train_reg, y_train)



X_test = X_test.drop(['carat_sqrd','carat_exp'], axis = 1)



y_pred = reg.predict(X_test)



# store errors

errors = y_pred - y_test
# plot error information

sns.distplot(errors.values)
# Check where errors happen

error_threshold = 0.05

err_idx = errors[(np.abs(errors/ y_test) > error_threshold)].index



print("With {0} as threshold, the model's accuracy is {1}".format(error_threshold,1 - len(err_idx)/len(errors)))



# Absolute Error

err_abs_thr = 1000

err = np.sum(np.abs(errors) > err_abs_thr) / len(errors)

print("With {0} as absolute threshold, the model's accuracy is {1}".format(err_abs_thr, 1 - err))



error_samples = y_test[err_idx]

error_samples.describe()

sns.distplot(error_samples,kde=False)