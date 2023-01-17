# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
## Importing our Data into a Pandas DataFrame
honda = pd.read_csv("../input/Honda CRV Data_042015.csv", low_memory=False)
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 24
fig_size[1] = 9
honda.hist();
honda = honda.dropna()
# features in our models are all dimensions except price
x = honda.iloc[:,1:]
y = honda.iloc[:,[0]]
y.head()
x.head()
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
lgrg = LogisticRegression()
# We have 10 variables that might explain price - let's try to reduce it to 5
strong = RFE(lgrg,5)
strong = strong.fit(x,y)
print(strong.support_)
print(strong.ranking_)
x.head(2)
# Our analysis seems to suggest that Model Year, EX, New Listing, Certified, and 2WD are the most significant variables
# I am going to run an OLS model with ALL variables and the reduced variable
# lets define x_rd as the reduced set of variables
x_rd = x.iloc[:,[1,2,5,7,8]]
x_rd.head(2)
x_rd.head(2)
import statsmodels.api as sm
import statsmodels.formula.api as smf
x = sm.add_constant(x)
honda_value_all = sm.OLS(y,x).fit()
print(honda_value_all.summary())
## Now let us try the reduced model
x_rd = sm.add_constant(x_rd)
honda_value_rd = sm.OLS(y,x_rd).fit()
print(honda_value_rd.summary())
Ford = pd.read_csv("../input/Ford Fiesta_042015.csv", low_memory=False)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 24
fig_size[1] = 9
Ford.hist()
Ford = Ford.dropna()
FX = Ford.iloc[:,1:]
FY = Ford.iloc[:,[0]]
FX = sm.add_constant(FX)
Ford_Value = sm.OLS(FY,FX).fit()
print(Ford_Value.summary())
Toyota = pd.read_csv("../input/Toyota Corolla_042015.csv")
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 24
fig_size[1] = 9
Toyota.hist(bins=20)
Toyota = Toyota.dropna()
TX = Toyota.iloc[:,1:]
TY = Toyota.iloc[:,[0]]
TO = Toyota.iloc[:,[0,1,4,5,6,7,8,9,11,12,13]]
TO.head()
Toyota_Value = sm.OLS(TY,TX).fit()
print(Toyota_Value.summary())
print(Toyota.describe())
print(honda.describe())
print(FX.describe())