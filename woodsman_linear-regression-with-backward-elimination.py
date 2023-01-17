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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

from sklearn import linear_model

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

%matplotlib inline
df = pd.read_csv("/kaggle/input/startup-logistic-regression/50_Startups.csv")

df.head()
df.isna().sum()
sns.pairplot(df, kind='reg', diag_kind='kde')
x = df.iloc[:,:4].values

y = df.loc[:, 'Profit'].values

print('features = ', x)

print('\ntarget = ', y)


transformer = ColumnTransformer(transformers = [("asda",OneHotEncoder(),[3])],remainder = 'passthrough')



x = transformer.fit_transform(x)

x
x = x[:,1:]

x


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

scalar = StandardScaler()

x_train[:,3:] = scalar.fit_transform(x_train[:,3:]) 

x_test[:,3:] =scalar.fit_transform(x_test[:,3:])
model = linear_model.LinearRegression()

model.fit(x_train,y_train)

bscr = model.score(x_test,y_test)

print(bscr)
x = np.append(np.ones((x.shape[0],1),dtype=np.int), values = x, axis=1)

x
import statsmodels.api as sm

x_opt = np.array(x[:, [0, 1, 2, 3, 4, 5]], dtype=float)

regressor = sm.OLS(y, x_opt).fit()

regressor.summary()
x_opt = np.array(x[:, [0, 1, 3, 4, 5]], dtype=float)

regressor = sm.OLS(y, x_opt).fit()

regressor.summary()
x_opt = np.array(x[:, [0, 3, 4, 5]], dtype=float)

regressor = sm.OLS(y, x_opt).fit()

regressor.summary()
x_opt = np.array(x[:, [0, 3, 5]], dtype=float)

regressor = sm.OLS(y, x_opt).fit()

regressor.summary()
x_opt = np.array(x[:, [0, 3]], dtype=float)

regressor = sm.OLS(y, x_opt).fit()

regressor.summary()
#spliting the data into training and test 

x_opt = np.array(x[:, [3]], dtype=float)

x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size = 0.2, random_state = 0)



# train the linear reggression model

model = linear_model.LinearRegression()

model.fit(x_train,y_train)

scr = model.score(x_test,y_test)



print('Score before Backward Elimination :', bscr)

print('\nFinal Score after Backward Elimination :', scr)
