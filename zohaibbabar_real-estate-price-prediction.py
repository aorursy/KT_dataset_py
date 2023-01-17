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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

from sklearn.linear_model import LinearRegression

sns.set()
raw_data = pd.read_csv("../input/real-estate-price-prediction/Real estate.csv")
raw_data.head()
data = raw_data.copy()
data = data.drop(['No'],axis=1)
data.describe(include='all')
y = data["Y house price of unit area"]

x1 = data[['X1 transaction date','X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores', 'X5 latitude']]
x = sm.add_constant(x1)

results = sm.OLS(y,x).fit()

results.summary()
sns.distplot(data['Y house price of unit area'])
q = data["Y house price of unit area"].quantile(0.99)

data_1 = data[data['Y house price of unit area']<q]

data_1.describe(include='all')
sns.distplot(data_1['Y house price of unit area'])
sns.distplot(data_1['X3 distance to the nearest MRT station'])
q= data_1['X3 distance to the nearest MRT station'].quantile(0.99)

data_2 = data_1[data_1['X3 distance to the nearest MRT station']<q]

data_2.describe(include='all')
sns.distplot(data_2['X3 distance to the nearest MRT station'])
sns.distplot(data_2['X4 number of convenience stores'])
data_cleaned = data_2.reset_index(drop=True)
data_cleaned.describe(include='all')
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) 

ax1.scatter(data_cleaned['X2 house age'],data_cleaned['Y house price of unit area'])

ax1.set_title('Y hourse price of unit area and X2 house age')

ax2.scatter(data_cleaned['X3 distance to the nearest MRT station'],data_cleaned['Y house price of unit area'])

ax2.set_title('Y house price of unit area and X3 distance to the nearest MRT station')

ax3.scatter(data_cleaned['X4 number of convenience stores'],data_cleaned['Y house price of unit area'])

ax3.set_title('Y house price of unit area and X4 number of convenience stores')





plt.show()
sns.distplot(data_cleaned['Y house price of unit area'])
log_price = np.log(data_cleaned['Y house price of unit area'])



data_cleaned["log_price"] = log_price



data_cleaned
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) 

ax1.scatter(data_cleaned['X2 house age'],data_cleaned['log_price'])

ax1.set_title('log_price and X2 house age')

ax2.scatter(data_cleaned['X3 distance to the nearest MRT station'],data_cleaned['log_price'])

ax2.set_title('log_price and X3 distance to the nearest MRT station')

ax3.scatter(data_cleaned['X4 number of convenience stores'],data_cleaned['log_price'])

ax3.set_title('log_price and X4 number of convenience stores')





plt.show()
data_cleaned = data_cleaned.drop(['Y house price of unit area'],axis=1)
from statsmodels.stats.outliers_influence import variance_inflation_factor



# To make this as easy as possible to use, we declare a variable where we put

# all features where we want to check for multicollinearity

# since our categorical data is not yet preprocessed, we will only take the numerical ones

variables = data_cleaned[['X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores']]



# we create a new data frame which will include all the VIFs

# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)

vif = pd.DataFrame()



# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 

vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]

# Finally, I like to include names so it is easier to explore the result

vif["Features"] = variables.columns
vif
targets = data_cleaned['log_price']

inputs = data_cleaned.drop(['log_price'],axis=1)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=365)
reg = LinearRegression()

reg.fit(x_train,y_train)
y_hat = reg.predict(x_train)
plt.scatter(y_train,y_hat)



plt.xlabel('Target y_train', size=18)

plt.ylabel('Predictions y_hat', size=18)



plt.show()
# We can plot the PDF of the residuals and check for anomalies

sns.distplot(y_train - y_hat)



# Include a title

plt.title("Residuals PDF", size=18)
reg.score(x_train,y_train)
reg.intercept_
reg.coef_
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])

reg_summary["Weights"]= reg.coef_

reg_summary
y_hat_test = reg.predict(x_test)
plt.scatter(y_test,y_hat_test)

plt.xlabel('Targets y_test',size=18)

plt.ylabel('Predictions y_hat_test', size=18)

plt.show()
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])

df_pf.head()
df_pf['Target'] = np.exp(y_test)

df_pf

y_test = y_test.reset_index(drop=True)



# Check the result

y_test.head()
# Again, we need the exponential of the test log price

df_pf['Target'] = np.exp(y_test)

df_pf
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)

df_pf
pd.options.display.max_rows = 999

# Moreover, to make the dataset clear, we can display the result with only 2 digits after the dot 

pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Finally, we sort by difference in % and manually check the model

df_pf.sort_values(by=['Difference%'])