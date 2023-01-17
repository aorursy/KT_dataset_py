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
data = pd.read_csv('../input/CustomerData.csv')

data.head()
# !pip install pandas_profiling

import pandas_profiling as pp

pp.ProfileReport(data)
data1 = data.drop(['CustomerID','NoOfGamesBought','NoOfUnitsPurchased'] , axis=1)

data1.head()
corr = data1.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
missing = data1.isnull().sum(axis=0).reset_index()

missing.columns = ['column_name', 'missing_count']

missing['missing_ratio'] = (missing['missing_count'] / data1.shape[0])*100

missing.sort_values(by='missing_ratio', ascending=False)
data1.dtypes
cat_cols = ['FavoriteChannelOfTransaction','FavoriteGame']

for c in cat_cols:

    data1[c] = data1[c].astype('category')

data1.dtypes
import seaborn as sns

sns.distplot(data1['TotalRevenueGenerated'])

print("Skewness: %f" % data1.TotalRevenueGenerated.skew())

print("Kurtosis: %f" % data1.TotalRevenueGenerated.kurt())
data1['log_Revenue']=np.log(data1['TotalRevenueGenerated'])

data1.drop('TotalRevenueGenerated',axis=1,inplace=True)
sns.distplot(data1['log_Revenue'])

print("Skewness: %f" % data1.log_Revenue.skew())

print("Kurtosis: %f" % data1.log_Revenue.kurt())
#BOXPLOTS

import matplotlib.pylab as plt

import seaborn as sns

%matplotlib notebook

fig, axs = plt.subplots()

sns.boxplot(data=data1,orient='h',palette="Set2")

plt.show()
q75, q25 = np.percentile(data1["NoOfGamesPlayed"], [75 ,25])

iqr = q75-q25

print("IQR",iqr)

whisker = q75 + (1.5*iqr)

print("Upper whisker",whisker)
data1=pd.DataFrame(data1)

data1["NoOfGamesPlayed"] = data1["NoOfGamesPlayed"].clip(upper=whisker)
fig, axs = plt.subplots()

sns.boxplot(data=data1,orient='h',palette="Set2")

plt.show()
q75, q25 = np.percentile(data1["FrequencyOFPlay"], [75 ,25])

iqr = q75-q25

print("IQR",iqr)

whisker = q75 + (1.5*iqr)

print("Upper whisker",whisker)
data1=pd.DataFrame(data1)

data1["FrequencyOFPlay"] = data1["FrequencyOFPlay"].clip(upper=whisker)
fig, axs = plt.subplots()

sns.boxplot(data=data1,orient='h',palette="Set2")

plt.show()
data1.dtypes
data2 = pd.get_dummies(data1,columns = ["FavoriteChannelOfTransaction","FavoriteGame"],drop_first=True)

data2.head()
X = data2.copy().drop("log_Revenue",axis=1)

y = data2["log_Revenue"]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
x_train.iloc[:,:8].head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x_train.iloc[:,:8])



x_train.iloc[:,:8] = scaler.transform(x_train.iloc[:,:8])

x_test.iloc[:,:8] = scaler.transform(x_test.iloc[:,:8])
x_train.head()
x_test.head()
#LINEAR MODEL

from sklearn import linear_model

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)



preds_test = linear.predict(x_test)

preds_test1= np.exp(preds_test)

from sklearn.metrics import mean_squared_error
lr_mse = mean_squared_error(preds_test, y_test)

lr_rmse = np.sqrt(lr_mse)

print("Linear Regression MSE on val: %.4f" %lr_mse)

print('Linear Regression RMSE on val: %.4f' % lr_rmse)
def mape(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print('mape:',mape(y_test, preds_test))