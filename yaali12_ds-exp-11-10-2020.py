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
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

x = np.array([1,2,3,4,5])
y = np.array([7,14,15,18,19])
n = np.size(x)
x_mean = np.mean(x)
y_mean = np.mean(y)
x_mean,y_mean
Sxy = np.sum(x*y)- n*x_mean*y_mean
Sxx = np.sum(x*x)-n*x_mean*x_mean
b1 = Sxy/Sxx
b0 = y_mean-b1*x_mean
print('slope b1 is', b1)
print('intercept b0 is', b0)
plt.scatter(x,y)
plt.xlabel('Independent variable X')
plt.ylabel('Dependent variable y')

y_pred = b1 * x + b0
plt.scatter(x, y, color = 'red')
plt.plot(x, y_pred, color = 'green')
plt.xlabel('X')
plt.ylabel('y')
error = y - y_pred
se = np.sum(error**2)
print('squared error is', se)
mse = se/n
print('mean squared error is', mse)
rmse = np.sqrt(mse)
print('root mean square error is', rmse)
SSt = np.sum((y - y_mean)**2)
R2 = 1- (se/SSt)
print('R square is', R2)

x = x.reshape(-1,1)
regression_model = LinearRegression()
# Fit the data(train the model)
regression_model.fit(x, y)
# Predict
y_predicted = regression_model.predict(x)
# model evaluation
mse=mean_squared_error(y,y_predicted)
rmse = np.sqrt(mean_squared_error(y, y_predicted))
r2 = r2_score(y, y_predicted)
# printing values
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('MSE:',mse)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)

data=pd.read_csv("/kaggle/input/boston-housing-dataset/HousingData.csv")
data
data.head()
import matplotlib.pyplot as plt
data.plot(x="LSTAT", y="MEDV",style="o")
plt.xlabel("LSTAT")
plt.ylabel("MEDV")
plt.show()
x = pd.DataFrame(data["LSTAT"])
y = pd.DataFrame(data["MEDV"])
x,y
data_ = data.loc[:,['LSTAT','MEDV']]
data_.head(5)
from sklearn.model_selection  import train_test_split
print(x_train,y_train,x_test)
y_test= train_test_split(x, y ,test_size=0.2,random_state=1)
print(y_test)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()