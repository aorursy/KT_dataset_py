

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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
customer = pd.read_csv("../input/Ecommerce Customers.csv")
customer.head()
customer.info()
sns.heatmap(customer.corr(), linewidth=0.5, annot=True)
sns.jointplot(data = customer, x = "Time on Website", y = "Yearly Amount Spent")
sns.jointplot(data = customer, x = "Time on App", y = "Yearly Amount Spent")
sns.jointplot(data = customer, x = "Time on App", y = "Length of Membership", kind = "hex")
sns.pairplot(customer)
sns.lmplot(data = customer, x= "Length of Membership", y ="Yearly Amount Spent")
x=customer[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

y=customer['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)
lm.coef_
result = lm.predict(x_test)
plt.scatter(y_test, result)

plt.xlabel("Actual values")

plt.ylabel("Predicted values")
from sklearn import metrics
print('MAE ', metrics.mean_absolute_error(y_test,result))

print('MSE ', metrics.mean_squared_error(y_test,result))

print('RMSE ', np.sqrt(metrics.mean_squared_error(y_test,result)))
metrics.explained_variance_score(y_test,result)
plt.hist((y_test-result))