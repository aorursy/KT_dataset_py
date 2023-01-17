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
dataset=pd.read_csv('/kaggle/input/advertising-dataset/advertising.csv')

dataset.head()
#Dimesionality check

dataset.shape
dataset.isna().any() # No missing values
dataset.dtypes # All are numeric column. Not categorical col is present
import matplotlib.pyplot as plt
fig,axs= plt.subplots(1,3,sharey=True) # sharey : share same y axis across the plot

dataset.plot(kind="scatter",x='TV',y='Sales',ax=axs[0],figsize=(16,8))

dataset.plot(kind="scatter",x='Radio',y='Sales',ax=axs[1],figsize=(16,8))

dataset.plot(kind="scatter",x='Newspaper',y='Sales',ax=axs[2],figsize=(16,8))
feature_x=['TV','Radio','Newspaper']

X=dataset[feature_x]

y=dataset.Sales
from sklearn.linear_model import LinearRegression

from sklearn import model_selection
xtrain,xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.3,random_state=30)

print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)
lm=LinearRegression()

lm.fit(xtrain,ytrain)
print(lm.intercept_,lm.coef_)
#calculation sales price manually with tv,radio, newspaper=50

0.0549163*50+ 0.09294998*50 +0.00974809*50 +4.63226201614081 
x_new=pd.DataFrame({'TV':[50],

                   'Newspaper':[50],

                   'Radio':[50]})

x_new.head()
new_pred=lm.predict(x_new)

new_pred
# lets predict on test data

preds=lm.predict(xtest)

from sklearn.metrics import mean_squared_error

from math import sqrt
print(sqrt(mean_squared_error(ytest,preds)))