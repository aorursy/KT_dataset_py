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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



data=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")     #import data

data.head()

data.info()
x = np.array(data.loc[:,'age']).reshape(-1,1)       

y = np.array(data.loc[:,'thalach']).reshape(-1,1)



# Scatter

plt.figure(figsize=[10,10])

plt.scatter(x=x,y=y,color="red")

plt.xlabel("Age")                            # name of label

plt.ylabel("Maximum Heart Rate")

plt.title("MAXIMUM HEART RATE BY AGES")      # title of plot

plt.show()

from sklearn.linear_model import LinearRegression      #sklearn library



linear_reg=LinearRegression()                          #linear regression model



predict_space = np.linspace(min(x), max(x)).reshape(-1,1)

linear_reg.fit(x,y)

predicted = linear_reg.predict(predict_space)



print('R^2 score: ',linear_reg.score(x, y))

# Plot regression line and scatter

plt.plot(predict_space, predicted,color='black', linewidth=3)

plt.scatter(x=x,y=y,color="red")

plt.xlabel("Age")                        

plt.ylabel("Maximum Heart Rate") 

plt.show()
