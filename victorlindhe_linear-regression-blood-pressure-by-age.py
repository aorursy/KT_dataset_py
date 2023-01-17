# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/heart.csv")



xAge = data['age'] # our only feature

yRestingBloodPressure = data['trestbps'] # blood pressure



sns.set()

plt.style.use('classic')

plt.plot(xAge,yRestingBloodPressure,'ro')
# See whats happening when using sci-kit learn

from sklearn.linear_model import LinearRegression 

linearRegression = LinearRegression()



x = np.array(list(map(lambda i : [1, i], xAge))) # adding x0, which is 1

linearRegression.fit(x, yRestingBloodPressure)



xPlot = range(20,80)

yPlot = linearRegression.predict(np.array(list(map(lambda i : [1, i], xPlot))))



line = plt.plot(xAge,yRestingBloodPressure,'ro')

plt.setp(line, linewidth = 3.0)

plt.plot(xPlot,yPlot)

plt.xlabel('Age')

plt.ylabel('Blood pressure')
linearRegressionPoly = LinearRegression(normalize=True) # Feature scaling on



xPoly = np.array(list(map(lambda i : [1, i, i**2], xAge)))

linearRegressionPoly.fit(xPoly, yRestingBloodPressure)



xPlot = range(20,80)

yPlot = linearRegressionPoly.predict(np.array(list(map(lambda i : [1, i, i**2], xPlot))))



line = plt.plot(xAge,yRestingBloodPressure,'ro')

plt.setp(line, linewidth = 3.0)

plt.plot(xPlot,yPlot)

plt.xlabel('Age')

plt.ylabel('Blood pressure')