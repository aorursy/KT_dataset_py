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

import pandas as pd

from sklearn import datasets, linear_model

import matplotlib.pyplot as plt



length = 1000    #total number of data rows

x = np.arange(length, dtype=float).reshape((length, 1))

y = x + (np.random.rand(length)*10).reshape((length, 1))   #define equation first! 



# print("x = ", x)

# print("y = ", y)



data = pd.read_csv('/kaggle/input/student-performance/datasets_74977_169835_StudentsPerformance.csv', index_col=False, header=0)

print(data.head(3))

x = data['reading score'].values

y = data['writing score'].values



x = x.reshape(length, 1)

y = y.reshape(length, 1)



regr = linear_model.LinearRegression()

regr.fit(x, y)



# plot the scatter chart

plt.scatter(x, y,  color='red')

plt.plot(x, regr.predict(x), color='blue', linewidth=3)

plt.xticks(())

plt.yticks(())

plt.show()