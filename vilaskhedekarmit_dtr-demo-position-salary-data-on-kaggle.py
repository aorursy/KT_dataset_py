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
print('Hello Vilas')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
data = pd.read_csv('../input/dtr-dataset/Position_Salary_Data.csv')
data.head(10)
real_x = data.iloc[:,1:2].values
real_y = data.iloc[:,2].values

real_x
real_y
data.head(10)
reg = DecisionTreeRegressor(random_state=0)
reg.fit(real_x, real_y)
#newv = [[6]]
y_pred = reg.predict([[6]])
y_pred
x_grid = np.arange(min(real_x),max(real_x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))

plt.scatter(real_x, real_y, color='red')
plt.plot(x_grid,reg.predict(x_grid), color='blue')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
data = pd.read_csv('../input/dtr-dataset/Position_Salary_Data.csv')
data.head(10)
real_x = data.iloc[:,1:2].values
real_y = data.iloc[:,2].values
reg = DecisionTreeRegressor(random_state=0)
reg.fit(real_x, real_y)
y_pred = reg.predict([[6]])
y_pred
x_grid = np.arange(min(real_x),max(real_x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))

plt.scatter(real_x, real_y, color='red')
plt.plot(x_grid,reg.predict(x_grid), color='blue')
plt.title('DTR Demo')
plt.xlabel('Level of Position')
plt.ylabel('Salary')