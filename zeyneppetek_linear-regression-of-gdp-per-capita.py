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
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("2019.csv",sep=",")

x=df.iloc[:,3].values.reshape(-1,1)
y=df.iloc[:,2].values.reshape(-1,1)


plt.scatter(x,y)
plt.xlabel("GDP per capita")
plt.ylabel("score")
plt.show()

#%% Linear Regression

from sklearn.linear_model import LinearRegression
regrs=LinearRegression()
predict_space=np.linspace(min(x),max(x)).reshape(-1,1)

regrs.fit(x,y)

predic=regrs.predict(predict_space)

##R-square

print("r^2 score:", regrs.score(x,y))

##Regressionplot
plt.plot(predict_space,predic,color="red",linewidth=3)
plt.scatter(x,y)
plt.xlabel("GDP per capita")
plt.ylabel("score")
plt.show()