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
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df
df.isnull().sum()
df = df.fillna(0)

df
df = df.drop(columns=['sl_no','salary'])

df
x = df.drop(columns=['status'])

x
df.plot
y = df['status']

y
from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import GridSearchCV 
c_space = np.logspace(-5, 8, 15) 

param_grid = {'C': c_space}
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv = 8)
logreg_cv.fit(x, y) 
logreg_cv.best_params_
logreg_cv.best_score_