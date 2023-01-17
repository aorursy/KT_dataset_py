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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
#import library
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#import data
dt_train = pd.read_csv('/kaggle/input/titanic/train.csv')#training set
dt_test = pd.read_csv('/kaggle/input/titanic/test.csv')#test set
#simple feature engineering (just for beginners)
# Create target object and call it y
y_train = dt_train.Survived
# Create X
features = ["Pclass", "Sex", "SibSp", "Parch"]
X_train = pd.get_dummies(dt_train[features])
X_test = pd.get_dummies(dt_test[features])
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
reg = LinearRegression()
cv_scores = cross_val_score(reg, X_train, y_train, cv=5)
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))