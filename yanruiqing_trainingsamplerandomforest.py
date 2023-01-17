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
trainingdata_filepath="../input/trainingdata/trainingdata.csv"
trainingdata = pd.read_csv(trainingdata_filepath)
trainingdata.describe()
trainingdata.columns
y = trainingdata.hip
trainingdata_features=['age', 'weight', 'height', 'BMI', 'waist','energy','alcoholgv']
X=trainingdata[trainingdata_features]
X.describe()
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
train_X.describe()
val_X.describe()
train_y.describe()
val_y.describe()
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
hip_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, hip_preds))
hip_preds
val_y.head()
val_X.head()