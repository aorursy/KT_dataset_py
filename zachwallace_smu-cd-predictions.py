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
smu_data = '/kaggle/input/biggersmudata/BiggerSMUData.csv'

smucd = pd.read_csv(smu_data)
print(smucd.columns)
smucd.columns = ['Incident','Report#','DnTr','DnTo','Location','DoN']

smucd.head()

print(smucd.columns)
smucde = pd.get_dummies(smucd.Location)

smucde.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn import preprocessing

y = (smucd.Location)
feature_names = ['Incident', 'DnTr', 'DnTo']

X = pd.get_dummies(smucd[feature_names])

X_test = smucde
model = RandomForestClassifier(n_estimators=10, max_depth=50, random_state=1)

model.fit(X, y)

predictions = model.predict(X)
output = pd.DataFrame({'Location': predictions})

print(predictions)