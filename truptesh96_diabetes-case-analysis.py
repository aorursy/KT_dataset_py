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
import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

data.head()
y = data.iloc[::,-1]

x = data.iloc[::, 0:-1]
def get_na(data):

    flag = False

    na_cols = dict(data.isna().sum())

    for x in na_cols.keys():

        if(na_cols[x] > 0):

            flag = True

            print(x,na_cols[x])

    if(flag == False):

        print("No Columns having NA values :)")
get_na(data)
data.columns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
glm = LogisticRegression(solver='liblinear')

model = glm.fit(x_train,y_train)

predicted = model.predict(x_test)
conf_mat = confusion_matrix(y_test, predicted)

conf_mat
from sklearn.ensemble import RandomForestClassifier
rf_obj = RandomForestClassifier(n_estimators=1000, criterion='gini', min_samples_split=2, min_samples_leaf=5, 

min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 

min_impurity_split=None, bootstrap=True, n_jobs=1000, random_state=None, verbose=0)

model_rf = rf_obj.fit(x_train, y_train)

predicted_rf = model.predict(x_test)
conf_mat = confusion_matrix(y_test, predicted_rf)
conf_mat