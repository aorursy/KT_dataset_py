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
data = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")

data.head()
data.drop(['id'], axis=1, inplace=True)
temp_data = data[data.columns[0:88]]

temp_data.head()
corr_data = temp_data.corr()

corr_data.head()
corr_flag = 0

for column, content in corr_data.items():

    for index, value in content.items():

        if (column == index):

            continue

        else :

            if (value >= 0.85):

                corr_flag += 1

print(corr_flag)
unique_values = temp_data.nunique()

unique_values = unique_values.values

np.amin(unique_values)
print(temp_data.shape)

from sklearn.feature_selection import VarianceThreshold

transform = VarianceThreshold(threshold=0.5)

sel = transform.fit(temp_data)

delete_list = []

for i, variance in enumerate(sel.variances_):

    if variance < 0.5:

        delete_list.append(i)

delete_list = ["col_" + str(i) for i in delete_list]

selected_data = pd.DataFrame(transform.fit_transform(temp_data))

selected_data['target'] = data['target']

print(selected_data.shape)

selected_data.head()

print(f"{len(delete_list)} columns dropped")
X = pd.DataFrame(selected_data.iloc[:, 0:83].copy())

y = selected_data[['target']]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=121)

print(len(X_train), len(X_test))
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)

scaled_X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV

regressor = LogisticRegression(class_weight="balanced")

#parameters = {"C":[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}

#parameters = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}

regressor.fit(scaled_X_train, y_train)

y_pred = regressor.predict(scaled_X_test)

score = roc_auc_score(y_test, y_pred)

#acc  = regressor.score(scaled_X_test, y_test)

print(f"Score : {score}")

#clf = GridSearchCV(regressor_cv, parameters, verbose=1, scoring='roc_auc', cv=5)

#clf.fit(scaled_X_train, y_train)

#clf.score(scaled_X_test, y_test)
test_data = pd.read_csv("/kaggle/input/minor-project-2020/test.csv")

test_data_mod = test_data.drop('id', axis=1, inplace=False)

for column in delete_list:

    test_data_mod.drop(column, axis=1, inplace=True)

print(test_data_mod.shape)

test_data_mod.head()
scaled_X_true_test = scaler.transform(test_data_mod)

y_pred = regressor.predict(scaled_X_true_test)

submission = pd.DataFrame({'id':test_data['id'],'target':y_pred})

path = '/kaggle/working/predictions_2.csv'

submission.to_csv(path,index=False)

print('Saved file to: ' + path)