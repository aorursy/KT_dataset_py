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
df = pd.read_csv('/kaggle/input/summeranalytics2020/train.csv')
df.head()
df.shape
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['BusinessTravel_le'] = le.fit_transform(df['BusinessTravel'])
df.head()
df['Department_label_encoded'] = le.fit_transform(df['Department'])
df['EducationField_label_encoded'] = le.fit_transform(df['EducationField'])

df.head()
data = df.drop(['BusinessTravel', 'Department', 'EducationField', 'JobRole'], axis=1)
data.head()
df['JobRole_label_encoded'] = le.fit_transform(df['JobRole'])
data = df.drop(['JobRole'], axis=1)
data.head()
data.drop(['BusinessTravel', 'Department', 'EducationField'], axis=1, inplace=True)
data.head()
pd.set_printoptions(max_columns=32)
pd.set_option('max_columns', 32)
data.head()
df['Gender_label_encoded'] = le.fit_transform(df['Gender'])  
df['MaritalStatus_label_encoded'] = le.fit_transform(df['MaritalStatus'])
data['MaritalStatus_label_encoded'] = le.fit_transform(data['MaritalStatus'])
data['Gender_label_encoded'] = le.fit_transform(data['Gender'])  
data.head()
data.drop(['Gender', 'MaritalStatus'], axis=1, inplace=True)
data.head()
data.drop(['Id', 'EmployeeNumber'], axis=1, inplace=True)
data.head()
data['OverTime_label_encoded'] = le.fit_transform(data['OverTime'])
data.head()
data.drop(['OverTime'], axis=1, inplace=True)
data.head()
Y = data['Attrition']
Y.head()
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
clf =  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.3)
clf = RandomForestClassifier(n_estimators=180)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
