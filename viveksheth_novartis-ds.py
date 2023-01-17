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
train = '../input/Train.csv'
test = '../input/Test.csv'
df = pd.read_csv(train)
print(df.head())
print(df.describe())



#Checking for null values

null_cols = df.columns[df.isnull().any()]
df[null_cols].isnull().sum()

group = df.groupby('X_12')
group.groups
print(df[df["X_12"].isnull()][null_cols])
df['X_12'].fillna(1,inplace=True)
# null_cols = df.columns[df.isnull().any()]
# df[null_cols].isnull().sum()

# print(df[df["X_12"].isnull()][null_cols])
group = df.groupby('X_12')
group.groups
y = df.MULTIPLE_OFFENSE
features = ['X_1','X_2','X_3','X_4','X_5','X_6','X_7','X_8','X_9','X_10','X_11','X_12','X_13','X_14','X_15']
X = df[features]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import recall_score


# # Define model. Specify a number for random_state to ensure same results each run
# model = DecisionTreeRegressor(random_state=1)

# # Fit model
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# score = recall_score(y_test,y_pred,average='binary')
# print(score)
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=1)

model.fit(X, y)



df_test = pd.read_csv(test)
# print(df_test.head())

null_cols = df_test.columns[df_test.isnull().any()]
df_test[null_cols].isnull().sum()
df_test['X_12'].fillna(1,inplace=True)


X_test = df_test[features]
# y_test = df_test.MULTIPLE_OFFENSE
# print(X_test)
y_pred = model.predict(X_test)
print(y_pred)
unique, counts = np.unique(y_pred, return_counts=True)
dict(zip(unique, counts))
df_test_out = pd.DataFrame(y_pred,columns=['MULTIPLE_OFFENSE'],dtype='int32')
df_test_out['INCIDENT_ID'] = df_test['INCIDENT_ID']
df_test_out.head()

df_test_out.to_csv('output.csv',index=False)
# import seaborn as sns
# sns.heatmap(df)