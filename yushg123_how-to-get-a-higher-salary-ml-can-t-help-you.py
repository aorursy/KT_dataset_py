# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data
plt.hist(data['salary'], bins = 50)
ax = sns.boxplot(x=data["salary"])
plt.figure(figsize=(14,12))

data2 = data.loc[:,data.columns != 'Id']

sns.heatmap(data2.corr(), linewidth=0.2, cmap="YlGnBu", annot=True)
columns = data.columns

for col in columns:

    if data[col].dtype != 'object' and col != 'sl_no' and col != 'salary':

        plt.scatter(data[col], data['salary'])

        plt.xlabel(col)

        plt.ylabel('Salary')

        plt.show()
ax = sns.barplot(x="gender", y="salary", data=data)
ax = sns.barplot(x="workex", y="salary", data=data)
ax = sns.barplot(x="ssc_b", y="salary", data=data)
ax = sns.barplot(x="hsc_b", y="salary", data=data)
ax = sns.barplot(x="degree_t", y="salary", data=data)
ax = sns.barplot(x="hsc_s", y="salary", data=data)
ax = sns.barplot(x="specialisation", y="salary", data=data)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



data = data[data['status'] == 'Placed']



cols_to_use = ['gender', 'workex', 'degree_t', 'hsc_s', 'specialisation']

for col in data.columns:

    if data[col].dtype == 'object':

        le.fit(data[col])

        data[col] = le.transform(data[col])





y = data['salary']

#x = data[cols_to_use]

x = data.drop(['sl_no', 'salary'], axis=1)

x = pd.get_dummies(x)



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)



lr = LinearRegression()



lr.fit(x_train, y_train)



rf = RandomForestRegressor()

rf.fit(x_train, y_train)



print('R2 score of linear regression is ' + str(lr.score(x_test, y_test)))

print('R2 score of random forest is ' + str(rf.score(x_test, y_test)))