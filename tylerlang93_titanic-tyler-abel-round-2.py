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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
# from catboost import CatBoostClassifier
#######################################
########### LOADING THE DATA ##########
#######################################

train = pd.read_csv("PATH")
train.head()
test_data = pd.read_csv("PATH")
# test_data.head()
########################################
######## VARIABLE ANALYSIS #############
########################################

sns.pairplot(train)
# train[train['Survived'] == 1].Fare.mean() #higher
# train[train['Survived'] == 0].Fare.mean() #lower
# train[train['Parch'] != 0].Survived.mean() #0.511 --> make a dummy for 0 / 4 / 5 / 6
# train[train['SibSp'] == 1].Survived.mean() #MUCH HIGHER THAN OTHERS. MAKE DUMMY FOR SibSP being 1
# plt.hist(train[train['Survived'] == 1].Age)
# plt.hist(train[train['Survived'] == 0].Age)


train['total_members'] = train.SibSp + train.Parch + 1
# train[train['total_members'] == 4].Survived.mean() # 0.7!!  Make dummy for fam size of 2-4.
train['small_unit'] = train.total_members.apply(lambda x: 1 if x in [2,3,4] else 0) # INCLUDE THIS IN MODEL
train['Fare_pp'] = train.Fare / train.total_members
train['good_age'] = train.Age.apply(lambda x: 1 if x <= 40 and x>=20 else 0)
train = train.dropna(subset = ['Age'])



test_data['total_members'] = test_data.SibSp + test_data.Parch + 1
test_data['small_unit'] = test_data.total_members.apply(lambda x: 1 if x in [2,3,4] else 0) # INCLUDE THIS IN MODEL
test_data['Fare_pp'] = test_data.Fare / test_data.total_members
test_data['good_age'] = test_data.Age.apply(lambda x: 1 if x <= 40 and x>=20 else 0)

########################################
######## RUNNING THE MODEL #############
########################################

y = train["Survived"]
features = ["Pclass", "Sex", "small_unit", "Fare_pp", "Age"]

X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test_data[features])
X_test.shape
X_test = X_test.fillna(21.8)
X_test.shape
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=324)

# classifier = CatBoostClassifier()
# classifier.fit(X_train, y_train)
# predictions = classifier.predict(X_test)
model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

accuracy_score(y_true = y_test, y_pred = predictions)


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('PATH + NAME', index=False)
print("Your submission was successfully saved!")