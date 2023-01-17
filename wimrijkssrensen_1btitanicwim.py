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
import math

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
train_data["Pclass"].describe()
#TRAIN FILLED

train_mean_age = train_data["Age"].mean()

train_data["Age_filled"] = train_data["Age"].fillna(train_mean_age)

train_data["Age_filled"] = train_data['Age_filled'].apply(math.ceil)



train_data["Embarked_filled"] = train_data["Embarked"].fillna("S")



#TEST FILLED

test_mean_age = test_data["Age"].mean()

test_data["Age_filled"] = test_data["Age"].fillna(test_mean_age)

test_data["Age_filled"] = test_data['Age_filled'].apply(math.ceil)



test_data["Embarked_filled"] = test_data["Embarked"].fillna("S")



#FAMILY FILLED

train_data["Family"] = train_data["SibSp"] + train_data["Parch"]

test_data["Family"] = test_data["SibSp"] + test_data["Parch"]
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex_cat_codes", "SibSp", "Parch", "Age_filled", "Family"]



#SEX TO CATEGORY

train_data["Sex_cat"] = train_data["Sex"].astype("category")

train_data["Sex_cat_codes"] = train_data["Sex_cat"].cat.codes

X = train_data[features].copy()



test_data["Sex_cat"] = test_data["Sex"].astype("category")

test_data["Sex_cat_codes"] = test_data["Sex_cat"].cat.codes

X_test = test_data[features].copy()
import seaborn as sns

sns.boxplot(x=train_data["Fare"])
#TRAIN DATA

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



#SAVE FILE

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")