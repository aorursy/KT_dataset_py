# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
titanic_df=pd.read_csv("../input/train.csv")

test_df=pd.read_csv("../input/test.csv")

titanic_df.info()

test_df.info()
titanic_df.describe()
titanic_df.head()
titanic_df.loc[titanic_df["Sex"]=="male","Sex"]=0

titanic_df.loc[titanic_df["Sex"]=="female","Sex"]=1
titanic_df["S"]=0

titanic_df.loc[titanic_df["Embarked"]=="S","S"]=1
titanic_df.head()
titanic_df["Embarked"].unique()
titanic_df["C"]=0

titanic_df.loc[titanic_df["Embarked"]=="C","C"]=1

titanic_df["Q"]=0

titanic_df.loc[titanic_df["Embarked"]=="Q","Q"]=1
titanic_df.head(10)
titanic_df["Cabin"].unique()
titanic_df.drop(["PassengerId","Name","Cabin","Ticket"],axis=1)
titanic_df.drop(["Embarked"],axis=1)

titanic_df.describe()
titanic_df["Age"]=titanic_df["Age"].fillna(titanic_df["Age"].mean())
titanic_df.describe()
titanic_df.info()
titanic_df=titanic_df.drop(["PassengerId","Name","Ticket","Cabin","Embarked"],axis=1)
titanic_df.head()
titanic_df.describe()
titanic_df["Sex"]=titanic_df["Sex"].astype(int)
titanic_df.describe()
test_df.info()
test_df.describe()
test_df.loc[test_df["Sex"]=="male","Sex"]=0

test_df.loc[test_df["Sex"]=="female","Sex"]=1
test_df.info()
test_df["Sex"]=test_df["Sex"].astype(int)
test_df.info()
test_df["Age"]=test_df["Age"].fillna(test_df["Age"].mean())
test_df["Fare"]=test_df["Fare"].fillna(test_df["Fare"].mean())
test_df["S"]=0

test_df.loc[test_df["Embarked"]=="S","S"]=1

test_df["C"]=0

test_df.loc[test_df["Embarked"]=="C","C"]=1

test_df["Q"]=0

test_df.loc[test_df["Embarked"]=="Q","Q"]=1
test_array=test_df.drop(["PassengerId","Name","Ticket","Cabin","Embarked"],axis=1).values
from sklearn.ensemble import RandomForestClassifier
titanic_array=titanic_df.values

titanic_array.shape
train_x=titanic_array[:,1:]

train_y=titanic_array[:,0]

test_x=test_array



clf = RandomForestClassifier(criterion='gini', 

                             n_estimators=700,

                             min_samples_split=10,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)

clf = clf.fit(train_x, train_y)

y_predicted=np.asarray(clf.predict(test_x),dtype=int)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": y_predicted

    })

submission.to_csv('titanic.csv', index=False)