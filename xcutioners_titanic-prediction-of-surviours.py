# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# pandas
import pandas as pd
from pandas import Series, DataFrame
# numpy
import numpy as np

#matplotlib, seaborn for Plotting proper graphs(as graphs are proper representation and talks more than number)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
# get titanic & test csv files as a DataFrame
titanic_train_df = pd.read_csv("../input/train.csv")
titanic_test_df    = pd.read_csv("../input/test.csv")

# preview the data
titanic_train_df.head()

titanic_train_df.info()
print("----------------------------")

titanic_test_df.info()
print("----------------------------")
# Lets pick the column with least missing values, in our case Embarked column has two missing values.
# The most repeating value in Embarked column is S, so we are going to fill those two missing values with S.

# only in titanic_df, fill the two missing values with the most occurred value, which is "S".
titanic_train_df["Embarked"] = titanic_train_df["Embarked"].fillna("S")
# Fare

# only for test_df, since there is a missing "Fare" values
titanic_test_df["Fare"].fillna(titanic_test_df["Fare"].median(), inplace=True)
# get average, std, and number of NaN values in titanic_df
average_age_titanic   = titanic_train_df["Age"].mean()
std_age_titanic       = titanic_train_df["Age"].std()
count_nan_age_titanic = titanic_train_df["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test   = titanic_test_df["Age"].mean()
std_age_test       = titanic_test_df["Age"].std()
count_nan_age_test = titanic_test_df["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

# fill NaN values in Age column with random values generated
titanic_train_df["Age"][np.isnan(titanic_train_df["Age"])] = rand_1
titanic_test_df["Age"][np.isnan(titanic_test_df["Age"])] = rand_2

titanic_train_df.info()
print("----------------------------")
titanic_test_df.info()
# Family

# Instead of having two columns Parch & SibSp, 
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
titanic_train_df['Family'] =  titanic_train_df["Parch"] + titanic_train_df["SibSp"]
titanic_train_df['Family'].loc[titanic_train_df['Family'] > 0] = 1
titanic_train_df['Family'].loc[titanic_train_df['Family'] == 0] = 0

titanic_test_df['Family'] =  titanic_test_df["Parch"] + titanic_test_df["SibSp"]
titanic_test_df['Family'].loc[titanic_test_df['Family'] > 0] = 1
titanic_test_df['Family'].loc[titanic_test_df['Family'] == 0] = 0
# Sex

# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex
    
titanic_train_df['Person'] = titanic_train_df[['Age','Sex']].apply(get_person,axis=1)
titanic_test_df['Person'] = titanic_test_df[['Age','Sex']].apply(get_person,axis=1)
# drop unnecessary columns, these columns won't be useful in analysis and prediction
titanic_train_df = titanic_train_df.drop(['PassengerId','Name','Ticket'], axis=1)
titanic_test_df = titanic_test_df.drop(['Name','Ticket'], axis=1)

# Droping Cabin Column as it has lot of NULL or NAN values, and as we cant fill all the NULL values.
titanic_train_df.drop("Cabin",axis=1,inplace=True)
titanic_test_df.drop("Cabin",axis=1,inplace=True)

# drop Parch & SibSp
titanic_train_df = titanic_train_df.drop(['SibSp','Parch'], axis=1)
titanic_test_df    = titanic_test_df.drop(['SibSp','Parch'], axis=1)

# No need to use Sex column since we created Person column
titanic_train_df.drop(['Sex'],axis=1,inplace=True)
titanic_test_df.drop(['Sex'],axis=1,inplace=True)


titanic_train_df.info()
print("----------------------------")
titanic_test_df.info()
titanic_train_df.head()
# Either to consider Embarked column in predictions,
# and remove "S" dummy variable, 
# and leave "C" & "Q", since they seem to have a good rate for Survival.

# OR, don't create dummy variables for Embarked column, just drop it, 
# because logically, Embarked doesn't seem to be useful in prediction.

embark_dummies_titanic  = pd.get_dummies(titanic_train_df['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

embark_dummies_test  = pd.get_dummies(titanic_test_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)

titanic_train_df = titanic_train_df.join(embark_dummies_titanic)
titanic_test_df    = titanic_test_df.join(embark_dummies_test)

#lets drop column Embarked as we created C, Q and to skip multilinear colinearity we dont create S column from Embarked column.
titanic_train_df.drop(['Embarked'], axis=1,inplace=True)
titanic_test_df.drop(['Embarked'], axis=1,inplace=True)
titanic_train_df.head()
# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic  = pd.get_dummies(titanic_train_df['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(titanic_test_df['Person'])
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

titanic_train_df = titanic_train_df.join(person_dummies_titanic)
titanic_test_df    = titanic_test_df.join(person_dummies_test)

titanic_train_df.drop(['Person'],axis=1,inplace=True)
titanic_test_df.drop(['Person'],axis=1,inplace=True)
titanic_train_df.head()
# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_titanic  = pd.get_dummies(titanic_train_df['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(titanic_test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

titanic_train_df.drop(['Pclass'],axis=1,inplace=True)
titanic_test_df.drop(['Pclass'],axis=1,inplace=True)

titanic_train_df = titanic_train_df.join(pclass_dummies_titanic)
titanic_test_df    = titanic_test_df.join(pclass_dummies_test)
titanic_train_df.head()
# convert Age type from float to int
titanic_train_df['Age'] = titanic_train_df['Age'].astype(int)
titanic_test_df['Age']    = titanic_test_df['Age'].astype(int)
# convert Fare from float to int
titanic_train_df['Fare'] = titanic_train_df['Fare'].astype(int)
titanic_test_df['Fare']    = titanic_test_df['Fare'].astype(int)
titanic_train_df.head()
# define training and testing sets

X_train = titanic_train_df.drop("Survived",axis=1)
Y_train = titanic_train_df["Survived"]
X_test  = titanic_test_df.drop("PassengerId",axis=1).copy()
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
# Random Forests
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, random_state=1)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
# get Correlation Coefficient for each feature using Logistic Regression
from sklearn.linear_model import LogisticRegression
coeff_df = DataFrame(titanic_train_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
coeff_df
submission = pd.DataFrame({
        "PassengerId": titanic_test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)
