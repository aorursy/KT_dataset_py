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
import pandas as pd

import numpy as np

%matplotlib inline



train_df = pd.read_csv("../input/train.csv")



# Print the first 5 rows of the dataframe.

train_df.head(5)
# check the stats

train_df.describe()
train_df.hist(figsize=(15,10))
train_df[train_df.Age.isnull()]
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
# drop these columns, since we don't plan to use them for now

train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# let's encode values

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_df['Sex']=le.fit_transform(train_df['Sex'])

# embarked is an object, need convert to string before encoding

train_df['Embarked']=le.fit_transform(train_df['Embarked'].astype('str'))
# Import the linear regression class

from sklearn.linear_model import LinearRegression

# Sklearn also has a helper that makes it easy to do cross validation

from sklearn.cross_validation import KFold



# The columns we'll use to predict the target

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



# Initialize our algorithm class

alg = LinearRegression()

# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.

# We set random_state to ensure we get the same splits every time we run this.

kf = KFold(train_df.shape[0], n_folds=3, random_state=1)



predictions = []

for train, test in kf:

    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.

    train_predictors = (train_df[predictors].iloc[train,:])

    # The target we're using to train the algorithm.

    train_target = train_df["Survived"].iloc[train]

    # Training the algorithm using the predictors and target.

    alg.fit(train_predictors, train_target)

    # We can now make predictions on the test fold

    test_predictions = alg.predict(train_df[predictors].iloc[test,:])

    predictions.append(test_predictions)
import numpy as np



# The predictions are in three separate numpy arrays.  Concatenate them into one.  

# We concatenate them on axis 0, as they only have one axis.

predictions = np.concatenate(predictions, axis=0)



# Map predictions to outcomes (only possible outcomes are 1 and 0)

predictions[predictions > .5] = 1

predictions[predictions <=.5] = 0

accuracy = sum(predictions[predictions == train_df["Survived"]])/len(train_df["Survived"])

print(accuracy)
from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression



# Initialize our algorithm

alg = LogisticRegression(random_state=1)

# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)

scores = cross_validation.cross_val_score(alg, train_df[predictors], train_df["Survived"], cv=3)

# Take the mean of the scores (because we have one for each fold)

print(scores.mean())
titanic_test = pd.read_csv("../input/test.csv")

print(titanic_test.describe())

titanic_test['Age'] = titanic_test['Age'].fillna(train_df['Age'].median())

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0

titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1