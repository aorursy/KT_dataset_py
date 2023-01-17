# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra # numerial arrays

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# load data ans store in trainDF



trainDF = pd.read_csv('/kaggle/input/titanic/train.csv')

# Show first 5 rows with the help of head() method

trainDF.head()
# To know numerical features here.

trainDF.describe()
# To show how many rows and null and data types

trainDF.info()



# We can see that, there are 891 rows and 12 columns


# define function here

def bar_plot(column):

    Survived = trainDF[ trainDF['Survived'] == 1 ][ column ].value_counts()

    Dead = trainDF[ trainDF['Survived'] == 0 ][ column ].value_counts()

    newDF = pd.DataFrame([Survived, Dead])

    newDF.index = ['Survived', 'Dead']

    

    # show plot here

    newDF.plot(kind='bar', stacked=True, figsize=( 15, 6 ))

    
# Call the bar_plot() function to show barplot

bar_plot('Sex')

# Now create X and y 

columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']



# Saperate variables

X = trainDF[columns] # for Prediction - Dependent Variable



y = trainDF['Survived'] # Indipendet Variable

y
# To check how many null values and diffrent data type

X.info()
# We can use this method also

X.isnull().sum()



# Here age and Embarked Missing values
#  For this problem we need to give some numeriac values to Age and Embarked Column
# Fill missing values

X['Age'] = X['Age'].fillna(X['Age'].median())

X['Embarked'] = X.fillna(X['Embarked'].value_counts().index[0])

# Import LabelEncoder here



from sklearn.preprocessing import LabelEncoder


# Now, Create an Instance of LabelEncoder

Le = LabelEncoder()



# Convert and Transforming



# For Sex

X['Sex'] = Le.fit_transform(X['Sex'])



# For Embarked

X['Embarked'] = Le.fit_transform(X['Embarked'])
print(X.info(), "\n") # Here we can see all the values are filled.

print(y.describe)
# Load train_test_split method for training and testing

from sklearn.model_selection import train_test_split
#  Create Training and testing Data using X and y

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=.1, random_state =10)

# Here we are using XGBoost Library.



from xgboost import XGBClassifier
# creating instance

classify = XGBClassifier(colsample_bylevel=.9,

                        colsample_bytree=.8,

                        gamma=.99,

                        max_depth=5,

                        min_child_weight=1,

                        n_estimators=10,

                        nthread=10

                        )
# Model Fitting.

classify.fit(X_train, y_train)

score_is =classify.score(X_test, y_test)

print("Score is", score_is * 100)
# Load Testing data

testDF = pd.read_csv('/kaggle/input/titanic/test.csv')



# show fist five rows

testDF.head()
# Test data

test_X = testDF[columns]

test_X.head()
# Fill missing values

test_X['Age'] = test_X['Age'].fillna(test_X['Age'].median())

test_X['Embarked'] = test_X.fillna(test_X['Embarked'].value_counts().index[0])

# Convert and Transforming



# For Sex

test_X['Sex'] = Le.fit_transform(test_X['Sex'])



# For Embarked

test_X['Embarked'] = Le.fit_transform(test_X['Embarked'])
from sklearn.ensemble import RandomForestClassifier


# Try new features

features = ["Pclass", "Sex", "SibSp", "Parch"]

y = trainDF["Survived"]



X = pd.get_dummies(trainDF[features])

X_test = pd.get_dummies(trainDF[features])
# Create and fit the model

model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=0)



model.fit(X, y)
print(model.score(X_test, y) *100)
# Predict the Random Forest Model

predictions = model.predict(X_test)



# print("Score", predictions.score_)

outputRFM = pd.DataFrame({'PassengerId': trainDF.PassengerId, 'Survived': predictions})

outputRFM.to_csv('my_submissionRFM.csv', index=False)

print("Your submission was successfully saved!")


prediction = classify.predict(test_X)

#  Now create Output DataFrame



dict = { 'PassengerId' : testDF['PassengerId'],

       'Survived' : prediction

       }



outputXGB = pd.DataFrame(dict)



# outputDF.head()
outputXGB.to_csv('my_submission_XGB.csv', index=False)

outputXGB
