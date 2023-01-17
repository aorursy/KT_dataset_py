import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("../input/train.csv")
#Check that the file was read in properly and explore the columns

df.head()
plt.figure(figsize=(12,8))

sns.heatmap(df.isnull(),cbar=False, yticklabels=False, cmap='viridis')
sns.set_style('darkgrid')
sns.countplot(x='Survived', data=df, hue='Pclass')
sns.countplot(x='SibSp', data=df, hue='Survived')
df['Fare'].hist(bins=40)
plt.figure(figsize=(12,6))

sns.boxplot(x='Pclass', y='Age', data=df)
def inpute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else: return 24

    else: return Age
df['Age']=df[['Age','Pclass']].apply(inpute_age, axis=1)
plt.figure(figsize=(12,8))

sns.heatmap(df.isnull(),cbar=False, yticklabels=False, cmap='viridis')
df.drop('Cabin', axis=1, inplace=True)
plt.figure(figsize=(12,6))

sns.heatmap(df.isnull(),cbar=False, yticklabels=False, cmap='viridis')
df.dropna(inplace=True)
plt.figure(figsize=(12,6))

sns.heatmap(df.isnull(),cbar=False, yticklabels=False, cmap='viridis')
df.info()
#We make a new 'Male columns because getDummies will drop one the the dummy variables

#to ensure linear independence.

df['Male'] = pd.get_dummies(df['Sex'], drop_first=True)
#The embarked column indicates where the passenger boarded the Titanic.

#It has three values ['S','C','Q']

embarked = pd.get_dummies(df['Embarked'], drop_first=True)

df = pd.concat([df, embarked], axis=1)
#These columns do not provide us any information for the following reasons:

#PassengerID: we consider 'PassengerID' a randomly assigned ID thus not correlated with surviability

#Name: we are not performing any feature extraction from the name, so we must drop tihs non-numerical column

#Sex: the 'Male' column already captures all information about the sex of the passenger

#Ticket: we are not performing any feature extraction, so we must drop this non-numerical column

#Embarked: we have extracted the dummy values, so those two numerical dummy values encapsulate all the embarked info



df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True)
#Take a look at our new dataframe

df.head()
df.info()
#Seperate the feature columns from the target column

X = df.drop('Survived', axis=1)

y = df['Survived']
#Split the data into two. I don't think this is necessary since there are two files.

#I will keep this here for now

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X, y)
#Read in the test data

test_df = pd.read_csv('../input/test.csv')
#Clean the test data the same way we did the training data

test_df['Age']=test_df[['Age','Pclass']].apply(inpute_age, axis=1)

test_df.drop('Cabin', axis=1, inplace=True)

test_df.dropna(inplace=True)

test_df['Male'] = pd.get_dummies(test_df['Sex'], drop_first=True)

embarked = pd.get_dummies(test_df['Embarked'], drop_first=True)

test_df = pd.concat([test_df, embarked], axis=1)

pass_ids = test_df['PassengerId']

test_df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True)
test_df.tail()
predictions = logmodel.predict(test_df)
submission = pd.DataFrame({

        "PassengerId": pass_ids,

        "Survived": predictions

    })

submission.to_csv('titanic.csv', index=False)