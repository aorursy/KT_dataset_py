import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import cufflinks as cf

cf.go_offline()

%matplotlib inline
train = pd.read_csv('../input/train.csv')
#Lest see how the data is distributed

train.head()
train.info()
#Lets do some exploratory data analysis.



#Lets find out the missing data in our data set

#This heatmap shows the null values in yellow line.

sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
#EDA

#Lets see how many of survived andd how many of not.

sns.set_style('whitegrid')

sns.countplot(x = 'Survived', data = train)
#Lets see this survival rate as per sex.

sns.countplot(x = 'Survived', data = train, hue = 'Sex')
#Now the survival rate as per of Pclass

#Here we see the the passenger are in 3rd clas are more likely to not survived.

sns.countplot(x = 'Survived', data = train, hue = 'Pclass')
#Lets see distribution of age in the dataset

#Here we see that mor number of peoples as of the young age between 20 to 30

sns.distplot(train['Age'].dropna(), kde = False, bins = 30)
#Lets look at the sibling and spouse column 

sns.countplot(x = 'SibSp', data = train)
#Now lets explore the fair column

train['Fare'].hist(bins = 40, figsize=(10,4))
#Now lets explore this column with some interactive plot using cufflinks.

train['Fare'].iplot(kind='hist')
#Now lets deal with the missing data.

#Filling the age as the mean of the present ages.

#Lets do it the smart way by considering Pclass.

sns.boxplot(x = 'Pclass', y='Age', data = train)
#Now impute the age as considering the Pclass

def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age

        
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)
#Now lets check our data.

#We see that our data is sucessfully filled the age values.

sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap="viridis")
#Now lets drop the coloumn Cabin because it has high number of missing valus.

train.drop('Cabin', axis = 1, inplace = True)

sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap="viridis")



#Now our data is free from missing values as the plot showing one solid color only.
#Now lets create dummy varibles for the categorical columns.

sex = pd.get_dummies(train['Sex'],drop_first = True)
embark = pd.get_dummies(train['Embarked'],drop_first = True)
#lets join the two dummies we have make in our original dataset

train = pd.concat([train, sex, embark], axis = 1)

train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace = True)
#Now we see that our dataset is good for performing machine learning algorithms.

#We have now all numerical columns



train.drop(['PassengerId'], axis=1, inplace = True)
train.head()
#Now lets train our model to do predictions.

#For that first we need to divide our data set in two datasets

X=train.drop('Survived', axis = 1)

y= train['Survived']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
#Lets create a instance of LogisticRegression

logmodel = LogisticRegression()
#Lets fit a model first

logmodel.fit(X_train, y_train)
#Lets do predictions

predictions = logmodel.predict(X_test)
#we have predict our predictions now.

#Lets check the classification report of our predictions

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))
#Lets check the confusion matrix for the same.

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)