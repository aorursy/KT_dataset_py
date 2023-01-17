#imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.graphics.mosaicplot import mosaic

from sklearn.linear_model import LogisticRegression

from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier
#load data & add the two for data cleaning

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

full = pd.concat([train,test]).set_index("PassengerId")

#Description of the data

print(full.head())
#Data--Sex

#set the values to numeric

full.set_value(full["Sex"] == "male", "Sex", 0)

full.set_value(full["Sex"] == "female", "Sex", 1)
#Data--Age

print("The percentage of NaN Values in Age column is: %s %%" % (((pd.isnull(full["Age"]).sum()) * 100)/1309))

plt.figure(figsize=(8,6))

#Distribution of the original Age values (blue)

sns.distplot(full["Age"].dropna().astype(int), bins=70, label="old")

#fill the NaN values in Age with random values

for index, row in full.iterrows():

    if np.isnan(row["Age"]):

        rand = np.random.randint(full["Age"].mean() - full["Age"].std(), 

                                 full["Age"].mean() + full["Age"].std())        

        full.set_value(index, "Age", rand)

#convert all float values to int       

full["Age"] = full["Age"].astype(int)

#Distribution of the new Age values (green)

sns.distplot(full["Age"], bins=70, label="new")

plt.legend()

#Distribution of survived/not survived passengers by age

plt.figure(figsize=(10,4))

av_age = full.groupby(full["Age"]).mean()["Survived"]

av_age_plot = sns.barplot(x=av_age.index, y=av_age.values)
#Data--Cabin

print("The percentage of NaN Values in Cabin column is: %s %%" % (((pd.isnull(full["Cabin"]).sum()) * 100)/1309))

#Since there are so many NaN values, we can discard the Cabin column

full = full.drop("Cabin", axis=1)
#Data--Embarked

print("The percentage of NaN Values in Embarked column is: %s %%" % (((pd.isnull(full["Embarked"]).sum()) * 100)/1309))

print(full[full["Embarked"].isnull()])

#Both passengers in the NaN columns paid a Fare of 80.0 and boarded in the 1st class.

#We can use this fact to fill the appropriate value.

tmp = full[["Pclass", "Fare"]][full["Embarked"].isnull()]

new_value = full[full["Pclass"]==1].groupby(full["Embarked"]).median()

print(new_value["Fare"])

#Since C has the closest value to 80.0, we can fill in 'C'

full["Embarked"] = full["Embarked"].fillna('C')

#set the values to numeric.

full.set_value(full["Embarked"] == "S", "Embarked", 0)

full.set_value(full["Embarked"] == "C", "Embarked", 1)

full.set_value(full["Embarked"] == "Q", "Embarked", 2)
#Data--Fare

print("The percentage of NaN Values in Fare column is: %s %%" % (((pd.isnull(full["Fare"]).sum()) * 100)/1309))

print(full[full["Fare"].isnull()])

#Passenger in the NaN column embarked at 0("S"), and boarded in the 3rd class.

new_value_2 = full[full["Pclass"] == 3][full["Embarked"] == 0]["Fare"].median()

full["Fare"] = full["Fare"].fillna(new_value_2)
#Data --Name

print("The percentage of NaN Values in Name column is: %s %%" % (((pd.isnull(full["Fare"]).sum()) * 100)/1309))

#While the Name value itself might be useless in analysis, we can extract the respective titles.

#split_title: function that helps creating the title column.

def split_title(x):

    return (x.split(",")[1].split(".")[0].strip())

#Creating title column using the split_title function.

full["Title"] = full["Name"].apply(split_title)

#Table of the distribution of title by sexes

title_by_sex = pd.DataFrame(index = full["Title"].drop_duplicates().values)

title_by_sex["Male"] = full[full["Sex"] == 0]["Title"].value_counts()

title_by_sex["Female"] = full[full["Sex"] == 1]["Title"].value_counts()

title_by_sex = title_by_sex.fillna(value = 0)

print(title_by_sex)

#It seems that we can only keep the 4 titles, and set the rest to "Rare Title"

rare_title = ["Don", "Dona", "Rev", "Dr", "Major", "Lady", "Sir",

              "Col", "Capt", "the Countess", "Jonkheer"]

#Putting "Mlle" & "Ms" to "Miss", "Mme" to "Mr", and other titles to "Rare Title"             

for index, row in full.iterrows():

    if row['Title'] == "Mlle":

        full.set_value(index, 'Title', 'Miss')

    elif row['Title'] == "Ms":

        full.set_value(index, 'Title', 'Miss')

    elif row['Title'] == "Mme":

        full.set_value(index, 'Title', 'Mrs')

    elif row['Title'] in rare_title:

        full.set_value(index, 'Title', 'Rare Title')

#Table of the distribution of title by sexes        

title_by_sex2 = pd.DataFrame(index = ["Master", "Miss", "Mr", "Mrs", "Rare Title"])

title_by_sex2["Male"] = full[full["Sex"] == 0]["Title"].value_counts()

title_by_sex2["Female"] = full[full["Sex"] == 1]["Title"].value_counts()

title_by_sex2 = title_by_sex2.fillna(0)

print(title_by_sex2)

#Surname column: column of every surnames (might be useful for additional research)

#split_surname: function that helps creating the surname column

def split_surname(x):

    return (x.split(",")[0])

#Creating surname column using the function.

full["Surname"] = full["Name"].apply(split_surname)

#set the values to numeric

full.set_value(full["Title"] == "Mr", "Title", 0)

full.set_value(full["Title"] == "Mrs", "Title", 1)

full.set_value(full["Title"] == "Miss", "Title", 2)

full.set_value(full["Title"] == "Master", "Title", 3)

full.set_value(full["Title"] == "Rare Title", "Title", 4)
#Data -- Parch & SibSp

print("The percentage of NaN Values in Parch column is: %s %%" % (((pd.isnull(full["Parch"]).sum()) * 100)/1309))

print("The percentage of NaN Values in SibSp column is: %s %%" % (((pd.isnull(full["SibSp"]).sum()) * 100)/1309))

#Family column: adding the Parch and SipSp column to a more simpler column.

full["Family"] = full["SibSp"] + full["Parch"] + 1

#Graph to compare the rate of survival

plt.figure(figsize=(8,6))

avg_fm = full.groupby(full["Family"]).mean()["Survived"]

sns.barplot(x=avg_fm.index, y=avg_fm.values)

#It seems that a family of 4 boasts the highest survival rate.

#To deal with the more fewer larger families, we will create a simplified,

#discretized family size variable.

#assign_size: function that divides the family into 3 groups

def assign_size(x):

    if x == 1:

        return 'singleton'

    elif (x < 5) & (x > 1):

        return 'small'

    elif (x > 4):

        return 'large'

#Re-create family column using the assign_size        

full["Family"] = full["Family"].apply(assign_size)

mosaic(full, ['Family', 'Survived'])

#set the values to numeric

full.set_value(full["Family"] == "singleton", "Family", 0)

full.set_value(full["Family"] == "small", "Family", 1)

full.set_value(full["Family"] == "large", "Family", 2)
#Machine Learning

#Define Training/Test sets.

train = full[:891]

test = full[891:1310]

#Define the predictor variables

predictors = ["Age", "Embarked", "Fare", "Pclass", "Sex", "Title", "Family"]

x_train = train[predictors]

y_train = train["Survived"]

x_test= test[predictors]
#Logistic Regression -- cross validation w/ cv=3

alg = LogisticRegression(random_state = 1)

scores = cross_validation.cross_val_score(alg, x_train, y_train, cv=3)

print(scores.mean())

#Random Forest -- cross validation w/ cv=3

alg_2 = RandomForestClassifier(random_state = 1, n_estimators = 150, min_samples_split = 4, min_samples_leaf = 2)

scores_2 = cross_validation.cross_val_score(alg_2, train[predictors], train["Survived"], cv=3)

print(scores_2.mean())
#The Random Forest tends to have a higher percentage than the Logistic model

alg_2.fit(x_train, y_train)

predictions = alg_2.predict(x_test)

submission = pd.DataFrame({'PassengerId': test.index, 'Survived': predictions})

submission.to_csv('titanic_submission.csv', index=False)