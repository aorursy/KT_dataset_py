# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd;
import matplotlib;
import matplotlib.pyplot as plt
import numpy as np;
# Loading Data
train = pd.read_csv("../input/train.csv");
test = pd.read_csv("../input/test.csv");
gender_data = pd.read_csv("../input/gender_submission.csv");
# No.of rows in each
print(train.shape);
print(test.shape);
print(gender_data.shape);
print(train.columns);
print(test.columns);
print(gender_data.columns);
print(train.info());
print(test.info());
print(gender_data.info());

print(train.head());


print(test.head());
print(gender_data.head());
print(set(train.Survived));
print(set(gender_data.Survived));

# Survived has the value of 1. Else 0.
print(set(train.Pclass));
print(set(test.Pclass));

#Socio-economic status. Values - 1 (upper), 2 (middle), 3 (lower)
print(set(train.Sex));
print(set(test.Sex));
print(set(train.SibSp));
print(set(test.SibSp));

# No of siblings + spouse aboard
print(set(train.Parch));
print(set(test.Parch));

# No of parents / children aboard
print(set(train.Embarked));
print(set(test.Embarked));

#Port of embarkation - values  = C = Cherbourg, Q = Queenstown, S = Southampton
print(train.shape[0])

print(train.isna().sum());

print('*'*100);

print(test.shape[0])

print(test.isna().sum());

# In training data, only 2 rows have null Embarked values.
# Replacing Na with mode.

print(train.Embarked.value_counts());

# 'S' is most occuring value
# Replacing Na with 'S' in Embarked.

train["Embarked"] = train["Embarked"].fillna('S');
# Test data has 1 row where fare is null.
#Replacing the null value with median.
test["Fare"].fillna(test["Fare"].median(), inplace=True)
# Cabin has lot of Na Values. Therefore removing that column.
train.drop("Cabin", inplace=True, axis=1);
test.drop("Cabin", inplace=True, axis=1);
# Age has many Na values.
print(train.Age.mean());
print(train.Age.median());

print(test.Age.mean());
print(test.Age.median());
#Extracting title to fill age.
print(train.Name.head(20));
train["Title"] = train["Name"].str.extract(" ([a-zA-Z]+)\.");

test["Title"] = test["Name"].str.extract(" ([a-zA-Z]+)\.");
print(set(train["Title"]));
print(set(test["Title"]));
print(train["Title"].value_counts());
print(test["Title"].value_counts());
train.Title[train.Title == "Ms"] = "Miss";
train.Title[train.Title == "Mlle"] = "Miss";
train.Title[train.Title == "Mme"] = "Mrs";

test.Title[test.Title == "Ms"] = "Miss";
test.Title[test.Title == "Mlle"] = "Miss";
test.Title[test.Title == "Mme"] = "Mrs";


train.loc[train['Title'].isin(["Dr", "Don", "Dona", "Rev", "Major", "Col", "Countess", "Sir", "Jonkheer", "Lady", "Capt"]), 'Title'] = "Rare";
test.loc[test['Title'].isin(["Dr", "Don", "Dona", "Rev", "Major", "Col", "Countess", "Sir", "Jonkheer", "Lady", "Capt"]), 'Title'] = "Rare";
print(set(train.Title));
print(set(test.Title));
print(train.groupby(['Title']).Age.mean());
print(train.groupby(['Title']).Age.median());

print(train.Age.mean());
print(train.Age.median());

print('-'*100);

print(test.groupby(['Title']).Age.mean());
print(test.groupby(['Title']).Age.median());

print(test.Age.mean());
print(test.Age.median());
#Replacing Na in Age with median according to the title
train.Age[train.Title == "Master"] = train.Age[train.Title == "Master"].fillna(train.Age[train.Title == "Master"].median());
train.Age[train.Title == "Mr"] = train.Age[train.Title == "Mr"].fillna(train.Age[train.Title == "Mr"].median())
train.Age[train.Title == "Miss"] = train.Age[train.Title == "Miss"].fillna(train.Age[train.Title == "Miss"].median())
train.Age[train.Title == "Mrs"] = train.Age[train.Title == "Mrs"].fillna(train.Age[train.Title == "Mrs"].median());
train.Age[train.Title == "Rare"] = train.Age[train.Title == "Rare"].fillna(train.Age[train.Title == "Rare"].median());

#Replacing Na in Age with median according to the title
test.Age[train.Title == "Master"] = test.Age[train.Title == "Master"].fillna(test.Age[test.Title == "Master"].median());
test.Age[train.Title == "Mr"] = test.Age[train.Title == "Mr"].fillna(test.Age[test.Title == "Mr"].median())
test.Age[train.Title == "Miss"] = test.Age[train.Title == "Miss"].fillna(test.Age[test.Title == "Miss"].median())
test.Age[train.Title == "Mrs"] = test.Age[train.Title == "Mrs"].fillna(test.Age[test.Title == "Mrs"].median());
test.Age[train.Title == "Rare"] = test.Age[train.Title == "Rare"].fillna(test.Age[test.Title == "Rare"].median());
print(train.shape[0])

print(train.isna().sum());

print('*'*100);

print(test.shape[0])

print(test.isna().sum());
#familyfize
train["familysize"] = train.SibSp + train.Parch + 1;
#alone
train["alone"] = train.apply(lambda x : 1 if x["SibSp"] + x["Parch"] == 0 else 0, axis=1)



#familyfize
test["familysize"] = test.SibSp + test.Parch + 1;
#alone
test["alone"] = test.apply(lambda x : 1 if x["SibSp"] + x["Parch"] == 0 else 0, axis=1)
train.familysize.hist();
print(train.familysize.value_counts());
test.familysize.hist();
print(test.familysize.value_counts());
print(train.familysize.describe());
print(test.familysize.describe());
train.familysizetemp=None;
train.loc[train.familysize.astype(int) <= 3,'familysizetemp'] = "small";
train.loc[(train.familysize.astype(int)) > 3 & (train.familysize.astype(int) <= 6),'familysizetemp'] = "medium";
train.loc[train.familysize.astype(int) > 6,'familysizetemp'] = "big";


test.familysizetemp=None;
test.loc[test.familysize.astype(int) <= 3,'familysizetemp'] = "small";
test.loc[(test.familysize.astype(int)) > 3 & (test.familysize.astype(int) <= 6),'familysizetemp'] = "medium";
test.loc[test.familysize.astype(int) > 6,'familysizetemp'] = "big";
train["familysize"] = train["familysizetemp"];
test["familysize"] = test["familysizetemp"];

del train["familysizetemp"];
del test["familysizetemp"];



print(train.familysize.value_counts());
train.familysize.value_counts().plot(kind="bar")



print(test.familysize.value_counts());
test.familysize.value_counts().plot(kind="bar")
#AgeGroup
print(set(train.Age));
print(train["Age"].describe());

#AgeGroup
print(set(test.Age));
print(test["Age"].describe());
train.Age.hist();

test.Age.hist();
train["AgeGroup"]=None;
train.loc[train.Age <= 10,"AgeGroup"] = '[0-10]';
train.loc[(train.Age > 10) & (train.Age <= 20),"AgeGroup"] = '(10-20]';
train.loc[(train.Age > 20) & (train.Age <= 30),"AgeGroup"] = '(20-30]';
train.loc[(train.Age > 30) & (train.Age <= 40),"AgeGroup"] = '(30-40]';
train.loc[(train.Age > 40),"AgeGroup"] = '>40';



test["AgeGroup"]=None;
test.loc[test.Age <= 10,"AgeGroup"] = '[0-10]';
test.loc[(test.Age > 10) & (test.Age <= 20),"AgeGroup"] = '(10-20]';
test.loc[(test.Age > 20) & (test.Age <= 30),"AgeGroup"] = '(20-30]';
test.loc[(test.Age > 30) & (test.Age <= 40),"AgeGroup"] = '(30-40]';
test.loc[(test.Age > 40),"AgeGroup"] = '>40';

train.AgeGroup.value_counts().plot(kind="bar");


test.AgeGroup.value_counts().plot(kind="bar")
print(train.columns);
print(test.columns);
train = train.drop("Ticket",axis=1);
test = test.drop("Ticket",axis=1);

train = train.drop("Title",axis=1);
test = test.drop("Title",axis=1);

train = train.drop("Name",axis=1);
test = test.drop("Name",axis=1);

train = train.drop("SibSp",axis=1);
test = test.drop("SibSp",axis=1);

train = train.drop("Parch",axis=1);
test = test.drop("Parch",axis=1);


print(train.info())
import seaborn as sns;
#sns.countplot(train["Survived"]);

# Bar graph

Survived = train.groupby('Survived').size();

print(Survived);

Survived.plot(kind="bar");
plt.xlabel("Survived");
plt.ylabel("Count");
plt.title("Survived 0 vs 1");
plt.show()
Survived_Perc = (train['Survived'].value_counts()  / len(train)) * 100

print(Survived_Perc);

Survived_Perc.plot.bar();


Survived_Perc.plot(kind="bar");
plt.xlabel("Survived");
plt.ylabel("Percentage");
plt.title("Survived 0 vs 1");
plt.show()
# Bar graph

Gender = train.groupby('Sex').size();

print(Gender);

Survived.plot(kind="bar");
plt.xlabel("Gender");
plt.ylabel("Count");
plt.title("Males vs Females");
plt.show()
Gender_Perc = (train['Sex'].value_counts()  / len(train)) * 100

print(Gender_Perc);

Gender_Perc.plot.bar();


Gender_Perc.plot(kind="bar");
plt.xlabel("Gender");
plt.ylabel("Percentage");
plt.title("Male vs Females");
plt.show()
# Bar graph

Pclass = train.groupby('Pclass').size();

print(Pclass);

Pclass.plot(kind="bar");
plt.xlabel("Class");
plt.ylabel("Count");
plt.title("Population distribution across different classes");
plt.show()
Pclass_Perc = (train['Pclass'].value_counts()  / len(train)) * 100

print(Pclass_Perc);

Pclass_Perc.plot.bar();


Pclass_Perc.plot(kind="bar");
plt.xlabel("class");
plt.ylabel("Percentage");
plt.title("Population distribution across different classes");
plt.show()
# Bar graph

agegrp = train.groupby('AgeGroup').size();

print(agegrp);

agegrp.plot(kind="bar");
plt.xlabel("Age Groups");
plt.ylabel("Count");
plt.title("Population distribution across different age groups");
plt.show()
agegrp_perc = (train['AgeGroup'].value_counts()  / len(train)) * 100

print(agegrp_perc);

agegrp_perc.plot.bar();


agegrp_perc.plot(kind="bar");
plt.xlabel("Age group");
plt.ylabel("Percentage");
plt.title("Population distribution across different Age groups");
plt.show()

alone = train.groupby('alone').size();

print(alone);

alone.plot(kind="bar");
plt.xlabel("Travelling Alone");
plt.ylabel("Count");
plt.title("Number of passengers travelling alone");
plt.show()

Age = train['Age'].value_counts().sort_index(ascending=True)

Age.plot.line()
Age.plot.area()

plt.xlabel("Age");
plt.title("Age Distribution");
plt.show()
plt.boxplot(train.Age);

plt.xlabel("Age");
plt.title("Age Distribution");
plt.show()


Survived_gender = train.groupby(['Survived','Sex']).size().unstack();

print(Survived_gender);

Survived_gender.plot(kind="bar");
plt.xlabel("Survived");
plt.ylabel("Count");
plt.title("Survival ratio across males and females");
plt.show()
Survived_class = train.groupby(['Survived','Pclass']).size().unstack();

print(Survived_class);

Survived_class.plot(kind="bar");
plt.xlabel("Survived");
plt.ylabel("Count");
plt.title("Survival ratio across diff classes");
plt.show()
Survived_agegrp = train.groupby(['Survived','AgeGroup']).size().unstack();

print(Survived_agegrp);

Survived_agegrp.plot(kind="bar");
plt.xlabel("Survived");
plt.ylabel("Count");
plt.title("Survival ratio across age groups");
plt.show()
Survived_alone = train.groupby(['Survived','alone']).size().unstack();

print(Survived_alone);

Survived_alone.plot(kind="bar");
plt.xlabel("Survived");
plt.ylabel("Count");
plt.title("Survival rate if alone on ship");
plt.show()
train_sex = pd.get_dummies(train['Sex']);
#print(train_sex);
train = pd.concat([train, train_sex], axis=1);


print(train[["Sex","female","male"]].head());
train = train.drop("Sex",axis=1);
test_sex = pd.get_dummies(test['Sex']);
#print(train_sex);
test = pd.concat([test, test_sex], axis=1);


print(test[["Sex","female","male"]].head());
test = test.drop("Sex",axis=1);
train_familysize =pd.get_dummies(train["familysize"]);
train = pd.concat([train, train_familysize], axis=1);


train.rename({"big":"familysize_big","medium":"familysize_medium","small":"familysize_small"},axis="columns", inplace=True)

train = train.drop("familysize",axis=1);
test_familysize =pd.get_dummies(test["familysize"]);
test = pd.concat([test, test_familysize], axis=1);


test.rename({"big":"familysize_big","medium":"familysize_medium","small":"familysize_small"},axis="columns", inplace=True)

test = test.drop("familysize",axis=1);
train_pclass =pd.get_dummies(train["Pclass"]);
train = pd.concat([train, train_pclass], axis=1);


train.rename({1:"Pclass_1", 2:"Pclass_2", 3:"Pclass_3"},axis="columns", inplace=True)
train = train.drop("Pclass",axis=1);
test_pclass =pd.get_dummies(test["Pclass"]);
test = pd.concat([test, test_pclass], axis=1);


test.rename({1:"Pclass_1", 2:"Pclass_2", 3:"Pclass_3"},axis="columns", inplace=True)
test = test.drop("Pclass",axis=1);
train_agegrp =pd.get_dummies(train["AgeGroup"]);
train = pd.concat([train, train_agegrp], axis=1);

train.rename({"[0-10]":"age-[0-10]", "[10-20]":"age-[10-20]", "(20-30]": "age-(20-30]", "(30-40]": "age-(30-40]", ">40": "age->40"},axis="columns", inplace=True)
train = train.drop("AgeGroup",axis=1);
test_agegrp =pd.get_dummies(test["AgeGroup"]);
test = pd.concat([test, test_agegrp], axis=1);

test.rename({"[0-10]":"age-[0-10]", "[10-20]":"age-[10-20]", "(20-30]": "age-(20-30]", "(30-40]": "age-(30-40]", ">40": "age->40"},axis="columns", inplace=True)
test = test.drop("AgeGroup",axis=1);

train_embark =pd.get_dummies(train["Embarked"]);
train = pd.concat([train, train_embark], axis=1);

train.rename({"C":"embarked_Cherbourg", "Q":"embarked_Queenstown", "S":"embarked_Southampton"},axis="columns", inplace=True)


train = train.drop("Embarked",axis=1);
test_embark =pd.get_dummies(test["Embarked"]);
test = pd.concat([test, test_embark], axis=1);

test.rename({"C":"embarked_Cherbourg", "Q":"embarked_Queenstown", "S":"embarked_Southampton"},axis="columns", inplace=True)

test = test.drop("Embarked",axis=1);
from sklearn import preprocessing;

train["Age"] = preprocessing.scale(train["Age"]);
train["Fare"] = preprocessing.scale(train["Fare"])
from sklearn import preprocessing;

test["Age"] = preprocessing.scale(test["Age"]);
test["Fare"] = preprocessing.scale(test["Fare"])



