# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series,DataFrame

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Presentation

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')

# Any results you write to the current directory are saved as output.
# import original train data

train_orig_df = pd.read_csv("../input/train.csv")



#preview the imported train data

# preview 1 -- entry count

train_orig_df.index

# preview 1 -- data structure

train_orig_df.info()   

# preview 2 -- table view

train_orig_df.head(5)
# Clean dataframe by dropping columns that are not going to be used by analysis

# columns to drop: PassengerId, Name, Ticket

train_orig_df.drop(['PassengerId','Name', 'Ticket'], axis=1, inplace=True)
# Embarked

# plot the first figure : simple survival rate of each embarking location

# sanity check



none_embarked_count = train_orig_df["Embarked"].isnull().sum()

print(none_embarked_count, "passengers have no 'embarked' value")

#train_orig_df['Embarked'][train_orig_df['Embarked']==''] = 'S'

train_orig_df['Embarked'].fillna('S')

embark_dummies_titanic  = pd.get_dummies(train_orig_df['Embarked'])

embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

train_orig_df = train_orig_df.join(embark_dummies_titanic)



sns.factorplot('Embarked','Survived', data=train_orig_df)



# plot the second figure : a more comprehensive summary of each embarking location

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

sns.countplot( x='Embarked', data=train_orig_df, ax=axis1)

sns.countplot( x='Survived', data=train_orig_df, ax=axis2)

# group by embarked, and get the mean for survived passengers for each value in Embarked

embark_perc = train_orig_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

train_orig_df.drop("Embarked", axis=1, inplace=True)
# Fare



# First plot : a distribution of fare of all passengers

train_orig_df['Fare'].plot(kind='hist', bins=100, xlim=(0,100))



# Secodn plot : average fare for survived and unfortunate

fair_survive = train_orig_df['Fare'][train_orig_df['Survived']==1]

fair_not_survive = train_orig_df['Fare'][train_orig_df['Survived']==0]



fair_survive.mean()

fair_not_survive.mean()
# Family 

train_orig_df.head()

train_orig_df['Family'] =  train_orig_df["Parch"] + train_orig_df["SibSp"]

train_orig_df['Family'].loc[train_orig_df['Family'] > 0] = 1

train_orig_df['Family'].loc[train_orig_df['Family'] == 0] = 0

train_orig_df = train_orig_df.drop(['SibSp','Parch'], axis=1)



# Age



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,4))

axis1.set_title('Original Age values - Titanic')

axis2.set_title('New Age values - Titanic')



# Plot 1: A distribution of passenger ages

train_orig_df['Age'].hist(bins=70, ax=axis1)



# Ensure data sanity -- add 

avg_age = train_orig_df["Age"].mean()

std_age = train_orig_df["Age"].std()

none_age_count = train_orig_df["Age"].isnull().sum()

print(none_age_count, "passengers' ages are not available\n")



# generate random ages between (mean - std) & (mean + std) for those passengers

rand_1 = np.random.randint(avg_age - std_age, avg_age + std_age, size = none_age_count)

train_orig_df['Age'][np.isnan(train_orig_df['Age'])] = rand_1



#convert from float to int

train_orig_df['Age'] = train_orig_df['Age'].astype(int)



# Plot 2: A new distribution of passenger ages

train_orig_df['Age'].hist(bins=70, ax=axis2, color='green' )
# peaks for survived/not survived passengers by their age

facet = sns.FacetGrid(train_orig_df, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train_orig_df['Age'].max()))

facet.add_legend()



# average survived passengers by age

fig, axis1 = plt.subplots(1,1,figsize=(18,4))

average_age = train_orig_df[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=average_age)
# Cabin

# It has a lot of NaN values, so it won't cause a remarkable impact on prediction

train_orig_df.drop("Cabin",axis=1,inplace=True)

 

# Sex



# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.

# So, we can classify passengers as males, females, and child

def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex

    

train_orig_df['Person'] = train_orig_df[['Age','Sex']].apply(get_person,axis=1)



# No need to use Sex column since we created Person column

train_orig_df.drop(['Sex'],axis=1,inplace=True)



# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers

person_dummies_titanic  = pd.get_dummies(train_orig_df['Person'])

person_dummies_titanic.columns = ['Child','Female','Male']

person_dummies_titanic.drop(['Male'], axis=1, inplace=True)



train_orig_df = train_orig_df.join(person_dummies_titanic)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)

sns.countplot(x='Person', data=train_orig_df, ax=axis1)



# average of survived for each Person(male, female, or child)

person_perc = train_orig_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])



train_orig_df.drop(['Person'],axis=1,inplace=True)

# Pclass



# sns.factorplot('Pclass',data=titanic_df,kind='count',order=[1,2,3])

sns.factorplot('Pclass','Survived',order=[1,2,3], data=train_orig_df,size=5)



# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers

pclass_dummies_titanic  = pd.get_dummies(train_orig_df['Pclass'])

pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)



#pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])

#pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

#pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)



train_orig_df.drop(['Pclass'],axis=1,inplace=True)

#test_df.drop(['Pclass'],axis=1,inplace=True)



train_orig_df = train_orig_df.join(pclass_dummies_titanic)

#test_df    = test_df.join(pclass_dummies_test)
# Warmup done, let's do real predictions



# Prepare training set

x_train = train_orig_df.drop(["Survived"], axis=1)

#x_train = train_orig_df[["Family","Fare", "Child", "Female"]]

y_train = train_orig_df["Survived"]



#Logistic Regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()



logreg.fit(x_train, y_train)

#y_pred = logreg.predict(x_test)

logreg.score(x_train,y_train)

#x_train.head() 