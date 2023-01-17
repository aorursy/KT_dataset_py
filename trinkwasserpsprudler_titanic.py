import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualisation



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.shape
train_data.head()
# show columns and dtypes

train_data.info()
test_data.info()
# Column Description



# PassengerId -> id

# Survived -> if the passenger survived or not (0=not survived / 1=survived)

# Pclass -> The class which the passenger travelled [3,2,1]

# Name -> The name of the passenger

# Sex -> gender ['male', 'female']

# Age -> the age of the passenger

# SibSp -> Amount of Siblings/Spouses

# Parch -> Amount of family members on board (mother, father, daughter, son, stepdaughter, stepson)

# Ticket -> Ticket Number

# Fare -> Passenger Fare

# Cabine -> Cabine Number

# Embarked -> Port of Embarkation (Port where the passengers went on board) C = Cherbourg, Q = Queenstown, S = Southampton
import pandas_profiling

train_data.profile_report()
# Save the report in output

profile = train_data.profile_report(title='Pandas Profiling Report')

profile.to_file(output_file="Titanic data profiling.html")
# Show numerical description of the columns (-> object columsn are ignored)

train_data.describe()
# Show columsn with nan values

train_data.isna().sum()
test_data.isna().sum()
# Print correlation Matrix

corr = train_data.corr()

corr.style.background_gradient(cmap='coolwarm')
# drop embarked as it is not neccessary for prediction

train_data.drop(['Embarked'], axis=1,inplace=True)

test_data.drop(['Embarked'], axis=1,inplace=True)
# only for test_data, since there is a missing "Fare" values

test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)
# Man könnte die Werte hier auch normalisieren...



# convert from float to int

train_data['Fare'] = train_data['Fare'].astype(int)

test_data['Fare']    = test_data['Fare'].astype(int)
# get fare for survived & didn't survive passengers 

fare_not_survived = train_data["Fare"][train_data["Survived"] == 0]

fare_survived     = train_data["Fare"][train_data["Survived"] == 1]



# get average and std for fare of survived/not survived passengers

avgerage_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare      = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])



# plot

train_data['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))



avgerage_fare.index.names = std_fare.index.names = ["Survived"]

avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Original Age values - Titanic')

axis2.set_title('New Age values - Titanic')



# get average, std, and number of NaN values in titanic_df

average_age_titanic   = train_data["Age"].mean()

std_age_titanic       = train_data["Age"].std()

count_nan_age_titanic = train_data["Age"].isnull().sum()



# get average, std, and number of NaN values in test_df

average_age_test   = test_data["Age"].mean()

std_age_test       = test_data["Age"].std()

count_nan_age_test = test_data["Age"].isnull().sum()



# generate random numbers between (mean - std) & (mean + std)

rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)



# plot original Age values

# NOTE: drop all null values, and convert to int

train_data['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# test_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)



# fill NaN values in Age column with random values generated

train_data["Age"][np.isnan(train_data["Age"])] = rand_1

test_data["Age"][np.isnan(test_data["Age"])] = rand_2



# convert from float to int

train_data['Age'] = train_data['Age'].astype(int)

test_data['Age']    = test_data['Age'].astype(int)

        

# plot new Age Values

train_data['Age'].hist(bins=70, ax=axis2)

# test_df['Age'].hist(bins=70, ax=axis4)
# .... continue with plot Age column



# peaks for survived/not survived passengers by their age

facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train_data['Age'].max()))

facet.add_legend()



# average survived passengers by age

fig, axis1 = plt.subplots(1,1,figsize=(18,4))

average_age = train_data[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=average_age)
# It has a lot of NaN values, so it won't cause a remarkable impact on prediction

train_data.drop("Cabin",axis=1,inplace=True)

test_data.drop("Cabin",axis=1,inplace=True)
# Instead of having two columns Parch & SibSp, 

# we can have only one column represent if the passenger had any family member aboard or not,

# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.



# combine the values in train_data

train_data['Family'] =  train_data["Parch"] + train_data["SibSp"]

train_data['Family'].loc[train_data['Family'] > 0] = 1

train_data['Family'].loc[train_data['Family'] == 0] = 0



# combine the values in test_data

test_data['Family'] =  test_data["Parch"] + test_data["SibSp"]

test_data['Family'].loc[test_data['Family'] > 0] = 1

test_data['Family'].loc[test_data['Family'] == 0] = 0



# drop Parch & SibSp

train_data = train_data.drop(['SibSp','Parch'], axis=1)

test_data    = test_data.drop(['SibSp','Parch'], axis=1)



# plot

fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))



# sns.factorplot('Family',data=titanic_df,kind='count',ax=axis1)

sns.countplot(x='Family', data=train_data, order=[1,0], ax=axis1)



# average of survived for those who had/didn't have any family member

family_perc = train_data[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)



axis1.set_xticklabels(["With Family","Alone"], rotation=0)
# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.

# So, we can classify passengers as males, females, and child



def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex

    

train_data['Person'] = train_data[['Age','Sex']].apply(get_person,axis=1)

test_data['Person']    = test_data[['Age','Sex']].apply(get_person,axis=1)



# No need to use Sex column since we created Person column

train_data.drop(['Sex'],axis=1,inplace=True)

test_data.drop(['Sex'],axis=1,inplace=True)



# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers

person_dummies_titanic  = pd.get_dummies(train_data['Person'])

person_dummies_titanic.columns = ['Child','Female','Male']

person_dummies_titanic.drop(['Male'], axis=1, inplace=True)



person_dummies_test  = pd.get_dummies(test_data['Person'])

person_dummies_test.columns = ['Child','Female','Male']

person_dummies_test.drop(['Male'], axis=1, inplace=True)



train_data = train_data.join(person_dummies_titanic)

test_data    = test_data.join(person_dummies_test)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)

sns.countplot(x='Person', data=train_data, ax=axis1)



# average of survived for each Person(male, female, or child)

person_perc = train_data[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])



train_data.drop(['Person'],axis=1,inplace=True)

test_data.drop(['Person'],axis=1,inplace=True)
# sns.factorplot('Pclass',data=titanic_df,kind='count',order=[1,2,3])

sns.factorplot('Pclass','Survived',order=[1,2,3], data=train_data,size=5)



# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers

pclass_dummies_titanic  = pd.get_dummies(train_data['Pclass'])

pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)



pclass_dummies_test  = pd.get_dummies(test_data['Pclass'])

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)



train_data.drop(['Pclass'],axis=1,inplace=True)

test_data.drop(['Pclass'],axis=1,inplace=True)



train_data = train_data.join(pclass_dummies_titanic)

test_data    = test_data.join(pclass_dummies_test)
# define training and testing sets



X_train = train_data.drop("Survived",axis=1)

Y_train = train_data["Survived"]

X_test  = test_data.drop("PassengerId",axis=1).copy()



# from sklearn.model_selection import train_test_split

# # if we set the random state to a fix value we alwas get the same result in splitting

# train_set, test_set = train_test_split(train_data_selected, test_size=0.2, random_state=42)
# Random Forests

from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)