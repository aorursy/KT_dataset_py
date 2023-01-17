import pandas as pd

from pandas import Series, DataFrame



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



from xgboost import XGBClassifier
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



train_df.head()
train_df.info()

test_df.info()
#Drop unnecessary columns

train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

test_df.drop(['Name', 'Ticket'], axis=1, inplace=True)
#Drop the Embarked column

train_df.drop(['Embarked'], axis=1, inplace=True)

test_df.drop(['Embarked'], axis=1, inplace=True)

test_df.head()
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
train_df['Fare'] = train_df['Fare'].astype(int)

test_df['Fare'] = test_df['Fare'].astype(int)



fare_not_survived = train_df['Fare'][train_df['Survived'] == 0]

fare_survived = train_df['Fare'][train_df['Survived'] == 1]



avg_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])



train_df['Fare'].plot(kind='hist', figsize=(12, 3), bins=100, xlim=(0, 50))
avg_fare.index.names = std_fare.index.names = ['Survived']

avg_fare.plot(yerr=std_fare, kind='bar', legend=False)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))

axis1.set_title('Original age values - Titanic')

axis2.set_title('New age values - Titanic')



avg_age_train = train_df['Age'].mean()

std_age_train = train_df['Age'].std()

count_nan_age_train = train_df['Age'].isnull().sum()



avg_age_test = test_df['Age'].mean()

std_age_test = test_df['Age'].std()

count_nan_age_test = test_df['Age'].isnull().sum()



rand_train = np.random.randint(avg_age_train - std_age_train, avg_age_train + std_age_train, size = count_nan_age_train)

rand_test = np.random.randint(avg_age_test - std_age_test, avg_age_test + std_age_test, size = count_nan_age_test)



train_df['Age'].dropna().astype(int).hist(bins = 70, ax = axis1)



print(rand_train)



train_df['Age'][np.isnan(train_df['Age'])] = rand_train

test_df['Age'][np.isnan(test_df['Age'])] = rand_test



train_df['Age'] = train_df['Age'].astype(int)

test_df['Age'] = test_df['Age'].astype(int)



train_df['Age'].hist(bins = 70, ax = axis2)
facet = sns.FacetGrid(train_df, hue='Survived', aspect=4)

facet.map(sns.kdeplot, 'Age', shade=True)

facet.set(xlim = (0, train_df['Age'].max()))

facet.add_legend()



fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))

avg_age = train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=avg_age)
train_df.drop(['Cabin'], axis=1, inplace=True)

test_df.drop(['Cabin'], axis=1, inplace=True)
train_df['Family'] = train_df['Parch'] + train_df['SibSp']

train_df['Family'].loc[train_df['Family'] > 0] = 1

train_df['Family'].loc[train_df['Family'] == 0] = 0



test_df['Family'] = test_df['Parch'] + test_df['SibSp']

test_df['Family'].loc[test_df['Family'] > 0] = 1

test_df['Family'].loc[test_df['Family'] == 0] = 0



train_df.drop(['Parch', 'SibSp'], axis=1, inplace=True)

test_df.drop(['Parch', 'SibSp'], axis=1, inplace=True)



fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)



sns.countplot(x='Family', data=train_df, order=[1, 0], ax=axis1)



family_perc = train_df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1, 0], ax=axis2)



axis1.set_xticklabels(['With Family', 'Alone'])
def get_person(passenger):

    age, sex = passenger

    return 'child' if age < 16 else sex



train_df.head()
train_df['Person'] = train_df[['Age', 'Sex']].apply(get_person, axis=1)

test_df['Person'] = test_df[['Age', 'Sex']].apply(get_person, axis=1)



train_df.drop(['Sex'], axis=1, inplace=True)

test_df.drop(['Sex'], axis=1, inplace=True)



train_person_dummies = pd.get_dummies(train_df['Person'])

train_person_dummies.columns = ['Child', 'Female', 'Male']

train_person_dummies.drop(['Male'], axis=1, inplace=True)



test_person_dummies = pd.get_dummies(test_df['Person'])

test_person_dummies.columns = ['Child', 'Female', 'Male']

test_person_dummies.drop(['Male'], axis=1, inplace=True)



train_df = train_df.join(train_person_dummies)

test_df = test_df.join(test_person_dummies)



fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 4))



sns.countplot(x='Person', data=train_df, ax=axis1)



person_perc = train_df[['Person', 'Survived']].groupby(['Person'], as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male', 'female', 'child'])



train_df.drop(['Person'], axis=1, inplace=True)

test_df.drop(['Person'], axis=1, inplace=True)
sns.factorplot('Pclass', 'Survived', data=train_df, order=[1, 2, 3], size = 5)



train_pclass_dummies = pd.get_dummies(train_df['Pclass'])

train_pclass_dummies.columns = ['Class_1', 'Class_2', 'Class_3']

train_pclass_dummies.drop(['Class_3'], axis=1, inplace=True)



test_pclass_dummies = pd.get_dummies(test_df['Pclass'])

test_pclass_dummies.columns = ['Class_1', 'Class_2', 'Class_3']

test_pclass_dummies.drop(['Class_3'], axis=1, inplace=True)



train_df.drop(['Pclass'], axis=1, inplace=True)

test_df.drop(['Pclass'], axis=1, inplace=True)



train_df = train_df.join(train_pclass_dummies)

test_df = test_df.join(test_pclass_dummies)
train_df.info()

test_df.info()

test_df.head()
X_train = train_df.drop('Survived', axis=1)

Y_train = train_df['Survived']

X_test = test_df.drop('PassengerId', axis=1).copy()
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

svc.score(X_train, Y_train)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

knn.score(X_train, Y_train)
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

gaussian.score(X_train, Y_train)
random_forest = RandomForestClassifier(n_estimators = 100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
xgboost = XGBClassifier()

xgboost.fit(X_train, Y_train)

Y_pred = xgboost.predict(X_test)

xgboost.score(X_train, Y_train)
coeff_df = DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df['Coefficient Estimate'] = pd.Series(logreg.coef_[0])



coeff_df
submission = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived': Y_pred

})

submission.to_csv('predictions.csv', index=False)