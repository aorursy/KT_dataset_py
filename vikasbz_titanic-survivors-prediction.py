import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn import tree, preprocessing

#%matplotlib inline
#sbn.set()
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.describe()
train.info()
test.head()
test.describe()
test.info()
Survived = train['Survived']
Survived.is_copy = False

train.drop('Survived', axis=1, inplace=True)
print('All Good!')
data = pd.concat([train, test])

data.describe()
data.info()
print(data['Age'].isnull().sum())

round( sum(pd.isnull(data['Age'])) / len(data['PassengerId']) * 100, 2)
ax = data["Age"].hist(color='teal', alpha=0.6)
ax.set(xlabel='Age', ylabel='Count')
plt.show()
print(data['Cabin'].isnull().sum())

round(sum(pd.isnull(data['Cabin'])) / len(data['PassengerId']) * 100, 2)
print(data['Embarked'].isnull().sum())

round(sum(pd.isnull(data['Embarked'])) / len(data['PassengerId']) * 100, 2)
sbn.countplot(x='Embarked', data=data, palette='Set2')
plt.show()
data['Age'].fillna(data['Age'].median(skipna=True), inplace=True)

data['Fare'].fillna(data['Fare'].mean(skipna=True), inplace=True)

data.drop('Cabin', axis=1, inplace=True)

data['Embarked'].fillna('S', inplace=True)

print('All Good!')
data.info()
data.head()
data.drop('Name', axis=1, inplace=True)

data.drop('Ticket', axis=1, inplace=True)

print('All Good!')
data['Family'] = data['SibSp'] + data['Parch']

data['Alone'] = np.where(data['Family'] > 0, 0, 1)

print('All Good!')
data.drop('SibSp', axis=1, inplace=True)

data.drop('Parch', axis=1, inplace=True)

data.drop('Family', axis=1, inplace=True)

print('All Good!')
data.head()
print(data['Pclass'].unique())
print(data['Sex'].unique())
print(data['Embarked'].unique())
#Convert Sex into categorical variable:
data = pd.get_dummies(data, columns=['Pclass', 'Sex', 'Embarked'])

data.head()
data.drop('Sex_male', axis=1, inplace=True)

print('All Good!')
data.head()
PassengerId = data['PassengerId']

data.drop('PassengerId', axis=1, inplace=True)

print('All Good!')
data['Pclass_1'] = data.Pclass_1.apply(lambda x: int(x))
data['Pclass_2'] = data.Pclass_2.apply(lambda x: int(x))
data['Pclass_3'] = data.Pclass_3.apply(lambda x: int(x))
data['Sex_female'] = data.Sex_female.apply(lambda x: int(x))
data['Embarked_C'] = data.Embarked_C.apply(lambda x: int(x))
data['Embarked_Q'] = data.Embarked_Q.apply(lambda x: int(x))
data['Embarked_S'] = data.Embarked_S.apply(lambda x: int(x))

data.head()

data.info()
print('Original datasets')
print(len(train['PassengerId'])) #Original training set
print(len(test['PassengerId'])) #Original testing set

#Add PassengerId back to the dataset
data['PassengerId'] = PassengerId 

train_df = data.iloc[:891] #891 data rows to training set
test_df = data.iloc[891:] #Remaining data rows to the testing set.

train_df.is_copy = False
test_df.is_copy = False

#Verify
print('\nFinal datasets')
print(len(train_df['PassengerId']))
print(len(test_df['PassengerId']))
train_df['Survived'] = Survived

train_df.head()
sbn.countplot(x='Survived', data=train_df)
test_df['Survived'] = 0

no_survivors = test_df[['PassengerId', 'Survived']]
no_survivors.is_copy = False

no_survivors.head()
no_survivors.to_csv("no_survivors.csv", index=False)
sbn.factorplot(x='Survived', col='Sex_female', kind='count', palette='Set1', data=train_df);
print(train_df.groupby(by=['Sex_female']).Survived.sum())
print(train_df.groupby(by=['Sex_female']).Survived.count())

train_df.groupby(by=['Sex_female']).Survived.sum()/train_df.groupby(by=['Sex_female']).Survived.count()
test_df.drop('Survived', axis=1, inplace=True)

test_df['Survived'] = test_df['Sex_female'] == 1

test_df['Survived'] = test_df.Survived.apply(lambda x: int(x))
test_df.head()
female_survivors = test_df[['PassengerId', 'Survived']]
female_survivors.is_copy = False

female_survivors.head()
female_survivors.to_csv("female_survivors.csv", index=False)
Train_survived = train_df['Survived']
Train_survived.is_copy = False

train_df.drop('Survived', axis=1, inplace=True)

print('All Good!')
X = train_df.values
y = Train_survived.values

test_df.drop('Survived', axis=1, inplace=True)
test = test_df.values

print('All Good!')
#From sklearn:
dtc5 = tree.DecisionTreeClassifier(max_depth=5)
dtc5.fit(X, y)
Y_pred5 = dtc5.predict(test)

print('All Good!')
test_df['Survived'] = Y_pred5

dtc5_survivors = test_df[['PassengerId', 'Survived']]
dtc5_survivors.is_copy = False

dtc5_survivors.to_csv("dtc5_survivors.csv", index=False)
dtc3 = tree.DecisionTreeClassifier(max_depth=3)
dtc3.fit(X, y)

Y_pred3 = dtc3.predict(test)

print('All Good!')
test_df.drop('Survived', axis=1, inplace=True)

test_df['Survived'] = Y_pred3

dtc3_survivors = test_df[['PassengerId', 'Survived']]
dtc3_survivors.is_copy = False

dtc3_survivors.to_csv("dtc3_survivors.csv", index=False)