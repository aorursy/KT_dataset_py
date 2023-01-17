import numpy as np

import pandas as pd

import re as re

import matplotlib.pyplot as mlt

%matplotlib inline

train = pd.read_csv('../input/train.csv',header=0,dtype={'Age':np.float64})

test = pd.read_csv('../input/test.csv',header=0,dtype={'Age':np.float64})

full_data = [train,test]

print(train.info())
for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print(train[['FamilySize','Survived']].groupby('FamilySize',as_index=False).mean())
for dataset in full_data:

    dataset['IsAlone'] = dataset['FamilySize'].apply(lambda x :1 if x ==1 else 0)
for dataset in full_data:

    dataset['Embarked']=dataset['Embarked'].fillna('S')
for dataset in full_data:

    dataset['Fare']=dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'],4)
for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = pd.isnull(dataset['Age']).count()

    dataset.loc[dataset['Age'].isnull(),'Age']=np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge']=pd.cut(train['Age'],5)
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.',name)

    if title_search :

        return title_search.group(1)

    return ""
for dataset in full_data:

    dataset['Title']=dataset['Name'].apply(get_title)
for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer',

                                                'Lady','Major','Rev','Sir'],'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
train.head()
for dataset in full_data:

    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1})
for dataset in full_data:

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)
for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
for dataset in full_data:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
for dataset in full_data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',\

                 'Parch', 'FamilySize']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)



test  = test.drop(drop_elements, axis = 1)



print (train.head(10))



train = train.values

test  = test.values
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression
classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

	AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



log_cols = ["Classifier", "Accuracy"]

log 	 = pd.DataFrame(columns=log_cols)
log
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



X = train[0::, 1::]

y = train[0::, 0]



acc_dict = {}



for train_index, test_index in sss.split(X, y):

	X_train, X_test = X[train_index], X[test_index]

	y_train, y_test = y[train_index], y[test_index]

	

	for clf in classifiers:

		name = clf.__class__.__name__

		clf.fit(X_train, y_train)

		train_predictions = clf.predict(X_test)

		acc = accuracy_score(y_test, train_predictions)

		if name in acc_dict:

			acc_dict[name] += acc

		else:

			acc_dict[name] = acc



for clf in acc_dict:

	acc_dict[clf] = acc_dict[clf] / 10.0

	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

	log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
candidate_classifier = SVC()

candidate_classifier.fit(train[0::, 1::], train[0::, 0])

result = candidate_classifier.predict(test)