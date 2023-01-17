import numpy as np 

import matplotlib.pyplot as plt

import matplotlib.style as style

import pandas as pd

import seaborn as sns
from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.describe()
train_df.info()
data = [train_df, test_df]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)

    dataset['Age'] = dataset['Age'].fillna(0)

    dataset['Age'] = dataset['Age'].astype(int)
columns = train_df.columns

percent_missing = train_df.isnull().sum() * 100 / len(train_df)

missing_value_df = pd.DataFrame({'column_name': columns,

                                 'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing', ascending=False)
survived = 'Survived'

not_survived = 'Not Survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train_df[train_df['Sex']=='female']

men = train_df[train_df['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
FacetGrid = sns.FacetGrid(train_df, row='Embarked', height=4.5, aspect=1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )

FacetGrid.add_legend()
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
data = [train_df, test_df]

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0

    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1

    dataset['not_alone'] = dataset['not_alone'].astype(int)

  

train_df['not_alone'].value_counts()
axes = sns.factorplot('relatives','Survived', 

                      data=train_df, aspect = 2.5, )
import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8, "T":9}

data = [train_df, test_df]



for dataset in data:

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    #dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int)



# we can now drop the cabin feature

train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)
data = [train_df, test_df]



for dataset in data:

    mean = train_df["Age"].mean()

    std = test_df["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_df["Age"].astype(int)

    

train_df["Age"].isnull().sum()
train_df['Embarked'].describe()
data = [train_df, test_df]

common_value ='S'



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
data = [train_df, test_df]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in data:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    

    #replace title with common title

    dataset['Title'] = dataset['Title'].replace('Ms','Miss')

    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')

    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

    dataset['Title'] = dataset['Title'].replace([ 'Don', 'Rev', 'Dr','Major', 'Lady', 'Sir','Col', 'Capt', 'Countess',

       'Jonkheer','Dona'],'Rare')

    

    #convert title into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    

    #fill blank title with 0

    dataset['Title'] = dataset['Title'].fillna(0)

    

# we can now drop the cabin feature

train_df = train_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Name'], axis=1)  
data = [train_df, test_df]

genders = {"female": 1, "male": 0}



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)
train_df['Ticket'].describe()
data = [train_df, test_df]

train_df = train_df.drop(['Ticket'], axis=1)

test_df = test_df.drop(['Ticket'], axis=1)  
data = [train_df, test_df]

ports = {"S": 0, "C": 1, "Q":2}



for dataset in data:

    dataset['Embarked']= dataset['Embarked'].map(ports)
train_df['AgeGroup'] = pd.cut(train_df['Age'], 5)

train_df[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean().sort_values(by='AgeGroup', ascending=True)  
data = [train_df, test_df]



for dataset in data:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

    

#Drop AgeGroup feature

train_df = train_df.drop(['AgeGroup'], axis =1)
train_df['FareGroup'] = pd.qcut(train_df['Fare'], 6)

train_df[['FareGroup', 'Survived']].groupby(['FareGroup'], as_index=False).mean().sort_values(by='FareGroup', ascending=True) 

data = [train_df, test_df]



for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)

    

#Drop AgeGroup feature

train_df = train_df.drop(['FareGroup'], axis =1)
#Age times Pclass

data = [train_df, test_df]



for dataset in data:

    dataset['AgeClass'] = dataset['Age'] * dataset['Pclass']

    

#Fare per Person



for dataset in data:

    dataset['Fare_Per_Person']= dataset['Fare']/(dataset['relatives']+1)

    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)

    

train_df.head()   
X_train = train_df.drop(["Survived","PassengerId"], axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)



sgd.score(X_train, Y_train)



acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(X_train, Y_train)  



Y_pred = knn.predict(X_test)  



acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
gaussian = GaussianNB() 

gaussian.fit(X_train, Y_train)  



Y_pred = gaussian.predict(X_test)  



acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
perceptron = Perceptron(max_iter=5)

perceptron.fit(X_train, Y_train)



Y_pred = perceptron.predict(X_test)



acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)



Y_pred = linear_svc.predict(X_test)



acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, Y_train)  



Y_pred = decision_tree.predict(X_test)  



acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
Result_by_Model = pd.DataFrame({'Score': [acc_linear_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_decision_tree],'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression','Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent','Decision Tree']})

Result_by_Model = Result_by_Model.sort_values(by='Score', ascending=False)

Result_by_Model = Result_by_Model.set_index('Score')

Result_by_Model.head(10)
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.head(15)
importances.plot.bar()
train_df  = train_df.drop(["not_alone","Parch"], axis=1)

test_df  = test_df.drop(["not_alone","Parch"], axis=1)
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)



acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)

confusion_matrix(Y_train, predictions)
from sklearn.metrics import precision_score, recall_score



print("Precision:", precision_score(Y_train, predictions))

print("Recall:",recall_score(Y_train, predictions))
from sklearn.metrics import f1_score

f1_score(Y_train, predictions)