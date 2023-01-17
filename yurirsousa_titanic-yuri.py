#Algebra Linear

import numpy as np 



# Processamento/manipulação dos dados

import pandas as pd 



# Visualização dos dados

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



#Algoritimos Machine Learning

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB
test_df = pd.read_csv("../input/test.csv")

train_df = pd.read_csv("../input/train.csv")

train_df.info()
train_df.describe()
train_df.head(15)
total = train_df.isnull().sum().sort_values(ascending=False)

percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
train_df.columns.values

survived = 'survived'

not_survived = 'not survived'

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
FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette="muted",  order=None, hue_order=None )

FacetGrid.add_legend()
sns.barplot(x='Pclass', y='Survived', data=train_df)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

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
train_df = train_df.drop(['PassengerId'], axis=1)
import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

data = [train_df, test_df]



for dataset in data:

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int) 
# Agora podemos excluir a variavel cabine

train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)

data = [train_df, test_df]



for dataset in data:

    mean = train_df["Age"].mean()

    std = test_df["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    # computar números aleatórios entre a média, std e is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # preencher valores NaN na coluna Idade com valores aleatórios gerados

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_df["Age"].astype(int)
train_df["Age"].isnull().sum()
train_df['Embarked'].describe()
common_value = 'S'

data = [train_df, test_df]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
train_df.info()
data = [train_df, test_df]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
data = [train_df, test_df]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in data:

    # extraindo os titulos

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # substituir títulos com um título mais comum ou como Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # converter títulos em números

    dataset['Title'] = dataset['Title'].map(titles)

    # Inserindo o 0 nos campos NaN 

    dataset['Title'] = dataset['Title'].fillna(0)
train_df = train_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Name'], axis=1)
genders = {"male": 0, "female": 1}

data = [train_df, test_df]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)
train_df['Ticket'].describe()
train_df = train_df.drop(['Ticket'], axis=1)

test_df = test_df.drop(['Ticket'], axis=1)

ports = {"S": 0, "C": 1, "Q": 2}

data = [train_df, test_df]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)
data = [train_df, test_df]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
# Verificando a distribuição

train_df['Age'].value_counts()
train_df.head(10)
data = [train_df, test_df]



for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)
data = [train_df, test_df]

for dataset in data:

    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
for dataset in data:

    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)

    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
# Vamos dar uma última olhada no conjunto de treinamento, antes de começarmos a treinar os modelos.

train_df.head(20)
#Criando Modelo Machine Learn

X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()
# stochastic gradient descent (SGD) learning

#sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

#sgd.fit(X_train, Y_train)

#Y_pred = sgd.predict(X_test)

#

#sgd.score(X_train, Y_train)

#

#acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

#

#

#print(round(acc_sgd,2,), "%")
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")



# Logistic Regression

#logreg = LogisticRegression()

#logreg.fit(X_train, Y_train)

#

#Y_pred = logreg.predict(X_test)

#

#acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

#print(round(acc_log,2,), "%")

# KNN

#knn = KNeighborsClassifier(n_neighbors = 3)

#knn.fit(X_train, Y_train)

#

#Y_pred = knn.predict(X_test)

#

#acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

#print(round(acc_knn,2,), "%")
# Gaussian Naive Bayes

#gaussian = GaussianNB()

#gaussian.fit(X_train, Y_train)

#

#Y_pred = gaussian.predict(X_test)

#

#acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

#print(round(acc_gaussian,2,), "%")
# Perceptron

#perceptron = Perceptron(max_iter=5)

#perceptron.fit(X_train, Y_train)

#

#Y_pred = perceptron.predict(X_test)

#

#acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

#print(round(acc_perceptron,2,), "%")
# Linear SVC

#linear_svc = LinearSVC()

#linear_svc.fit(X_train, Y_train)

#

#Y_pred = linear_svc.predict(X_test)

#

#acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

#print(round(acc_linear_svc,2,), "%")
# Decision Tree

#decision_tree = DecisionTreeClassifier()

#decision_tree.fit(X_train, Y_train)



#Y_pred = decision_tree.predict(X_test)



#acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

#print(round(acc_decision_tree,2,), "%")
#results = pd.DataFrame({

#    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

#              'Random Forest', 'Naive Bayes', 'Perceptron', 

#              'Stochastic Gradient Decent', 

#              'Decision Tree'],

#    'Score': [acc_linear_svc, acc_knn, acc_log, 

#              acc_random_forest, acc_gaussian, acc_perceptron, 

#              acc_sgd, acc_decision_tree]})

#result_df = results.sort_values(by='Score', ascending=False)

#result_df = result_df.set_index('Score')

#result_df.head(9)




from sklearn.ensemble import RandomForestClassifier

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_prediction

    })

submission = submission[['PassengerId','Survived']]

submission.head()



submission.to_csv('Titanic-Predicao-1.csv', index=False)

print('Saved file: ' + 'Titanic-Predicao-1.csv')