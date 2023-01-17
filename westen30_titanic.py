# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.sort_values(by=['Fare'], ascending=False)

df_train['ticketcount'] = df_train.groupby(['Ticket'])['Name'].transform(len)

df_train['ticketcount'] = df_train['ticketcount'].fillna(1)

df_train['individualticket'] = df_train['Fare'] / df_train['ticketcount']



df_test.loc[df_test.isnull().Fare, 'Fare'] = 0

df_test['ticketcount'] = df_test.groupby(['Ticket'])['Name'].transform(len)

df_test['ticketcount'] = df_test['ticketcount'].fillna(1)

df_test['individualticket'] = df_test['Fare'] / df_test['ticketcount']
df_train.sort_values(by=['Fare'], ascending=False)
df_train.describe()
df_train.describe(include=['O'])

df_train.groupby(df_train.Ticket)



sns.distplot(df_train['individualticket'])
df_test.describe()
df_train['Pclass'] = df_train['Pclass'].astype('category')

df_test['Pclass'] = df_test['Pclass'].astype('category')
df_train = df_train.drop(['Ticket', 'Cabin', 'Fare', 'ticketcount'], axis=1)

df_test = df_test.drop(['Ticket', 'Cabin', 'Fare', 'ticketcount'], axis=1)
def setisalone(row):

    if row['SibSp'] == 0 and row['Parch'] == 0:

        return 1

    else:

        return 0

    

df_train['isalone'] = df_train.apply(setisalone, axis=1)

df_test['isalone'] = df_test.apply(setisalone, axis=1)

df_train['isalone'] = df_train['isalone'].astype('category')

df_test['isalone'] = df_test['isalone'].astype('category')
#df_train = pd.get_dummies(df_train, columns=['Sex'])

#df_test = pd.get_dummies(df_test, columns=['Sex'])

df_train['Sex'].replace(['male', 'female'], [1,0], inplace=True)

df_test['Sex'].replace(['male', 'female'], [1,0], inplace=True)
#df_train = pd.get_dummies(df_train, columns=['Embarked'], dummy_na=True)

#df_test = pd.get_dummies(df_test, columns=['Embarked'], dummy_na=True)

df_train['Embarked'] = df_train['Embarked'].fillna('S')

df_test['Embarked'] = df_test['Embarked'].fillna('S')

df_train['Embarked'].replace(['S', 'Q', 'C'], [0,1,2], inplace=True)

# df_train.Embarked = df_train.Embarked.astype(int)

df_test['Embarked'].replace(['S', 'Q', 'C'], [0,1,2], inplace=True)

df_train.info()
df_train.loc[df_train['Name'].str.contains("Master") & df_train.isnull().Age, 'Age'] = 10.0

df_train.loc[df_train['Name'].str.contains("Miss") & df_train.isnull().Age, 'Age'] = 10.0

df_train.loc[df_train.isnull().Age, 'Age'] = 35.0



df_test.loc[df_test['Name'].str.contains("Master") & df_test.isnull().Age, 'Age'] = 10.0

df_test.loc[df_test['Name'].str.contains("Miss") & df_test.isnull().Age, 'Age'] = 10.0

df_test.loc[df_test.isnull().Age, 'Age'] = 35.0
bins = [0, 16, 40, 60, 90]

group_names = [1, 2, 3, 4]

df_train['ageband'] = pd.cut(df_train['Age'], bins, labels=group_names)

df_test['ageband'] = pd.cut(df_test['Age'], bins, labels=group_names)


#bins = [-1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]

bins = [-1, 50, 100, 250, 600]

group_names = [1, 2, 3, 4]

df_train['fareband'] = pd.cut(df_train['individualticket'], bins, labels=group_names)

df_test['fareband'] = pd.cut(df_test['individualticket'], bins, labels=group_names)
df_train[df_train.isnull().fareband].head()

df_test[df_test.isnull().fareband].head()
df_train.info()

df_test.info()
df_train = df_train.drop(['individualticket', 'Age', 'Name', 'PassengerId', 'SibSp', 'Parch'], axis=1)

df_test = df_test.drop(['individualticket', 'Age', 'Name', 'SibSp', 'Parch'], axis=1)
df_train.head()

df_test.head()
X_train = df_train.drop("Survived", axis=1)

Y_train = df_train["Survived"]

X_test  = df_test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(df_train.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc


sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)