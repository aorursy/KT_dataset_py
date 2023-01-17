# Data analysis



import pandas as pd

from pandas import Series,DataFrame

import numpy as np

import random as rnd
# Data input



train_df = pd.read_csv("../input/train_titanic.csv")

test_df = pd.read_csv("../input/test_titanic.csv")

train_df.head()
train_df.shape
train_df.columns

train_df.describe()
train_df.info()
# Checking missing data



total = train_df.isnull().sum().sort_values(ascending=False)

percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
# Visualization



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
join_df = pd.concat([train_df, test_df])
join_df.drop(['Name','Ticket'], 1, inplace=True)
sns.countplot(x='Sex',data=join_df)
join_df["Embarked"] = join_df["Embarked"].fillna("S")

fig, (axis0) = plt.subplots(1,1,figsize=(6,6))



sns.countplot(x='Survived', hue="Sex", data=join_df, order=[1,0], ax=axis0)
join_df["Embarked"] = join_df["Embarked"].fillna("S")

fig, (axis1) = plt.subplots(1,1,figsize=(6,6))



sns.countplot(x='Embarked', data=join_df, ax=axis1)

join_df["Embarked"] = join_df["Embarked"].fillna("S")

fig, ( axis2) = plt.subplots(1,1,figsize=(6,6))



sns.countplot(x='Survived', hue="Embarked", data=join_df, order=[1,0], ax=axis2)
fig, (axis1) = plt.subplots(1,1,figsize=(6,6))

sns.boxplot(x='Pclass',y='Age',data=join_df, ax= axis1)
fig, (axis1) = plt.subplots(1,1,figsize=(6,6))

sns.swarmplot(x='Pclass',y='Age',data=join_df, ax = axis1)
FacetGrid = sns.FacetGrid(join_df, row='Embarked', size=4.5, aspect=1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )

FacetGrid.add_legend()
fig, (axis1) = plt.subplots(1,1,figsize=(6,6))



tc = join_df.corr()

sns.heatmap(tc,cmap='coolwarm', ax = axis1)

plt.title('Titanic')
# Dropping columns with least priority 



train_df.drop(['Name','Ticket','PassengerId', 'Fare', 'Cabin', 'Embarked'], 1, inplace=True)

test_df.drop(['PassengerId', 'Name','Ticket', 'Fare', 'Cabin', 'Embarked'], 1, inplace=True)

train_df.info()
test_df.info()
#Convert the Pclass and Sex to columns in pandas and drop them after conversion.



dummies_train = []

cols = ['Pclass', 'Sex']

for col in cols:

    dummies_train.append(pd.get_dummies(train_df[col]))
df_dummies = pd.concat(dummies_train, axis=1)
df_train_conc = pd.concat((train_df,df_dummies), axis=1)
df_train_conc.head(10)
dummies_test = []

cols = ['Pclass', 'Sex']

for col in cols:

    dummies_test.append(pd.get_dummies(test_df[col]))
df_dummies = pd.concat(dummies_test, axis=1)
df_test_conc = pd.concat((test_df,df_dummies), axis=1)
# Droping Sex and Pclass



df_test_conc = df_test_conc.drop(columns = {'Sex'})

df_test_conc = df_test_conc.drop(columns = {'Pclass'})

df_train_conc = df_train_conc.drop(columns = {'Sex'})

df_train_conc = df_train_conc.drop(columns = {'Pclass'})
df_train_conc['Age'] = df_train_conc['Age'].interpolate()

df_test_conc['Age'] = df_test_conc['Age'].interpolate()

df_train_conc.isna().sum()
df_test_conc.isna().sum()
# machine learning algorithms



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC, LinearSVC
X = df_train_conc.values

Y = df_train_conc['Survived'].values
X_train = df_train_conc.drop("Survived", axis=1)

Y_train = df_train_conc["Survived"]

X_test  = df_test_conc.copy()

X_train.shape, Y_train.shape, df_test_conc.shape
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



acc_log_reg = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log_reg
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


sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
xgboost = XGBClassifier()

xgboost.fit(X_train, Y_train)

y_pred = xgboost.predict(X_test)

acc_xgboost = round(xgboost.score(X_train, Y_train) * 100, 2)

acc_xgboost