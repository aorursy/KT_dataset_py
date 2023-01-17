import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from graphviz import Source

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.multioutput import MultiOutputRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.tree import export_graphviz

from sklearn.covariance import EllipticEnvelope

from sklearn.naive_bayes import GaussianNB
orig = pd.read_csv("../input/train.csv")

orig.describe()
gender_submission = pd.read_csv("../input/gender_submission.csv")

#sub = pd.read_csv("submit.csv")

test = pd.read_csv("../input/test.csv")

test = test.assign(Survived=gender_submission.Survived)

#orig = orig.append(test,sort=True)

orig.isna().sum()
orig.corr()
df = orig.replace("female", 1)

df = df.replace("male", 0)

df = df.replace("S", 0)

df = df.replace("C", 1)

df = df.replace("Q", 2)

df = df.drop_duplicates(keep='first')

del df['Cabin']

del df['Ticket']

del df['Name']

#del df['PassengerId']

#df.Age.fillna(df.Age.mean(), inplace=True)

#df.isna().sum()

#df = df.dropna()

df.Fare.fillna(df.Fare.mode()[0], inplace=True)

df.Embarked.fillna(df.Embarked.mode()[0], inplace=True)

notnans = df.notnull().all(axis=1)

df_notnans = df[notnans]

#df_notnans.head()

X=df_notnans[['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]

y=df_notnans[['Age']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=4)

regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=30, random_state=0))

regr_multirf.fit(X_train, y_train)

score = regr_multirf.score(X_test, y_test)

df_nans = df.loc[~notnans].copy()

df_nans['Age'] = regr_multirf.predict(df_nans[['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']])

df_nans.head()
#df.Age.fillna(df.Age.mean(), inplace=True)

df =df.dropna()

df = df.append(df_nans,'sort=True')

df_dirty = df.copy()

df.corr()

#df_nans.tail()

#df.isna().sum()
filterAge = df.Age <= 15

column_name = 'Age'

df.loc[filterAge,column_name] = 0

filterAge = df.Age >= 40

df.loc[filterAge,column_name] = 2

filterAge = df.Age > 15

df.loc[filterAge,column_name] = 1



filterFare = df.Fare <= 40 #170.7764

column_name = 'Fare'

df.loc[filterFare,column_name] = 0

filterFare = df.Fare >= 341.5528

df.loc[filterFare,column_name] = 3

filterFare = df.Fare >= 170.7764

df.loc[filterFare,column_name] = 2

filterFare = df.Fare > 40

df.loc[filterFare,column_name] = 1

df.corr()
all_colls = ['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

clf = EllipticEnvelope(contamination=0.3, support_fraction=0.9)



df = df[clf.fit_predict(df[all_colls])==1]

df.corr()
df.describe()
fig, axs = plt.subplots(ncols=3, figsize=(20, 5))

df.groupby("Sex")["Survived"].value_counts().plot.bar(ax=axs[0])

df.groupby("Pclass")["Survived"].value_counts().plot.bar(ax=axs[1])

df.groupby("Embarked")["Survived"].value_counts().plot.bar(ax=axs[2])

axs[0].set_title('sex vs survived')

axs[1].set_title('pclass vs survived')

axs[2].set_title('embarked vs survived')

plt.show()
fig, ax = plt.subplots(len(all_colls), 2, figsize=(22,30))



for i, col_val in enumerate(all_colls):



    sns.distplot(df[col_val], hist=True, ax=ax[i][0])

    ax[i][0].set_xlabel(col_val)

    ax[i][0].set_ylabel('Count')

    sns.distplot(df_dirty[col_val], hist=True, ax=ax[i][1])

    ax[i][1].set_xlabel(col_val)

    ax[i][1].set_ylabel('Count')



plt.show()
variables = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

target = 'Survived'



fig, ax = plt.subplots(len(variables), 2, figsize=(22,26))

for i, col_val in enumerate(variables):



    sns.regplot(x=col_val,y=target, data = df, ax=ax[i][0])

    sns.regplot(x=col_val,y=target, data = df_dirty, ax=ax[i][1])



plt.show()
best_vars = ['Pclass', 'Sex', 'Fare', 'Age', 'Parch']

#best_vars = ['Pclass', 'Sex', 'Fare', 'Embarked', 'Age', 'Parch']

X=df[best_vars]

y=df[target]                                       

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=10)

randomForest = RandomForestClassifier(n_estimators=1000, criterion='entropy',random_state=None)

randomForest.fit(X_train, y_train)

predictions = randomForest.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Random test accuracy: {0:.2f} %".format(accuracy*100)) 
def run_kfold(clf,data, data2):

    kf = KFold(n_splits=5)

    kf.get_n_splits(data)

    outcomes = []

    fold = 0

    for train_index, test_index in kf.split(data):

        fold += 1

        X_train, X_test = data.values[train_index], data.values[test_index]

        y_train, y_test = data2.values[train_index], data2.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1:.2f} %".format(fold, accuracy*100))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0:.2f} %".format(mean_outcome*100)) 



run_kfold(randomForest,X,y)
X=df[best_vars]

y=df[target]                                       

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=10)

cart = DecisionTreeClassifier(criterion='gini',random_state=None)

cart.fit(X_train, y_train)

predictions = cart.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Random test accuracy: {0:.2f} %".format(accuracy*100))
Source( export_graphviz(cart, out_file=None, feature_names=X.columns, rounded=True, filled=True, max_depth=3))
run_kfold(cart,X,y)
X=df[best_vars]

y=df[target]                                       

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=10)

naive = GaussianNB()

naive.fit(X_train, y_train)

predictions = naive.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Random test accuracy: {0:.2f} %".format(accuracy*100))
run_kfold(naive,X,y)
df2 = test.replace("female", 1)

df2 = df2.replace("male", 0)

df2 = df2.replace("S", 0)

df2 = df2.replace("C", 1)

df2 = df2.replace("Q", 2)

df2 = df2.drop_duplicates(keep='first')

del df2['Cabin']

del df2['Ticket']

del df2['Name']

#del df2['PassengerId']

df2.Fare.fillna(0, inplace=True)

df2.Age.fillna(0, inplace=True)

df2.describe()

#df2.isna().sum()
X_train=df[best_vars]

y_train=df[target]

X_train2=df_dirty[best_vars]

y_train2=df_dirty[target]

X_test=df2[best_vars]

y_test=df2[target]

rndm = RandomForestClassifier(n_estimators=1000, criterion='entropy',random_state=None)

rndm.fit(X_train, y_train)

predictions = rndm.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

rndm2 = RandomForestClassifier(n_estimators=1000, criterion='entropy',random_state=None)

rndm2.fit(X_train2, y_train2)

predictions2 = rndm2.predict(X_test)

accuracy2 = accuracy_score(y_test, predictions2)

print("Random test accuracy with clean model: {0:.2f} %".format(accuracy*100))

print("Random test accuracy without clean model: {0:.2f} %".format(accuracy2*100))

submission = pd.DataFrame({

        "PassengerId": df2["PassengerId"],

        "Survived": predictions

    })

submission.to_csv('submit.csv', index=False)
crt = DecisionTreeClassifier(criterion='gini',random_state=None)

crt.fit(X_train, y_train)

predictions = crt.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

crt2 = DecisionTreeClassifier(criterion='gini',random_state=None)

crt2.fit(X_train2, y_train2)

predictions2 = crt2.predict(X_test)

accuracy2 = accuracy_score(y_test, predictions2)

print("Random test accuracy with clean model: {0:.2f} %".format(accuracy*100))

print("Random test accuracy without clean model: {0:.2f} %".format(accuracy2*100)) 

#submission = pd.DataFrame({

#        "PassengerId": df2["PassengerId"],

#        "Survived": predictions

#    })

#submission.to_csv('submission_cart.csv', index=False)
gnb = GaussianNB()

gnb.fit(X_train, y_train)

predictions = gnb.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

gnb2 = GaussianNB()

gnb2.fit(X_train2, y_train2)

predictions2 = gnb2.predict(X_test)

accuracy2 = accuracy_score(y_test, predictions2)

print("Random test accuracy with clean model: {0:.2f} %".format(accuracy*100))

print("Random test accuracy without clean model: {0:.2f} %".format(accuracy2*100)) 

#submission = pd.DataFrame({

#        "PassengerId": df2["PassengerId"],

#        "Survived": predictions

#    })

#submission.to_csv('submission_naive.csv', index=False)