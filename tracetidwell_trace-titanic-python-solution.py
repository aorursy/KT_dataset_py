import matplotlib.pyplot as plt

%matplotlib inline

import random

import numpy as np

import pandas as pd

from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics

import sklearn.ensemble as ske

import tensorflow as tf

from tensorflow.contrib import skflow
titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test_df = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
titanic_df.head()
test_df.head()
titanic_df['Survived'].mean()
titanic_df.groupby('Pclass').mean()
class_sex_grouping = titanic_df.groupby(['Pclass', 'Sex']).mean()

print(class_sex_grouping['Survived'])
class_sex_grouping['Survived'].plot.bar()
group_by_age = pd.cut(titanic_df['Age'], np.arange(0, 90, 10))

age_grouping = titanic_df.groupby(group_by_age).mean()

age_grouping['Survived'].plot.bar()
titanic_df.count()
test_df.count()
titanic_df = titanic_df.drop(['Cabin'], axis = 1)
test_df = test_df.drop(['Cabin'], axis=1)
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
Title_list = pd.DataFrame(index = titanic_df.index, columns = ["Title"])

Surname_list = pd.DataFrame(index = titanic_df.index, columns = ["Surname"])

Name_list = list(titanic_df.Name)

NL_1 = [elem.split("\n") for elem in Name_list]

ctr = 0

for j in NL_1:

    FullName = j[0]

    FullName = FullName.split(",")

    Surname_list.loc[ctr,"Surname"] = FullName[0]

    FullName = FullName.pop(1)

    FullName = FullName.split(".")

    FullName = FullName.pop(0)

    FullName = FullName.replace(" ", "")

    Title_list.loc[ctr, "Title"] = str(FullName)

    ctr = ctr + 1

    

titanic_df["Title"] = Title_list
Title_list_test = pd.DataFrame(index = test_df.index, columns = ["Title"])

Surname_list_test = pd.DataFrame(index = test_df.index, columns = ["Surname"])

Name_list_test = list(test_df.Name)

NL_1 = [elem.split("\n") for elem in Name_list_test]

ctr = 0

for j in NL_1:

    FullName_test = j[0]

    FullName_test = FullName_test.split(",")

    Surname_list_test.loc[ctr,"Surname"] = FullName_test[0]

    FullName_test = FullName_test.pop(1)

    FullName_test = FullName_test.split(".")

    FullName_test = FullName_test.pop(0)

    FullName_test = FullName_test.replace(" ", "")

    Title_list_test.loc[ctr, "Title"] = str(FullName_test)

    ctr = ctr + 1

    

test_df["Title"] = Title_list_test
#titanic_df = titanic_df.dropna()
#test_df = test_df.dropna()
titanic_df.count()
test_df.count()
def preprocess_titanic_df(df) :

    processed_df = df.copy()

    le = preprocessing.LabelEncoder()

    processed_df.Sex = le.fit_transform(processed_df.Sex)

    processed_df.Embarked = le.fit_transform(processed_df.Embarked)

    processed_df = processed_df.drop(['Name', 'Ticket'], axis = 1)

    return processed_df
processed_df = preprocess_titanic_df(titanic_df)

processed_df.count()

processed_df
processed_test_df = preprocess_titanic_df(test_df)

processed_test_df.count()

processed_test_df
median_ages = np.zeros((2,3))

median_ages
median_ages_test = np.zeros((2,3))

median_ages_test
#for i in range(0, 2):

#    for j in range(0, 3):

#        median_ages[i,j] = processed_df[(processed_df['Sex'] == i) & (processed_df['Pclass'] == j+1)]['Age'].dropna().median()

#        median_ages_test[i,j] = processed_test_df[(processed_test_df['Sex'] == i) & (processed_test_df['Pclass'] == j+1)]['Age'].dropna().median()

        

#median_ages

#median_ages_test
titanic_list =  np.array(titanic_df.values)

for row in titanic_list :

    if np.isnan(row[5]) :

        if row[11] == 'Master' :

            row[5] = 12

        elif row[6] == 1:

            row[5] = 21

        else :

            row[5] = 12

            



titanic_list = pd.DataFrame(titanic_list)

processed_df['Age'] = titanic_list[5]

processed_df = processed_df.drop('Title', axis=1)

titanic_list_test = np.array(test_df.values)

for row in titanic_list_test :

    if np.isnan(row[5]) :

        if row[11] == 'Master' :

            row[5] = 12

        elif row[6] == 1:

            row[5] = 21

        else :

            row[5] = 12

            



titanic_list_test = pd.DataFrame(titanic_list_test)

processed_test_df['Age'] = titanic_list[5]

processed_test_df = processed_test_df.drop('Title', axis=1)
#for i in range(0, 2):

#    for j in range(0, 3):

#        processed_df.loc[ (processed_df.Age.isnull()) & (processed_df.Sex == i) & (processed_df.Pclass == j+1),'Age'] = median_ages[i,j]

#        processed_test_df.loc[ (processed_test_df.Age.isnull()) & (processed_test_df.Sex == i) & (processed_test_df.Pclass == j+1),'Age'] = median_ages_test[i,j]

        

processed_df.loc[processed_df.Fare.isnull(), 'Fare'] = processed_df['Fare'].median()

processed_test_df.loc[processed_test_df.Fare.isnull(), 'Fare'] = processed_test_df['Fare'].median()



#processed_df.count()

#processed_test_df.count()
X = processed_df.drop(['Survived'], axis = 1).values

Y = processed_df['Survived'].values

print(X)
X_test = processed_test_df.values
#x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
clf_dt = tree.DecisionTreeClassifier(max_depth=10)

clf_dt.fit(X, Y)

Y_test_dt = clf_dt.predict(X_test)

clf_dt.score(X, Y)
clf_rf = ske.RandomForestClassifier(n_estimators=50)

clf_rf.fit(X,Y)

Y_test_rf = clf_rf.predict(X_test)

clf_rf.score(X, Y)
clf_gb = ske.GradientBoostingClassifier(n_estimators=50)

clf_gb.fit(X,Y)

Y_test_gb = clf_gb.predict(X_test)

clf_gb.score(X, Y)

#test_classifier(clf_gb)
eclf = ske.VotingClassifier([('dt', clf_dt), ('rf', clf_rf), ('gb', clf_gb)])

eclf.fit(X,Y)

Y_test_eclf = eclf.predict(X_test)

eclf.score(X, Y)

#test_classifier(eclf)
#def custom_model(X, Y) :

#    layers = skflow.ops.dnn(X, [20, 40, 20], tf.tanh)

#    return skflow.models.logistic_regression(layers, Y)
#tf_clf_c = skflow.TensorFlowEstimator(model_fn=custom_model, n_classes=2, batch_size=256, steps=1000, learning_rate=0.05)

#tf_clf_c.fit(X,Y)

#Y_test = tf_clf_c.predict(X_test)

#metrics.accuracy_score(y_test, tf_clf_c.predict(x_test))
submission = pd.DataFrame({'PassengerId': processed_test_df['PassengerId'], 'Survived': Y_test_rf})

submission.to_csv('clf_titanic.csv', index=False)