#imports

import pandas as pd

import numpy as np

import graphviz 

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz



#load data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



#combine 

combined = pd.concat([train, test])

#convert Sex to numeric

dict1 = {

    'male': '1',

    'female': '0'

}

train['Sex'] = train['Sex'].map(dict1).astype(int)

test['Sex'] = test['Sex'].map(dict1).astype(int)



#comlete age based on title avarage

def title(row):

    return row['Name'].split(',')[1].split('.')[0]       

combined['title'] = combined.apply(title, axis = 1)

train['title'] = train.apply(title, axis = 1)

test['title'] = test.apply(title, axis = 1)



#find medians

dict2 = {}

for name, group in combined.groupby('title'):

    dict2[name] = group['Age'].median()

    

#complete missing ages

def complete(row):

    if pd.isnull(row['Age']):

        if row['title'] in dict2:

            return round(dict2[row['title']])

    else:

        return row['Age']

        

train['Age'] = train.apply(complete, axis = 1)

test['Age'] = test.apply(complete, axis = 1)



print(dict2)

train.head()
#DecisionTreeClassifier

X_train = train.drop(["Survived","PassengerId","Name","SibSp","Parch","Ticket","Fare","Cabin","Embarked","title"], axis=1)

Y_train = train["Survived"]

X_test  = test.drop(["PassengerId","Name","SibSp","Parch","Ticket","Fare","Cabin","Embarked","title"], axis=1).copy()



decision_tree = DecisionTreeClassifier(max_depth=4)

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

accuracy = round(decision_tree.score(X_train, Y_train) * 100, 2)



print(accuracy)

print(X_train.columns)

print(decision_tree.feature_importances_)

dot_data = export_graphviz(decision_tree, out_file=None) 

graph = graphviz.Source(dot_data)

graph 
#Split age to bands

def ageBands(row):

    if row['Age'] <= 13:

        return 1

    elif (row['Age'] > 13) & (row['Age'] <= 34):

        return 2

    else:

        return 3

train['Age_Band'] = train.apply(ageBands, axis = 1)

test['Age_Band'] = test.apply(ageBands, axis = 1)

#find intersection between ticket groups in training and test data

train_set = set(train['Ticket'])

test_set = set(test['Ticket'])

print(len(train_set.intersection(test_set)))



#group training data by 'Ticket' 

ticket_group = train.groupby('Ticket')

for name, group in ticket_group:    

    if len(group) > 1:

        print(name)

        print(group)

        

# Add column 'GroupSurvival' and assign the value 1 where the person was in a group with mostly survivning members,

# -1 mostly non survivning members, 0 otherwise  

ticket_group = combined.groupby('Ticket')

test['Survived'] = np.nan

def group_survive(row):

    ticket = ticket_group.get_group(row['Ticket'])

    count = 0

    if (pd.notnull(row['Survived'])): #exclude own survivial

        if row['Survived'] == 1:

            count -= 1

        elif row['Survived'] == 0:

            count += 1        

    for item in ticket['Survived']:

        if pd.notnull(item):

            if item == 1:

                count += 1

            elif item == 0:

                count -= 1

    if count > 0:

        return 1

    elif count < 0:

        return -1

    else:

        return 0

train['GroupSurvival'] = train.apply(group_survive, axis = 1)

test['GroupSurvival'] = test.apply(group_survive, axis = 1)



test
#DecisionTreeClassifier

X_train = train.drop(["Survived","Age","PassengerId","Name","SibSp","Parch","Ticket","Fare","Cabin","Embarked","title"], axis=1)

Y_train = train["Survived"]

X_test  = test.drop(["Survived","Age","PassengerId","Name","SibSp","Parch","Ticket","Fare","Cabin","Embarked","title"], axis=1).copy()



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

accuracy = round(decision_tree.score(X_train, Y_train) * 100, 2)



print(accuracy)

print(X_train.columns)

print(decision_tree.feature_importances_)

dot_data = export_graphviz(decision_tree, out_file=None) 

graph = graphviz.Source(dot_data)

graph 