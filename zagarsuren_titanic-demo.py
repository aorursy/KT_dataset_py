# data analysis and wrangling

import pandas as pd

import numpy as np



# visualization

import seaborn as sns

import matplotlib.pyplot as plt



# machine learning

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix





import os

print(os.listdir("../input"))
# Read data

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



# Concat train & test set

test['Survived'] = "0"

test['train_test'] = "test"

test = test[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',

       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'train_test']]



train['train_test'] = "train"

data = train.append(test)
# Зорчигдийн нэрнээс Title гэсэн шинэ хувьсагч гаргаж авах

data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

stat_min = 10 

title_names = (data['Title'].value_counts() < stat_min) 

data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

print(data['Title'].value_counts())

print("-"*10)
# Title хувьсагчийг тоо руу хөрвүүлэх

label = LabelEncoder()

data['Title_Code'] = label.fit_transform(data['Title'])



# Family Size гэсэн шинэ хувьсагч

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data2 = data[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'FamilySize',

        'Fare', 'Embarked', 'Title_Code', 'train_test']]



# Cabin шинэ хувьсагч

Cabin = data['Cabin']

cabin = pd.DataFrame(Cabin.str.slice(stop = 1)) 

cabin = cabin.fillna('cabin_NA')

data3 = pd.concat([data2, cabin], axis=1, ignore_index=False)



# One hot encoding - Категори хувьсагчдыг кодлох

cat_feat = data3[['Cabin', 'Embarked', 'Sex','Pclass','Title_Code']]

num_feat = data3[['Age', 'Fare', 'FamilySize']]

train_test = data3[['PassengerId', 'Survived','train_test']]

cat_feat['Pclass'] = cat_feat['Pclass'].astype(object)

cat_feat['Title_Code'] = cat_feat['Title_Code'].astype(object)

transformed_data = pd.get_dummies(cat_feat)



transformed_data = transformed_data[['Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F',

       'Cabin_G', 'Cabin_T', 'Cabin_cabin_NA', 'Embarked_C', 'Embarked_Q', 'Embarked_S',

       'Sex_male', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Title_Code_0', 'Title_Code_1', 'Title_Code_2', 'Title_Code_3',

       'Title_Code_4']]
# Scaling       

from sklearn.preprocessing import MinMaxScaler

def min_max_scale_data(numerical_data):

    scaler = MinMaxScaler().fit(numerical_data)

    data = pd.DataFrame(scaler.transform(numerical_data), columns=numerical_data.columns.values, index=numerical_data.index)

    return data



# Missing Value нөхөх

num_feat = num_feat.fillna(num_feat.mean())

num_feat2 = min_max_scale_data(num_feat)



data4 = pd.concat([num_feat2, transformed_data], axis=1, ignore_index=False)

data5 = pd.concat([train_test, data4], axis=1, ignore_index=False)



# Training data

train_F = (data5.train_test == 'train')

train = data5.loc[train_F]

train = train.drop("train_test", axis = 1)

train['Survived'] = train['Survived'].astype(int)



# Test data

test_F = (data5.train_test == 'test')

test_KAGGLE = data5.loc[test_F]

test_KAGGLE = test_KAGGLE.drop("train_test", axis = 1)
# X болон y-г салгах 

Model_subset = train.drop("PassengerId", axis = 1)

X = Model_subset.drop("Survived", axis=1)

y = Model_subset["Survived"].copy()



# Test, train хуваалт

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.3, random_state = 0)



# RF fit хийх

clf = RandomForestClassifier(n_estimators=720, max_leaf_nodes=32, max_features = 16, n_jobs=-1)

clf.fit(X_train, y_train)
# Чухал нөлөөтэй хувьсагчид

importances = pd.DataFrame({"feature": X_train.columns, "importance": clf.feature_importances_})

importances.head()   
sns.barplot(data=importances.sort_values("importance", ascending=False).head(10), x="importance", y="feature")
# pred on training data

y_train_pred = clf.predict(X_train)

accuracy_score(y_train, y_train_pred)
# pred on test set unseen data

y_test_pred = clf.predict(X_test)

accuracy_score(y_test, y_test_pred)
print (y_test_pred [0:5])

print (y_test [0:5])
sns.heatmap(confusion_matrix(y_test, y_test_pred),annot=True,fmt='2.0f')
# Submission

test_org = test_KAGGLE[['Age', 'Fare', 'FamilySize', 'Cabin_A', 'Cabin_B', 'Cabin_C', 

       'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_cabin_NA',

       'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Sex_male', 'Pclass_1',

       'Pclass_2', 'Pclass_3', 'Title_Code_0', 'Title_Code_1',

       'Title_Code_2', 'Title_Code_3', 'Title_Code_4']]

y_pred_subm = clf.predict(test_org)



y_pred_subm = pd.DataFrame(y_pred_subm)

y_pred_subm.columns = ['pred']

prediction = pd.concat([y_pred_subm, test], axis=1, ignore_index=False)

prediction = prediction[['PassengerId', 'pred']]

prediction.columns = ['PassengerId','Survived']

prediction.head()
prediction.to_csv('./submission.csv', index=False)