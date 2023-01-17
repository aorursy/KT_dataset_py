import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



%matplotlib inline

sns.set_style("whitegrid")

plt.style.use("fivethirtyeight")
df = pd.read_csv("../input/titanic/train.csv")

df.head()
df.info()
df.shape
pd.set_option("display.float", "{:.2f}".format)

df.describe()
df.Survived.value_counts()
df.Survived.value_counts().plot(kind="bar", color=["salmon", "lightblue"])
# Checking for messing values

df.isna().sum()
df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Cabin'] = df['Cabin'].fillna('Missing')

df = df.dropna()
categorical_val = []

continous_val = []

for column in df.columns:

    print('===============================================================================')

    print(f"{column} : {df[column].unique()}")

    if len(df[column].unique()) <= 6:

        categorical_val.append(column)

    else:

        continous_val.append(column)
categorical_val
plt.figure(figsize=(15, 15))



for i, column in enumerate(categorical_val, 1):

    plt.subplot(3, 3, i)

    df[df["Survived"] == 0][column].hist(bins=35, color='blue', label='Have Survived = NO', alpha=0.6)

    df[df["Survived"] == 1][column].hist(bins=35, color='red', label='Have Survived = YES', alpha=0.6)

    plt.legend()

    plt.xlabel(column)
plt.figure(figsize=(15, 15))



for i, column in enumerate(continous_val, 1):

    plt.subplot(3, 3, i)

    df[df["Survived"] == 0][column].hist(bins=35, color='blue', label='Have Survived = NO', alpha=0.6)

    df[df["Survived"] == 1][column].hist(bins=35, color='red', label='Have Survived = YES', alpha=0.6)

    plt.legend()

    plt.xlabel(column)
# Let's make our correlation matrix a little prettier

corr_matrix = df.corr()

fig, ax = plt.subplots(figsize=(15, 15))

ax = sns.heatmap(corr_matrix,

                 annot=True,

                 linewidths=0.5,

                 fmt=".2f");

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
df.drop('Survived', axis=1).corrwith(df.Survived).plot(kind='bar', grid=True, figsize=(12, 8), 

                                                   title="Correlation with Survival")
categorical_val.remove('Survived')

dataset = pd.get_dummies(df, columns = categorical_val)
dataset.head()
print(df.columns)

print(dataset.columns)
from sklearn import preprocessing

  

# label_encoder object knows how to understand word labels.

label_encoder = preprocessing.LabelEncoder()

  

# Encode labels in column 'species'.

for col in ['PassengerId', 'Name', 'Age', 'Ticket', 'Fare', 'Cabin']: 

    df[col] = label_encoder.fit_transform(df[col])

    dataset[col] = label_encoder.fit_transform(dataset[col])
from sklearn.preprocessing import StandardScaler



s_sc = StandardScaler()

col_to_scale = ['PassengerId', 'Name', 'Age', 'Ticket', 'Fare', 'Cabin']

dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])
dataset.head()
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score



def print_score(clf, X_train, y_train, X_test, y_test, train=True):

    if train:

        pred = clf.predict(X_train)

        print("Train Result:\n================================================")

        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")

        print("_______________________________________________")

        print("Classification Report:", end='')

        print(f"\tPrecision Score: {precision_score(y_train, pred) * 100:.2f}%")

        print(f"\t\t\tRecall Score: {recall_score(y_train, pred) * 100:.2f}%")

        print(f"\t\t\tF1 score: {f1_score(y_train, pred) * 100:.2f}%")

        print("_______________________________________________")

        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")

        

    elif train==False:

        pred = clf.predict(X_test)

        print("Test Result:\n================================================")        

        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")

        print("_______________________________________________")

        print("Classification Report:", end='')

        print(f"\tPrecision Score: {precision_score(y_test, pred) * 100:.2f}%")

        print(f"\t\t\tRecall Score: {recall_score(y_test, pred) * 100:.2f}%")

        print(f"\t\t\tF1 score: {f1_score(y_test, pred) * 100:.2f}%")

        print("_______________________________________________")

        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
from sklearn.model_selection import train_test_split



X = dataset.drop('Survived', axis=1)

y = dataset.Survived



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression(solver='liblinear')

log_reg.fit(X_train, y_train)
print_score(log_reg, X_train, y_train, X_test, y_test, train=True)

print_score(log_reg, X_train, y_train, X_test, y_test, train=False)
test_score = accuracy_score(y_test, log_reg.predict(X_test)) * 100

train_score = accuracy_score(y_train, log_reg.predict(X_train)) * 100



results_df = pd.DataFrame(data=[["Logistic Regression", train_score, test_score]], 

                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

results_df
from sklearn.neighbors import KNeighborsClassifier



knn_classifier = KNeighborsClassifier()

knn_classifier.fit(X_train, y_train)



print_score(knn_classifier, X_train, y_train, X_test, y_test, train=True)

print_score(knn_classifier, X_train, y_train, X_test, y_test, train=False)
test_score = accuracy_score(y_test, knn_classifier.predict(X_test)) * 100

train_score = accuracy_score(y_train, knn_classifier.predict(X_train)) * 100



results_df_2 = pd.DataFrame(data=[["K-nearest neighbors", train_score, test_score]], 

                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
from sklearn.svm import SVC



svm_model = SVC(kernel='rbf', gamma=0.1, C=1.0)

svm_model.fit(X_train, y_train)
print_score(svm_model, X_train, y_train, X_test, y_test, train=True)

print_score(svm_model, X_train, y_train, X_test, y_test, train=False)
test_score = accuracy_score(y_test, svm_model.predict(X_test)) * 100

train_score = accuracy_score(y_train, svm_model.predict(X_train)) * 100



results_df_2 = pd.DataFrame(data=[["Support Vector Machine", train_score, test_score]], 

                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
from sklearn.tree import DecisionTreeClassifier





tree = DecisionTreeClassifier(random_state=0)

tree.fit(X_train, y_train)



print_score(tree, X_train, y_train, X_test, y_test, train=True)

print_score(tree, X_train, y_train, X_test, y_test, train=False)
test_score = accuracy_score(y_test, tree.predict(X_test)) * 100

train_score = accuracy_score(y_train, tree.predict(X_train)) * 100



results_df_2 = pd.DataFrame(data=[["Decision Tree Classifier", train_score, test_score]], 

                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV



rand_forest = RandomForestClassifier(n_estimators=1000, random_state=0)

rand_forest.fit(X_train, y_train)



print_score(rand_forest, X_train, y_train, X_test, y_test, train=True)

print_score(rand_forest, X_train, y_train, X_test, y_test, train=False)
test_score = accuracy_score(y_test, rand_forest.predict(X_test)) * 100

train_score = accuracy_score(y_train, rand_forest.predict(X_train)) * 100



results_df_2 = pd.DataFrame(data=[["Random Forest Classifier", train_score, test_score]], 

                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb.fit(X_train, y_train)



print_score(xgb, X_train, y_train, X_test, y_test, train=True)

print_score(xgb, X_train, y_train, X_test, y_test, train=False)
test_score = accuracy_score(y_test, xgb.predict(X_test)) * 100

train_score = accuracy_score(y_train, xgb.predict(X_train)) * 100



results_df_2 = pd.DataFrame(data=[["XGBoost Classifier", train_score, test_score]], 

                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
def feature_imp(df, model):

    fi = pd.DataFrame()

    fi["feature"] = df.columns

    fi["importance"] = model.feature_importances_

    return fi.sort_values(by="importance", ascending=False)
feature_imp(X, rand_forest).plot(kind='barh', figsize=(12,7), legend=False)
test = pd.read_csv("../input/titanic/test.csv")

test.head()
test.shape
# Checking for messing values

df.isna().sum()
test['Cabin'] = test['Cabin'].fillna('Missing')

test['Age'] = test['Age'].fillna(test['Age'].mean())

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
categorical_val = []

continous_val = []

for column in test.columns:

    print('===================================================================================')

    print(f"{column} : {test[column].unique()}")

    if len(test[column].unique()) <= 6:

        categorical_val.append(column)

    else:

        continous_val.append(column)
data = pd.get_dummies(test, columns = categorical_val)

data.head()
from sklearn import preprocessing

  

# label_encoder object knows how to understand word labels.

label_encoder = preprocessing.LabelEncoder()

  

# Encode labels in column 'species'.

for col in [ 'Name', 'Age', 'Ticket', 'Fare', 'Cabin']: 

    test[col] = label_encoder.fit_transform(test[col])

    data[col] = label_encoder.fit_transform(data[col])
from sklearn.preprocessing import StandardScaler



s_sc = StandardScaler()

col_to_scale = ['Name', 'Age', 'Ticket', 'Fare', 'Cabin']

data[col_to_scale] = s_sc.fit_transform(data[col_to_scale])
y_pred = rand_forest.predict(data)
test['Survived'] = y_pred
test
submission = test.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
submission
submission.to_csv('results.csv',index=False)