# libaries used

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import KFold

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
# Read train dataset

df = pd.read_csv("../input/titanic/train.csv")
# Print a concise summary of a DataFrame

# Massive missing values in 'Cabin', some in 'Age' and few in 'Embarked'

df.info()
# Generate descriptive statistics

df.describe()
# number of death is greater than survive

fig = plt.figure(figsize=(8,6))

sns.countplot(x='Survived',data=df,palette='rainbow_r')
# Obviously most men could not survive 

fig = plt.figure(figsize=(8,6))

sns.countplot(x='Survived',data=df, hue='Sex')
# Class 1 is likely to survive

fig = plt.figure(figsize=(8,6))

sns.countplot(x='Survived',data=df,hue='Pclass')
# a number of sibling spouse is not relevant much

fig = plt.figure(figsize=(8,6))

sns.countplot(x='Survived',data=df,hue='SibSp')
# a number of Parent children is not relevant much

fig = plt.figure(figsize=(8,6))

sns.countplot(x='Survived',data=df,hue='Parch')
# clearly most survivors were women and children

plt.figure(figsize=(8,6))

sns.swarmplot(x="Sex",y="Age",hue='Survived',data=df,size=6,palette='spring_r')
# Embarkment from C is the only one with higher survived ratio

fig = plt.figure(figsize=(8,6))

sns.countplot(x='Embarked',hue='Survived',data=df,palette='rainbow_r')
# most people aged 20-35

fig = plt.figure(figsize=(8,6))

sns.distplot(df['Age'],kde=True,bins=50)
# average age at Pclass 1 is higher than other Pclass

fig = plt.figure(figsize=(8,6))

sns.boxplot(x='Pclass',y='Age',data=df)
# age is not a good predictor for survival overall

fig = plt.figure(figsize=(8,6))

sns.boxplot(x='Survived',y='Age',data=df)
# age is a good predictor for survival if specified by gender

fig = plt.figure(figsize=(8,6))

sns.boxplot(x='Sex',y='Age', hue='Survived',data=df)
# Check correlation

df.drop('PassengerId',axis=1).corr()
# Heatmap of Correlation

fig = plt.figure(figsize=(8,6))

sns.heatmap(df.drop('PassengerId',axis=1).corr(),cmap='coolwarm')
# Heatmap of missing values

fig = plt.figure(figsize=(12,10))

sns.heatmap(df.isnull(), cbar=False, cmap='Accent')
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

df
# fill missing values in 'Age' with average age

avg = df['Age'].mean()

df['Age'].fillna(value=avg,inplace=True)

print(avg)
# now we only have some missing values in 'Embarked'

fig = plt.figure(figsize=(12,10))

sns.heatmap(df.isnull(), cbar=False, cmap='Accent')
# S is the highest frequency one

df['Embarked'].value_counts()
# fill missing value of 'Embarked' with 'S'

df['Embarked'].fillna(value='S',inplace=True)
# we no longer have missing values

fig = plt.figure(figsize=(12,10))

sns.heatmap(df.isnull(), cbar=False, cmap='Accent')
# double check with df.info()

df.info()
# get_dummies is equivalent to OneHotEncoder in this situation

# drop_first to avoid multicollinearity

sex = pd.get_dummies(df['Sex'],drop_first=True)

sex
embark = pd.get_dummies(df['Embarked'],drop_first=True)

embark
# Concat dataframe with dummies

df = pd.concat([df,sex,embark],axis=1)

df
# Drop the original features

df.drop(['Sex','Embarked'],axis=1,inplace=True)

df
# Perform MinMaxScaler

minmax_scaler = MinMaxScaler()

df_minmax = minmax_scaler.fit_transform(df)

df = pd.DataFrame(df_minmax, columns=df.columns)

df
X = df.drop(['Survived'], axis=1)

y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=100)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

predicted = nb.predict(X_test)

print(confusion_matrix(y_test, predicted))

print('Accuracy = ', accuracy_score(y_test, predicted))

print('F1 score = ', f1_score(y_test, predicted))

print('Precision = ', precision_score(y_test, predicted))

print('Recall = ', recall_score(y_test, predicted))
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

predicted = dtree.predict(X_test)

print(confusion_matrix(y_test, predicted))

print('Accuracy = ', accuracy_score(y_test, predicted))

print('F1 score = ', f1_score(y_test, predicted))

print('Precision = ', precision_score(y_test, predicted))

print('Recall = ', recall_score(y_test, predicted))
from sklearn.neural_network import MLPClassifier

mlp =  MLPClassifier(max_iter=1000)

mlp.fit(X_train, y_train)

predicted = mlp.predict(X_test)

print(confusion_matrix(y_test, predicted))

print('Accuracy = ', accuracy_score(y_test, predicted))

print('F1 score = ', f1_score(y_test, predicted))

print('Precision = ', precision_score(y_test, predicted))

print('Recall = ', recall_score(y_test, predicted))
## Perform 5-fold Cross Validation

## at the results of each Validation, the followings are shown

## Accuracy

## Precision of each class

## Recall of each class

## F-Measure of each class

## Average F-Measure 
# KFold Cross Validation approach with Naive Bayes

kf = KFold(n_splits=5,shuffle=False)

kf.split(X)    

     

# Initialize the GaussianNB

model = GaussianNB()



# Initialize the lists to keep track of metrics

accuracy_model = []

precision_class0 = []

precision_class1 = []

recall_class0 = []

recall_class1 = []

f1_class0 = []

f1_class1 = []

f1_average = []

 

# Iterate over each train-test split

for train_index, test_index in kf.split(X):

    # Split train-test

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model

    model = model.fit(X_train, y_train)

    # Append to accuracy_model the accuracy of the model

    accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True)*100)

    # Append the precision of class 0 of the model

    precision_class0.append(precision_score(y_test, model.predict(X_test), pos_label=0)*100)

    # Append the precision of class 1 of the model

    precision_class1.append(precision_score(y_test, model.predict(X_test), pos_label=1)*100)

    # Append the recall of class 0 of the model

    recall_class0.append(recall_score(y_test, model.predict(X_test), pos_label=0)*100)

    # Append the recall of class 1 of the model

    recall_class1.append(recall_score(y_test, model.predict(X_test), pos_label=1)*100)

    # Append the f1 score of class 0 of the model

    f1_class0.append(f1_score(y_test, model.predict(X_test), pos_label=0)*100)

    # Append the f1 score of class 1 of the model

    f1_class1.append(f1_score(y_test, model.predict(X_test), pos_label=1)*100)

    # Append the average f1 score of the model

    f1_average.append(f1_score(y_test, model.predict(X_test), average='binary')*100)





print('Results of each CV from Naive Bayes\n')

print('Accuracy: ', np.around(np.array(accuracy_model),3))

print('Precision of Class 0: ', np.around(np.array(precision_class0),3))

print('Precision of Class 1: ', np.around(np.array(precision_class1),3))

print('Recall of Class 0: ', np.around(np.array(recall_class0),3))

print('Recall of Class 1: ', np.around(np.array(recall_class0),3))

print('F-Measure of Class 0: ', np.around(np.array(f1_class0),3))

print('F-Measure of Class 1: ', np.around(np.array(f1_class1),3))

print('Average F-Measure: ', np.around(np.array(f1_average),3))
# KFold Cross Validation approach with Decision Tree

kf = KFold(n_splits=5,shuffle=False)

kf.split(X)    

     

# Initialize the DTree classifier

model = DecisionTreeClassifier()



# Initialize the lists to keep track of metrics

accuracy_model = []

precision_class0 = []

precision_class1 = []

recall_class0 = []

recall_class1 = []

f1_class0 = []

f1_class1 = []

f1_average = []

 

# Iterate over each train-test split

for train_index, test_index in kf.split(X):

    # Split train-test

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model

    model = model.fit(X_train, y_train)

    # Append to accuracy_model the accuracy of the model

    accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True)*100)

    # Append the precision of class 0 of the model

    precision_class0.append(precision_score(y_test, model.predict(X_test), pos_label=0)*100)

    # Append the precision of class 1 of the model

    precision_class1.append(precision_score(y_test, model.predict(X_test), pos_label=1)*100)

    # Append the recall of class 0 of the model

    recall_class0.append(recall_score(y_test, model.predict(X_test), pos_label=0)*100)

    # Append the recall of class 1 of the model

    recall_class1.append(recall_score(y_test, model.predict(X_test), pos_label=1)*100)

    # Append the f1 score of class 0 of the model

    f1_class0.append(f1_score(y_test, model.predict(X_test), pos_label=0)*100)

    # Append the f1 score of class 1 of the model

    f1_class1.append(f1_score(y_test, model.predict(X_test), pos_label=1)*100)

    # Append the average f1 score of the model

    f1_average.append(f1_score(y_test, model.predict(X_test), average='binary')*100)





print('Results of each CV from Decision Tree\n')

print('Accuracy: ', np.around(np.array(accuracy_model),3))

print('Precision of Class 0: ', np.around(np.array(precision_class0),3))

print('Precision of Class 1: ', np.around(np.array(precision_class1),3))

print('Recall of Class 0: ', np.around(np.array(recall_class0),3))

print('Recall of Class 1: ', np.around(np.array(recall_class0),3))

print('F-Measure of Class 0: ', np.around(np.array(f1_class0),3))

print('F-Measure of Class 1: ', np.around(np.array(f1_class1),3))

print('Average F-Measure: ', np.around(np.array(f1_average),3))
# KFold Cross Validation approach with MLP

kf = KFold(n_splits=5,shuffle=False)

kf.split(X)    

     

# Initialize the MLP Classifier

model = MLPClassifier(max_iter=1000)



# Initialize the lists to keep track of metrics

accuracy_model = []

precision_class0 = []

precision_class1 = []

recall_class0 = []

recall_class1 = []

f1_class0 = []

f1_class1 = []

f1_average = []

 

# Iterate over each train-test split

for train_index, test_index in kf.split(X):

    # Split train-test

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model

    model = model.fit(X_train, y_train)

    # Append to accuracy_model the accuracy of the model

    accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True)*100)

    # Append the precision of class 0 of the model

    precision_class0.append(precision_score(y_test, model.predict(X_test), pos_label=0)*100)

    # Append the precision of class 1 of the model

    precision_class1.append(precision_score(y_test, model.predict(X_test), pos_label=1)*100)

    # Append the recall of class 0 of the model

    recall_class0.append(recall_score(y_test, model.predict(X_test), pos_label=0)*100)

    # Append the recall of class 1 of the model

    recall_class1.append(recall_score(y_test, model.predict(X_test), pos_label=1)*100)

    # Append the f1 score of class 0 of the model

    f1_class0.append(f1_score(y_test, model.predict(X_test), pos_label=0)*100)

    # Append the f1 score of class 1 of the model

    f1_class1.append(f1_score(y_test, model.predict(X_test), pos_label=1)*100)

    # Append the average f1 score of the model

    f1_average.append(f1_score(y_test, model.predict(X_test), average='binary')*100)





print('Results of each CV from MLP classifier\n')

print('Accuracy: ', np.around(np.array(accuracy_model),3))

print('Precision of Class 0: ', np.around(np.array(precision_class0),3))

print('Precision of Class 1: ', np.around(np.array(precision_class1),3))

print('Recall of Class 0: ', np.around(np.array(recall_class0),3))

print('Recall of Class 1: ', np.around(np.array(recall_class0),3))

print('F-Measure of Class 0: ', np.around(np.array(f1_class0),3))

print('F-Measure of Class 1: ', np.around(np.array(f1_class1),3))

print('Average F-Measure: ', np.around(np.array(f1_average),3))