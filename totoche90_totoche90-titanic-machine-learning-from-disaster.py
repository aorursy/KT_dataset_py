# Data analysis
import numpy as np
import pandas as pd

# Data viz
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Outputs
import warnings
warnings.filterwarnings('ignore')

# Machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
# Raw import
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
# Show head of training set
train_df.head()
# Show head of testing set
test_df.head()
# Temporarily convert training set to strings and use the describe function of pandas
train_df.applymap(lambda x: x if pd.isnull(x) else str(x)).describe()
# Drop the features
train_df.drop(['PassengerId', 'Ticket', 'Cabin'], inplace=True, axis=1)
test_df.drop(['PassengerId', 'Ticket', 'Cabin'], inplace=True, axis=1)
# We start by extracting the Title from the Name
train_df["Title"] = train_df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
# We then observe the distribution of the different titles
train_df[['Title','Name']].groupby(['Title'], as_index=False).count().sort_values(by='Name', ascending=False)
# Make the transformations
train_df["Title"] = train_df["Title"].replace(['Mlle', 'Ms'], 'Miss')
train_df["Title"] = train_df["Title"].replace('Mme', 'Mrs')
train_df["Title"] = train_df["Title"].replace(['Lady', 'Countess', 'Sir'], 'Royal')
train_df["Title"] = train_df["Title"].replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Other')
# We print the influence of Title using pandas' groupby
train_df[["Title", "Survived"]].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Or we can visualize the trend with seaborn's barplot
sns.barplot(x="Title", y="Survived", data=train_df)
# Check for missing values in the training set
print ("Name feature: There are", len(train_df.index) - train_df.Name.count(), "/", len(train_df.index) ,"null values in the training set.")
# Check for missing values in the testing set
print ("Name feature: There are", len(test_df.index) - test_df.Name.count(), "/", len(test_df.index) ,"null values in the testing set.")
# Replace Name with Title in the training set
train_df.drop(['Name'], inplace=True, axis=1)
# Apply the same transformations to testing set
test_df["Title"] = test_df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
test_df["Title"] = test_df["Title"].replace(['Mlle', 'Ms'], 'Miss')
test_df["Title"] = test_df["Title"].replace('Mme', 'Mrs')
test_df["Title"] = test_df["Title"].replace(['Lady', 'Countess', 'Sir'], 'Royal')
test_df["Title"] = test_df["Title"].replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Other')
test_df.drop(['Name'], inplace=True, axis=1)
# Correlation using pandas' groupby
train_df[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Correlation using seaborn's barplot
sns.barplot(x="Pclass", y="Survived", data=train_df)
# Check for missing values in the training set
print ("Pclass feature: There are", len(train_df.index) - train_df.Pclass.count(), "/", len(train_df.index) ,"null values in the training set.")
# Check for missing values in the testing set
print ("Pclass feature: There are", len(test_df.index) - test_df.Pclass.count(), "/", len(test_df.index) ,"null values in the testing set.")
# Correlation using pandas' groupby
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Correlation using seaborn's barplot
sns.barplot(x="Sex", y="Survived", data=train_df)
# Check for missing values in the training set
print ("Sex feature: There are", len(train_df.index) - train_df.Sex.count(), "/", len(train_df.index) ,"null values in the training set.")
# Check for missing values in the testing set
print ("Sex feature: There are", len(test_df.index) - test_df.Sex.count(), "/", len(test_df.index) ,"null values in the testing set.")
# Correlation using pandas' groupby
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Correlation using seaborn's barplot
sns.barplot(x="SibSp", y="Survived", data=train_df)
# Check for missing values in the training set
print ("SibSp feature: There are", len(train_df.index) - train_df.SibSp.count(), "/", len(train_df.index) ,"null values in the training set.")
# Check for missing values in the testing set
print ("SibSp feature: There are", len(test_df.index) - test_df.SibSp.count(), "/", len(test_df.index) ,"null values in the testing set.")
# Correlation using pandas' groupby
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Correlation using seaborn's barplot
sns.barplot(x="Parch", y="Survived", data=train_df)
# Check for missing values in the training set
print ("Parch feature: There are", len(train_df.index) - train_df.Parch.count(), "/", len(train_df.index) ,"null values in the training set.")
# Check for missing values in the testing set
print ("Parch feature: There are", len(test_df.index) - test_df.Parch.count(), "/", len(test_df.index) ,"null values in the testing set.")
# We visualize the correlation
sns.barplot(x="Embarked", y="Survived", data=train_df)
# Check for missing values in the training set
print ("Embarked feature: There are", len(train_df.index) - train_df.Embarked.count(), "/", len(train_df.index) ,"null values in the training set.")
# Check for missing values in the testing set
print ("Embarked feature: There are", len(test_df.index) - test_df.Embarked.count(), "/", len(test_df.index) ,"null values in the testing set.")
# Most frequent port of embarkation
most_freq_port = train_df.Embarked.dropna().mode()[0]
# Fill the training set
train_df['Embarked'] = train_df['Embarked'].fillna(most_freq_port)
# Print the value
print ("We have filled the missing values with: ",most_freq_port)
print ("Length of 'Embarked' series after modification: ",train_df.Embarked.count())
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Fare', bins=6)
# Check for missing values in the training set
print ("Fare feature: There are", len(train_df.index) - train_df.Fare.count(), "/", len(train_df.index) ,"null values in the training set.")
# Check for missing values in the testing set
print ("Fare feature: There are", len(test_df.index) - test_df.Fare.count(), "/", len(test_df.index) ,"null values in the testing set.")
# Median of existing values
missing_value = test_df.Fare.median()
# Fill the missing value
test_df['Fare'] = test_df['Fare'].fillna(missing_value)
# Print the value
print ("We have filled the missing value with: ",missing_value)
print ("Length of 'Fare' series after modification: ",test_df.Fare.count())
train_df.head()
# We create the feature in the training set
train_df['FareRangeRaw'] = pd.qcut(train_df['Fare'], 4) #if we try 5 we lose the trend
train_df.head()
# We visualize the correlation
sns.barplot(x="FareRangeRaw", y="Survived", data=train_df)
# Get max fare in testing set
test_df["Fare"].max()
# We apply the binning and labels to both datasets
bins = [-1, 7.91, 14.454, 31.0, 600]
train_df['FareRange'] = pd.cut(train_df['Fare'], bins, labels=[1,2,3,4])
train_df.head()
test_df['FareRange'] = pd.cut(test_df['Fare'], bins, labels=[1,2,3,4])
# We replace Fare with FareRange in training set
train_df.drop(["FareRangeRaw","Fare"],inplace=True,axis=1)
# Same for testing set
test_df.drop(["Fare"],inplace=True,axis=1)
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=30)
# Check for missing values in the training set
print ("Age feature: There are", len(train_df.index) - train_df.Age.count(), "/", len(train_df.index) ,"null values in the training set.")
# Check for missing values in the testing set
print ("Age feature: There are", len(test_df.index) - test_df.Age.count(), "/", len(test_df.index) ,"null values in the testing set.")
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', hue="Title", size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
# We define a function that will compute the median of a feature corresponding to any given combination of 2 other features
def fillna_custom(df,ax1,val1,ax2,val2,nanax):
    return df[(df[ax1] == val1) & (df[ax2] == val2)][[nanax]].median()[0]
    
# We fill the missing values in both datasets
list_sex = ["male", "female"]
list_pclass = [1, 2, 3]
for sex in list_sex:
    for pclass in list_pclass:
        train_df.loc[(train_df.Age.isnull()) & (train_df.Sex == sex) & (train_df.Pclass == pclass),'Age'] = fillna_custom(train_df.copy(),"Sex",sex,"Pclass",pclass,"Age")
        test_df.loc[(test_df.Age.isnull()) & (test_df.Sex == sex) & (test_df.Pclass == pclass),'Age'] = fillna_custom(test_df.copy(),"Sex",sex,"Pclass",pclass,"Age")
print ("Length of Age in training set:", train_df.Age.count())
print ("Length of Age in testing set:", test_df.Age.count())
# We cut the Age feature
bins = [0, 10, 20, 30, 40, 50, 75, 80]
train_df['AgeRangeRaw'] = pd.cut(train_df['Age'], bins)
# We visualize the correlation
sns.barplot(x="AgeRangeRaw", y="Survived", data=train_df)
test_df["Age"].max()
# We apply the binning and labels to both datasets
bins = [0, 10, 20, 30, 40, 50, 75, 80]
train_df['AgeRange'] = pd.cut(train_df['Age'], bins, labels=[1,2,3,4,5,6,7])
test_df['AgeRange'] = pd.cut(test_df['Age'], bins, labels=[1,2,3,4,5,6,7])
# We replace Age with AgeRange in training set
train_df.drop(["AgeRangeRaw","Age"],inplace=True,axis=1)
# Same for testing set
test_df.drop(["Age"],inplace=True,axis=1)
train_df.head()
test_df.head()
train_df.head()
test_df.head()
# X_train is train_df without the output column "Survived"
X_train = train_df.drop("Survived", axis=1)
X_train.head()
# Y_train is only the output "Survived" of train_df
Y_train = train_df["Survived"]
Y_train.head()
# X_test is simply test_df
X_test = test_df
X_test.head()
# Title feature mapping
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Other": 6}
# Application to X_train
X_train['Title'] = X_train['Title'].map(title_mapping)
# Application to X_test
X_test['Title'] = X_test['Title'].map(title_mapping)
# Sex feature mapping
sex_mapping = {"male": 1, "female": 2}
# Application to X_train
X_train['Sex'] = X_train['Sex'].map(sex_mapping)
# Application to X_test
X_test['Sex'] = X_test['Sex'].map(sex_mapping)
# Embarked feature mapping
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
# Application to X_train
X_train['Embarked'] = X_train['Embarked'].map(embarked_mapping)
# Application to X_test
X_test['Embarked'] = X_test['Embarked'].map(embarked_mapping)
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
# KNN neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
# Random forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
# Scores
acc_decision_tree = decision_tree.score(X_train, Y_train) * 100
acc_linear_svc = linear_svc.score(X_train, Y_train) * 100
acc_svc = svc.score(X_train, Y_train) * 100
acc_knn = knn.score(X_train, Y_train) * 100
acc_gaussian = gaussian.score(X_train, Y_train) * 100
acc_perceptron = perceptron.score(X_train, Y_train) * 100
acc_sgd = sgd.score(X_train, Y_train) * 100
acc_random_forest = random_forest.score(X_train, Y_train) * 100

scores_df = pd.DataFrame({ 'Algorithm' : ["Decision Tree","Linear SVC","Support Vector Machines","KNN neighbors","Gaussian Naive Bayes","Perceptron","Stochastic Gradient Descent","Random forest"],
                          'Score' : [acc_decision_tree,acc_linear_svc,acc_svc,acc_knn,acc_gaussian,acc_perceptron,acc_sgd,acc_random_forest]})

scores_df.sort_values(by='Score', ascending=False)
import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(decision_tree, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render("iris")
graph = graphviz.Source(dot_data)  
graph 
