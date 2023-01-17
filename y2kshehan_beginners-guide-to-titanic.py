

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn import metrics



#visualization libraries

import matplotlib.pyplot as plt

import missingno as msno # plotting missing data

import seaborn as sns # Seaborn for plotting and styling



#setup  Matplotlib (magic function) . Plots will render within the notebook itself

%matplotlib inline 



# Input data files are available in the "../input/" directory.

#Importing the dataset

train_dataset = pd.read_csv('../input/train.csv')

test_dataset = pd.read_csv('../input/test.csv')

train_dataset.head()



train_dataset.info()
train_dataset.describe(include="all")
train_dataset.isnull().sum()
#There are 891 total records in the dataset

print('Missing values percentages feature wise')

print('Age : %.2f%%' % ((train_dataset['Age'].isnull().sum()/891)*100))

print('Cabin : %.2f%%' % ((train_dataset['Cabin'].isnull().sum()/891)*100))

print('Embarked : %.2f%%' % ((train_dataset['Embarked'].isnull().sum()/891)*100))
#Visualizing missing data

msno.matrix(df=train_dataset, figsize=(20,14), color=(0.5,0,0))
sns.barplot(x="Pclass", y="Survived", data=train_dataset)



#Percentage of passengers who survived Pclass wise

print('Percentage of passengers who survived Pclass wise')

print('Pclass 1 : %.2f%%' % (train_dataset['Survived'][train_dataset['Pclass'] == 1].value_counts(normalize = True )[1]*100))

print('Pclass 2 : %.2f%%' % (train_dataset['Survived'][train_dataset['Pclass'] == 2].value_counts(normalize = True )[1]*100))

print('Pclass 3 : %.2f%%' % (train_dataset['Survived'][train_dataset['Pclass'] == 3].value_counts(normalize = True )[1]*100))

sns.barplot(x="Sex", y="Survived", data=train_dataset)



#Percentage of passengers who survived Sex wise

print('Percentage of passengers who survived Sex wise')

print('Sex Male : %.2f%%' % (train_dataset['Survived'][train_dataset['Sex'] == 'male'].value_counts(normalize = True )[1]*100))

print('Sex Female : %.2f%%' % (train_dataset['Survived'][train_dataset['Sex'] == 'female'].value_counts(normalize = True )[1]*100))

bins = [ 0, 4, 12, 18, 25, 35, 50 ,65, np.inf]

labels = [ '0-4', '4-12', '12-18', '18-25', '25-35', '35-50', '50-65','65>']

train_dataset['AgeGroup'] = pd.cut(train_dataset["Age"], bins, labels = labels)

test_dataset['AgeGroup'] = pd.cut(train_dataset["Age"], bins, labels = labels)



sns.barplot(x="AgeGroup", y="Survived", data=train_dataset)
sns.barplot(x="SibSp", y="Survived", data=train_dataset)
sns.barplot(x="Parch", y="Survived", data=train_dataset)
train_dataset['FareGroup'] = pd.qcut(train_dataset["Fare"], 4, labels = [1,2,3,4])

test_dataset['FareGroup'] = pd.qcut(train_dataset["Fare"], 4, labels = [1,2,3,4])



sns.barplot(x="FareGroup", y="Survived", data=train_dataset)
sns.barplot(x="Embarked", y="Survived", data=train_dataset)
train_dataset = train_dataset.drop(['PassengerId', 'Ticket'], axis=1)

test_dataset = test_dataset.drop(['PassengerId' , 'Ticket'], axis=1)
#combine both train_dataset and test_dataset 

combine = [train_dataset, test_dataset]



#Extract title data from both train_dataset and test_dataset  

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



#Show all the titles of train_dataset divided by Sex

pd.crosstab(train_dataset['Title'], train_dataset['Sex'])
#replace misspelled titles from both datasets

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
#replace misspelled titles from both datasets

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'Dona'], 'Uncommon')



pd.crosstab(train_dataset['Title'], train_dataset['Sex'])
train_dataset = train_dataset.drop('Name', axis=1)

test_dataset = test_dataset.drop('Name', axis=1)
train_dataset = pd.get_dummies(data = train_dataset , columns=['Sex'] , drop_first = True)

test_dataset = pd.get_dummies(data = test_dataset , columns=['Sex'] , drop_first = True)

train_dataset.head()
#Age values of the passengers who also belongs to Master group in the title feature 

train_dataset[train_dataset['Title']=='Master']['Age']
age_group_mapping = {'0-4':1,'4-12':2,'12-18':3,'18-25':4,'25-35':5, '35-50':6,'50-65':7,'65>':8}

train_dataset['AgeGroup'] = train_dataset['AgeGroup'].map(age_group_mapping)

test_dataset['AgeGroup'] = train_dataset['AgeGroup'].map(age_group_mapping)



train_dataset.head()
age_title_mode_values = {'Mr': train_dataset[train_dataset["Title"] == 'Mr']["AgeGroup"].mode() , 

                         'Mrs': train_dataset[train_dataset["Title"] == 'Mrs']["AgeGroup"].mode() , 

                         'Miss': train_dataset[train_dataset["Title"] == 'Miss']["AgeGroup"].mode() , 

                         'Master': train_dataset[train_dataset["Title"] == 'Master']["AgeGroup"].mode() , 

                         'Uncommon': train_dataset[train_dataset["Title"] == 'Uncommon']["AgeGroup"].mode()}



for x in range(len(train_dataset["AgeGroup"])):

    if np.isnan(train_dataset["AgeGroup"][x]):

        train_dataset["AgeGroup"][x] = age_title_mode_values[train_dataset["Title"][x]]

        

for x in range(len(test_dataset["AgeGroup"])):

    if np.isnan(test_dataset["AgeGroup"][x]):

        test_dataset["AgeGroup"][x] = age_title_mode_values[test_dataset["Title"][x]]        
train_dataset.head()
train_dataset=train_dataset.drop(['Age','Title'],axis=1)

test_dataset=test_dataset.drop(['Age','Title'],axis=1)
train_dataset['HasFamily'] = train_dataset.SibSp + train_dataset.Parch > 0

test_dataset['HasFamily'] = test_dataset.SibSp + test_dataset.Parch > 0
train_dataset=train_dataset.drop(['SibSp','Parch'],axis=1)

test_dataset=test_dataset.drop(['SibSp','Parch'],axis=1)



train_dataset = pd.get_dummies(data = train_dataset , columns=['HasFamily'] , drop_first = True)

test_dataset = pd.get_dummies(data = test_dataset , columns=['HasFamily'] , drop_first = True)



train_dataset.head()
train_dataset=train_dataset.drop('Fare',axis=1)

test_dataset=test_dataset.drop('Fare',axis=1)

train_dataset.head()
train_dataset=train_dataset.drop('Cabin',axis=1)

test_dataset=test_dataset.drop('Cabin',axis=1) 

train_dataset.head()
train_dataset['Embarked'] = train_dataset['Embarked'].fillna(train_dataset['Embarked'].mode())

test_dataset['Embarked'] = test_dataset['Embarked'].fillna(test_dataset['Embarked'].mode())
train_dataset = pd.get_dummies(data = train_dataset , columns=['Embarked'] , drop_first = True)

test_dataset = pd.get_dummies(data = test_dataset , columns=['Embarked'] , drop_first = True)
train_dataset.head()
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Correlation Heatmap', y=1.05, size=15)

sns.heatmap(train_dataset.corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
X = train_dataset.drop("Survived" , axis = 1)

y = train_dataset["Survived"]



X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)
#Feature Scaling 

sc_X = StandardScaler()

X_train_without_scaling = X_train

X_test_without_scaling = X_test

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train,y_train)



y_pred = classifier.predict(X_test)



#Confution Matrix

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm ,  annot=True ,cmap = "Blues" ,cbar =False ,fmt='d')
accuracy_lr = round(metrics.accuracy_score(y_test, y_pred)*100 , 2)

print(accuracy_lr)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0)

classifier.fit(X_train,y_train)



y_pred = classifier.predict(X_test)



#Confution Matrix

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm ,  annot=True ,cmap = "Blues" ,cbar =False ,fmt='d')
accuracy_dtc = round(metrics.accuracy_score(y_test, y_pred)*100 , 2)

print(accuracy_dtc)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10 ,criterion = 'entropy' , random_state = 0)

classifier.fit(X_train,y_train)



y_pred = classifier.predict(X_test)



#Confution Matrix

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm ,  annot=True ,cmap = "Blues" ,cbar =False ,fmt='d')
accuracy_rfc = round(metrics.accuracy_score(y_test, y_pred)*100 , 2)

print(accuracy_rfc)
ranking = np.argsort(-classifier.feature_importances_)

f, ax = plt.subplots(figsize=(10, 5))

sns.barplot(x=classifier.feature_importances_[ranking], y=X_train_without_scaling.columns.values[ranking], orient='h')

ax.set_xlabel("feature importance")

plt.tight_layout()

plt.show()
#Fitting logistic Regression to the Training set

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf' ,  random_state = 0)

classifier.fit(X_train,y_train)



y_pred = classifier.predict(X_test)



#Confution Matrix

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm ,  annot=True ,cmap = "Blues" ,cbar =False ,fmt='d')
accuracy_ksvm = round(metrics.accuracy_score(y_test, y_pred)*100 , 2)

print(accuracy_ksvm)
#Fitting logistic Regression to the Training set

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5 , metric = 'minkowski' , p = 2)

classifier.fit(X_train,y_train)



y_pred = classifier.predict(X_test)



#Confution Matrix

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm ,  annot=True ,cmap = "Blues" ,cbar =False ,fmt='d')
accuracy_knn = round(metrics.accuracy_score(y_test, y_pred)*100 , 2)

print(accuracy_knn)
#Fitting Naive Bayes Classification to the Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train,y_train)



y_pred = classifier.predict(X_test)



#Confution Matrix

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm ,  annot=True ,cmap = "Blues" ,cbar =False ,fmt='d')
accuracy_nbc = round(metrics.accuracy_score(y_test, y_pred)*100 , 2)

print(accuracy_nbc)
accuracies = pd.DataFrame({

    'Model': ['Logistic Regression', 'Decision Tree Classification', 'Random forests Classification', 

              'Kernel SVM Classification', 'K-Nearest Neighbor Classification', 'Naive Bayes Classification'],

    'Score': [accuracy_lr, accuracy_dtc, accuracy_rfc, 

              accuracy_ksvm, accuracy_knn, accuracy_nbc]})

accuracies.sort_values(by='Score', ascending=False)