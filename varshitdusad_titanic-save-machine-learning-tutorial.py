# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
print("Hello, World!!") #Tester code. Newbie here!
print("The task here is to apply machine learning to develop a classifier to identify which traveler could survive and which could not.")
data = pd.read_csv('../input/gender_submission.csv')
test_data = pd.read_csv('../input/test.csv')
train_data = pd.read_csv('../input/train.csv')
#print(train_data['Parch'].unique().tolist())
train_data.head(10)
#train_data[train_data['Name'].str.contains('Johnson')] #==2]
# Step1 is to explore the data and then clean the data (generally drop Nan valued rows)
# Which variables can make an impact on survival: PClass (economic status), Sex (Females likely to get lifeboat; 
# Sexism thou other name is Chivalry!), Age (children and aged people might be preferred or older people might be)
# left behind {No boat for old men}), SibSp (siblings and spouses survival/death may be correlated), 
# Fare (economic class proxy again. Though doubtful if it can give any more info than Pclass), Parch (Nannies may)
# abandon the kids but parents will not. Children travelling with nanny are prone to death.

## Which variables should be irrelevant for survival: PassengerId, Name (semi-useful as sir names can identify 
# family), Ticket (Redundant or useless information), 

# Explore the histogram of survival.
plt.hist(train_data['Survived'].values, 10, normed=False, facecolor='green')

plt.xlabel('Survived: 1, Died: 0')
plt.ylabel('# of People')
plt.title('Histogram Example')

plt.grid(True)
plt.show()
# Identify the length of training data
train_data.shape
# Plot Bar plot for both sexes corresponding to their deaths and survival.
# Extract data values.
female =  train_data['Sex'].str.contains('female')
male =  (train_data['Sex'] == 'male')
survived = train_data['Survived'].values == 1
died = train_data['Survived'].values == 0

female_death = train_data[female & died]
male_death = train_data[male & died]
female_survived = train_data[female & survived]
male_survived = train_data[male & survived]

plt.bar(['F: Died', 'F: Survived','M: Died', 'M: Survived'], [len(female_death),len(female_survived),len(male_death),len(male_survived)] ,align='center')
plt.xlabel('Distribution of survival by sex')
plt.ylabel('# of People')
plt.title('Barplot example')

plt.grid(True)
plt.show()
len(female_death)+len(female_survived)+len(male_death)+len(male_survived)
#[len(female_death),len(female_survived),len(male_death),len(male_survived)]

# PLot survival with respect to age groups: (0-18,18-50,>50)
survived = train_data['Survived'].values == 1
died = train_data['Survived'].values == 0

children = train_data['Age']<18
adult = (train_data['Age']<50) & (train_data['Age']>=18)
#adult2 = train_data['Age']>=18
#adult = train_data[adult1]
#adult
old = train_data['Age']>=50

children_died = train_data[children & died]
children_survived = train_data[children & survived]
adult_died = train_data[adult & died]
adult_survived = train_data[adult & survived]
old_died = train_data[old & died]
old_survived = train_data[old & survived]

plt.bar(['C: Died', 'C: Survived','A: Died', 'A: Survived', 'O: Died', 'O: Survived'], [len(children_died),len(children_survived),len(adult_died),len(adult_survived),len(old_died),len(old_survived)] ,align='center')
plt.xlabel('Distribution of survival by Age groups')
plt.ylabel('# of People')
plt.title('Barplot example')

plt.grid(True)
plt.show()
# The sum of all these values do not add upto 891 because there are values with NaN in ages.
[len(children_died),len(children_survived),len(adult_died),len(adult_survived),len(old_died),len(old_survived)]
train_data[adult].mean()
# train_data['Pclass'].isnull().values
# Survival rate by Passenger class
class_3 = train_data['Pclass'] == 3
class_2 = train_data['Pclass'] == 2
class_1 = train_data['Pclass'] == 1

class_3_died = train_data[class_3 & died]
class_3_survived = train_data[class_3 & survived]
class_2_died = train_data[class_2 & died]
class_2_survived = train_data[class_2 & survived]
class_1_died = train_data[class_1 & died]
class_1_survived = train_data[class_1 & survived]

plt.bar(['3: Died', '3: Survived','2: Died', '2: Survived', '1: Died', '1: Survived'], [len(class_3_died),len(class_3_survived),len(class_2_died),len(class_2_survived),len(class_1_died),len(class_1_survived)] ,align='center')
plt.xlabel('Distribution of survival by Passenger Class')
plt.ylabel('# of People')
plt.title('Barplot example')

plt.grid(True)
plt.show()
## Now let's build the classifier
## Delete the columns providing redundant or unuseful information for survival classification
useful_train_data =  train_data.copy()
del useful_train_data['Ticket']
del useful_train_data['Fare']
del useful_train_data['Cabin']
del useful_train_data['Embarked']
useful_train_data.columns
useful_train_data['Sex'] = (useful_train_data['Sex'] == 'male')*1
## Drop any row which contains NaN values
useful_train_data = useful_train_data.dropna()
#useful_train_data['Age'].isnull().any()
useful_train_data.columns
## 
y_train = useful_train_data['Survived'].copy()
decisive_features = ['Pclass', 'Sex', 'Age', 'SibSp','Parch']
X_train = useful_train_data[decisive_features].copy()

X_train
survival_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
survival_classifier.fit(X_train, y_train)
type(survival_classifier)
## Predict the results on test data
## First clean the test data also similar to train data
test_data[test_data['Name'].str.contains('Master')]
useful_test_data = test_data.copy()
del useful_test_data['Ticket']
del useful_test_data['Fare']
del useful_test_data['Cabin']
del useful_test_data['Embarked']
useful_test_data.columns
useful_test_data['Sex'] = (useful_test_data['Sex'] == 'male')*1

## Do not Drop any row which contains NaN values
#useful_test_data = useful_test_data.dropna()
useful_test_data['Age'].isnull().any()
#useful_test_data.columns
## Fix null values
null_child_boy = (useful_test_data['Name'].str.contains('Master') & useful_test_data['Age'].isnull())
null_adult = ((useful_test_data['Name'].str.contains('Ms.') | useful_test_data['Name'].str.contains('Miss') | useful_test_data['Name'].str.contains('Mr'))& useful_test_data['Age'].isnull())
useful_test_data.loc[null_child_boy, 'Age'] = 5#._update_inplace(30)
#child_fix['Age']
#adult_fix = useful_test_data.loc[null_adult]
useful_test_data.loc[null_adult, 'Age'] = 30#._update_inplace(30)
useful_test_data[useful_test_data['Name'].str.contains('Master')]

useful_test_data.isnull().any()


#y_test = useful_test_data['Survived'].copy()
decisive_features = ['Pclass', 'Sex', 'Age', 'SibSp','Parch']
X_test = useful_test_data[decisive_features].copy()

predictions = survival_classifier.predict(X_test)
data_to_submit = pd.DataFrame({'PassengerId': useful_test_data['PassengerId'], 'Survived': predictions})
data_to_submit.shape
data_to_submit.to_csv('csv_to_submit.csv', index = False)
