# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import all the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read the train.csv file and store the data in pandas dataframe
train_df = pd.read_csv('../input/train.csv')
train_df.head()
# Read the test.csv file and store the data in pandas dataframe
test_df = pd.read_csv('../input/test.csv')
test_df.head()       
# visually access the data by looking at the random samples from all two dataframes
train_df.sample(10)
# visually access the test data
test_df.sample(10)
train_df.info()
train_df.describe()
# Create a copy of dataframe before cleaning
train_df_clean = train_df.copy()
test_df_clean = test_df.copy()
test_df_clean.shape
# Name is in format (Lastname, title. Firstname) - Separate this information in 3 different columns for tidiness
train_df_clean['last_name'] = train_df_clean.Name.str.split(',').str[0]
train_df_clean['title'] = train_df_clean.Name.str.split(',').str[1].str.split('.').str[0]
train_df_clean['first_name'] = train_df_clean.Name.str.split(',').str[1].str.split('.').str[1]

test_df_clean['last_name'] = test_df_clean.Name.str.split(',').str[0]
test_df_clean['title'] = test_df_clean.Name.str.split(',').str[1].str.split('.').str[0]
test_df_clean['first_name'] = test_df_clean.Name.str.split(',').str[1].str.split('.').str[1]
# Ticket column has alphabets and numbers and is of type string
# As ticket will not be used for analysis, we will drop it
train_df_clean.drop('Ticket',axis=1,inplace=True)
test_df_clean.drop('Ticket',axis=1,inplace=True)
# Survived column should be a category type
train_df_clean.Survived = train_df_clean.Survived.astype('category')
# Age, cabin, Embarked has missing values
# Drop cabin column
train_df_clean.drop('Cabin',axis=1,inplace=True)
test_df_clean.drop('Cabin',axis=1,inplace=True)

# Embarked, Age, Fare (in Test dataset) has missing values, the rows will be imputed

# We will check from which station most the passengers boarded the Ship
train_df_clean.Embarked.value_counts()
# From above list, the station is set to 'S' for missing stations
train_df_clean.Embarked.fillna('S', inplace=True)
# As there are passengers from various age groups and data from age is missing in many rows, these rows will be dropped
train_df_clean.dropna(inplace=True)
# We will fill missing values from Fare column of Test dataset with median value of Fare
test_df_clean.Fare.fillna(test_df_clean.Fare.median(), inplace=True)
# We will the missing age values with average age of passengers in both train and test dataset
test_df_clean.Age.fillna(test_df_clean.Age.mean(), inplace=True)
# Count of Family members is divided in two columns Sibsp and Parch
train_df_clean['family_members'] = train_df_clean.SibSp + train_df_clean.Parch
test_df_clean['family_members'] = test_df_clean.SibSp + test_df_clean.Parch
# Remove the columns that are not needed for analysis (cabin) 
# Drop the columns SibSp, Parch, Name
train_df_clean.drop(['SibSp','Parch','Name'],axis=1,inplace=True)
test_df_clean.drop(['SibSp','Parch','Name'],axis=1,inplace=True)
# Convert all column names to lower case to meet python coding standards
train_df_clean.columns = [x.lower() for x in train_df_clean.columns]
test_df_clean.columns = [x.lower() for x in test_df_clean.columns]
train_df_clean.describe()
# Check for duplicates in the data
sum(train_df_clean.duplicated()) 
train_df_clean.age.hist(bins=20,figsize=(6,5));
plt.title('Histogram of passeger Age');
train_df_clean[train_df_clean.age < 1]
train_df_clean.fare.hist(bins=20,figsize=(6,5));
plt.title('Histogram of passeger Fare');
# The passengers who have paid higher Fare
train_df_clean.query("fare > 300") 
# Fare histogram after removing outliers
train_df_clean.query("fare < 200")['fare'].hist(figsize=(6,5));
plt.figure(figsize=(6,5),dpi=80)
sns.countplot(x='family_members', data=train_df_clean)
plt.title('Plot of passenger count vs family members')
plt.xlabel('Family Members')
plt.ylabel('Count');
# passenger count accross various titles
train_df_clean.title.value_counts()
# plot of count of passengers for each type of gender category
plt.figure(figsize=(6,5),dpi=80)
gender_count = train_df_clean.groupby('sex')['passengerid'].count()
gender_category = ['Female', 'Male']
sns.barplot(gender_category, gender_count)
plt.title('Plot of passenger count for each gender')
plt.xlabel('Gender Category')
plt.ylabel('Count');
# plot of count of passengers for each type of class
plt.figure(figsize=(6,5),dpi=80)
class_count = train_df_clean.groupby('pclass')['passengerid'].count()
class_category = [1,2,3]
sns.barplot(class_category, class_count)
plt.title('Plot of passengers count for each class')
plt.xlabel('Passenger Class')
plt.ylabel('Passenger Count');
# plot of count of passengers for each starting station
plt.figure(figsize=(6,5),dpi=80)
embarked_count = train_df_clean.groupby('embarked')['passengerid'].count()
embarked_category = ['Cherbourg','Queenstown','Southampton']
sns.barplot(embarked_category, embarked_count)
plt.title('Plot of passengers count stating from different stations')
plt.xlabel('Start station')
plt.ylabel('Passenger Count'); 
# plot of count of survived passengers for each type of gender category
gender_count = train_df_clean.groupby(['survived','sex'])['passengerid'].count().unstack('survived');
ax = gender_count.plot(kind='bar', rot=0, stacked=True, figsize=(6,5));
gender_category = ['Female', 'Male']
plt.title('Plot of survived passengers count for each gender category')
plt.xlabel('Gender Category')
plt.ylabel('Passenger Count');
plt.legend(['Not Survived','Survived'],loc = 6, bbox_to_anchor = (1.0, 0.5));
plt.xticks([0,1],gender_category);
# plot of count of passengers for each type of class
class_count = train_df_clean.groupby(['survived','pclass'])['passengerid'].count().unstack('survived');
class_category = [1,2,3]
ax = class_count.plot(kind='bar', rot=0, stacked=True, figsize=(6,5));
plt.title('Plot of passengers count for each class')
plt.xlabel('Passenger Class')
plt.legend(['Not Survived','Survived'],loc = 6, bbox_to_anchor = (1.0, 0.5));
plt.ylabel('Passenger Count');
plt.figure(figsize=(6,4), dpi = 100)
sns.boxplot(data = train_df_clean, x = 'survived', y = 'age', hue = 'sex')
plt.legend(loc = 6, bbox_to_anchor = (1.0, 0.5)) # legend to right of figure
plt.xticks(rotation = 0)
plt.title('Box plot for duration vs gender and user type')
plt.xlabel('Gender');
plt.ylabel('Age (years)');
plt.xticks([0,1],['Not Survived','Survived']);
train_df_clean.head()
train_df_clean.loc[train_df_clean["sex"] == "male", "sex"] = 0
train_df_clean.loc[train_df_clean["sex"] == "female", "sex"] = 1

train_df_clean.loc[train_df_clean["embarked"] == "C", "embarked"] = 0
train_df_clean.loc[train_df_clean["embarked"] == "Q", "embarked"] = 1
train_df_clean.loc[train_df_clean["embarked"] == "S", "embarked"] = 2

test_df_clean.loc[test_df_clean["sex"] == "male", "sex"] = 0
test_df_clean.loc[test_df_clean["sex"] == "female", "sex"] = 1

test_df_clean.loc[test_df_clean["embarked"] == "C", "embarked"] = 0
test_df_clean.loc[test_df_clean["embarked"] == "Q", "embarked"] = 1
test_df_clean.loc[test_df_clean["embarked"] == "S", "embarked"] = 2
train_df_clean.head()
test_df_clean.head()
# importing required sklearn modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
# Remove the columns not needed to train the model
x_all = train_df_clean.drop(['survived','title','first_name','last_name'], axis=1)
y_all = train_df_clean['survived']

# Remove the columns not needed for prediction
x_prob = test_df_clean.drop(['title','first_name','last_name'], axis=1)
# Split the training dataset into train and test data into 80:20 ratio
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=0)
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial') # create the classifier
clf.fit(x_train, y_train) # fit the classifier
y_pred = clf.predict(x_test) # predict the values for test set
# Check accuracy score of the predictions
accuracy_score(y_test, y_pred)
# predict the values for problem set
y_prob = clf.predict(x_prob)
# Store the predictions into a dataframe
df_sub = pd.DataFrame()
df_sub['PassengerId'] = x_prob.passengerid
df_sub['Survived'] = y_prob
df_sub.shape
# Save the dataframe to a csv file
df_sub.to_csv('submission.csv', index=False)