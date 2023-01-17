import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

sns.set("notebook")

# Suppressing Warnings

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# Glimpse of the train dataset.

train.head()
# Glimpse of the test dataset.

test.head()
# No. of rows/columns in the train dataset.

train.shape
test.shape
# Showing the information about the data type in each column of dataframe as well no. of rows.

train.info(verbose=True)
test.info(verbose=True)
train.describe(include = 'all')
test.describe(include = 'all')
# Percentage missing values for any cloumn having missing values.

null_columns=train.columns[train.isnull().any()]

round((100*train[null_columns].isnull().sum())/len(train),2)
# Percentage missing values for any cloumn having missing values in test data.

null_columns1=test.columns[test.isnull().any()]

round((100*test[null_columns1].isnull().sum())/len(test),2)
## `Cabin` has more than 77% missing data and is no use for building our model. Also, `Name` has no significance in model building.

## Hence we will remove both these columns from train data.

train = train.drop(['Cabin', 'Name'],axis = 1)
## Similar to Train dataset.

test = test.drop(['Cabin','Name'], axis=1)
train['Ticket'].describe()
train['Ticket'].head(10)
train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)
train['Embarked'].describe()
test['Embarked'].describe()
train['Embarked'] = train['Embarked'].replace(np.nan, "S")

test['Embarked'] = test['Embarked'].replace(np.nan, "S")
train['Age'].describe()
plt.figure(figsize=(12,9))

train.Age[train.Survived==1].plot(kind='hist', label = 'Survived')

train.Age[train.Survived==0].plot(kind='hist', alpha = 0.75,label = 'Died')

plt.title("Histogram of Survivors & Non Survivors with respect to Age");

plt.xlabel("Age Bins");

plt.legend();
plt.figure(figsize=(12,9));

train.Fare[train.Survived==1].plot(kind='hist', label = 'Survived');

train.Fare[train.Survived==0].plot(kind='hist', alpha = 0.75,label = 'Died');

plt.title("Histogram of Survivors & Non Survivors with respect to Fare");

plt.xlabel("Fare Bins");

plt.legend();
# Selecting numeric columns for outlier analysis and treatment

num_cols = ['Age','Fare']

train[num_cols].describe(percentiles=[.25,.5,.75,.90,.95,.99])
sns.catplot(x="Survived", y="Age", hue="Sex",kind="box", data=train);
sns.catplot(x="Survived", y="Fare", hue="Sex",kind="box", data=train);
# Percentage missing values for any cloumn having missing values.

null_columns=train.columns[train.isnull().any()]

round((100* train[null_columns].isnull().sum())/len(train),2)
# Percentage missing values for any cloumn having missing values in test data.

null_columns1=test.columns[test.isnull().any()]

round((100*test[null_columns1].isnull().sum())/len(test),2)
train['Age'].describe()
test['Age'].describe()
train["Age"].hist(bins=10)
train["Age"] = train["Age"].fillna(train["Age"].median())

test["Age"] = test["Age"].fillna(test["Age"].median())

## As there is no missing value in Fare cloumn for Train and only 1 value missing in Test, we will fill the missing value in Fare column with median too.

train["Fare"] = train["Fare"].fillna(train["Fare"].median())

test["Fare"] = test["Fare"].fillna(test["Fare"].median())
# Let's check finally of any missing values.

null_columns=train.columns[train.isnull().any()]

round((100* train[null_columns].isnull().sum())/len(train),2)
# Percentage missing values for any cloumn having missing values in test data.

null_columns1=test.columns[test.isnull().any()]

round((100* test[null_columns1].isnull().sum())/len(test),2)
train.shape
test.shape
cuts = [0,5,12,18,35,60,100]

labels = ["Infant","Child","Teenager","Young Adult","Adult","Senior"]

train['Age_Category'] = pd.cut(train['Age'], cuts, labels = labels)

test['Age_Category'] = pd.cut(test['Age'], cuts, labels = labels)
train['Family'] = train['SibSp'] + train['Parch']

test['Family'] = test['SibSp'] + test['Parch']
## now we can drop the Age column.

train = train.drop(['Age','SibSp','Parch'], axis=1)

test = test.drop(['Age','SibSp','Parch'], axis=1)
train["Survived"].value_counts()
## Survival Rates

round(100 * sum(train["Survived"])/len(train["Survived"].index),2)
sns.distplot(train['Fare'], bins=10);
train["Age_Category"].describe()
plt.figure(figsize=(12,9));

sns.countplot(train["Age_Category"]);
sns.catplot('Sex', data = train, kind = 'count');

plt.title('Passengers by Sex');
sns.catplot('Pclass', data = train, kind = 'count');

plt.title('Passengers by Passenger Class');
sns.catplot('Embarked', data = train, kind = 'count');

plt.title('Passengers by Embarked');
sns.countplot(x = "Sex", hue = "Survived", data=train);
sns.countplot(x = "Pclass", hue = "Survived", data=train);

sns.countplot(x = "Embarked", hue = "Survived", data=train);
sns.factorplot(x="Sex", y="Survived",kind='bar', data=train);

plt.title('Survival rate by Gender');

plt.ylabel("Survival Rate");
sns.factorplot(x="Pclass", y="Survived", hue = 'Sex',kind='bar', data=train);

plt.title('Survival Rate by Class');

plt.ylabel("Survival Rate");
sns.factorplot(x="Embarked", y="Survived", hue = 'Sex',kind='bar', data=train)

plt.title('Survival Rate by Port of Embarkment');

plt.ylabel("Survival Rate");
# List of variables to map



varlist =  ["Sex"]



# Defining the map function

def binary_map(x):

    return x.map({'male': 0, "female": 1})



# Applying the function to the housing list

train[varlist] = train[varlist].apply(binary_map)

test[varlist] = test[varlist].apply(binary_map)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(train[['Embarked', 'Age_Category']], drop_first=True)

dummy2 = pd.get_dummies(test[['Embarked', 'Age_Category']], drop_first=True)



train = pd.concat([train, dummy1], axis=1)

test = pd.concat([test, dummy2], axis=1)
# We have created dummies for the below variable, so we can drop it

train = train.drop(['Embarked', 'Age_Category'], 1)

test = test.drop(['Embarked', 'Age_Category'], 1)
## Putting the feature variable to X

X_train = train.drop(['PassengerId', 'Survived'], axis=1)

X_test = test.drop(['PassengerId'], axis=1)

## Putting response variable to y

y_train = train['Survived']
X_test.shape
X_train.shape
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[['Fare']] = scaler.fit_transform(X_train[['Fare']])
X_test[['Fare']] = scaler.transform(X_test[['Fare']])
X_train.head()
X_test.head()
y_train.head()
# Let's see the correlation matrix 

plt.figure(figsize = (20,10))        # Size of the figure

sns.heatmap(train.corr(),annot = True)

plt.show()
## Importing XGBoost library and additional scikit libraries for GridSearchCV and model evaluation

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn import model_selection, metrics   #Additional scklearn functions

from sklearn.model_selection import GridSearchCV   #Perforing grid search

cv_params = {'max_depth': [5,7,9], 'min_child_weight': [3,5,7]}

ind_params = {'learning_rate': 0.01, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 

             'objective': 'binary:logistic'}



gbm = xgb.XGBClassifier(**ind_params).fit(X_train, y_train)





optimized_GBM = GridSearchCV(gbm, cv_params, scoring = 'roc_auc', cv = 10, n_jobs = -1) 



optimized_GBM.fit(X_train, y_train)
optimized_GBM.cv_results_
predictions = optimized_GBM.predict(X_test)

submission = pd.DataFrame({ 'PassengerId': test['PassengerId'],

                            'Survived': predictions })

submission.to_csv("predictions_optimised.csv", index=False)