# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# 1. Import 
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Figure inline
%matplotlib inline
sns.set()
train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
gender = pd.read_csv("../input/gender_submission.csv")

# 2. Check and understand data
train.head()
train.describe()
train.info()
g = sns.factorplot(x="Survived", col="Sex", kind="count", data=train)
# take-away Female more likely to survive than male
train["Survived"].value_counts(normalize=True)
print("male survivors:\n", train["Survived"][train["Sex"]=='male'].value_counts(normalize=True))
print("female survivors:\n", train["Survived"][train["Sex"]=='female'].value_counts(normalize=True))
# take-away Pcclass 1 are more likely to survive
sns.factorplot(x="Survived", col="Pclass", kind='count', data=train)
# take-away Embarked are less likely to survive in Southhampton
sns.factorplot(x="Survived", col="Embarked", kind='count', data=train)
# Most passengers paid less than 100, paying more resulted in more surviaval chance
fig, ax = plt.subplots(ncols=2, figsize=(10,5))
sns.distplot(train.Fare, kde=False, ax=ax[0])
train.groupby('Survived').Fare.hist(alpha=0.6)
fig, ax = plt.subplots(ncols=2, figsize=(10,5))
train_drop = train.dropna()
sns.distplot(train_drop.Age, kde=False, ax=ax[0])
train.groupby('Survived').Age.hist(alpha=0.6)
sns.stripplot(x="Survived", y="Fare", data=train, alpha=0.3, jitter=True)
sns.swarmplot(x="Survived", y="Fare", data=train)
train.groupby("Survived").Fare.describe()
# scatter plot seems that survivors, either paid or were young
sns.lmplot(x="Age", y="Fare", hue="Survived", data=train, fit_reg=False, scatter_kws={'alpha':0.5})
sns.pairplot(train_drop, hue="Survived")
# Start to create the ML model. But first combine a dataset to do some preprocessing
train_survived = train["Survived"]
train_drop = train.drop('Survived', axis=1)
data = pd.concat([train_drop, df_test])
data.head()
# Check the feature if any family member is on the Titanic and its impact
#data["Fam_size"] = data["SibSp"] + data["Parch"]
# clean and format data
median_age = data["Age"].median()
median_fare = data["Fare"].median()
data["Age"] = data["Age"].fillna(median_age)
data["Fare"] = data["Fare"].fillna(median_fare)
data['Embarked'] = data['Embarked'].fillna('S') # most common value
data.info()
#bin numerical data because there can be outliers that skew/noise the data
data["CatAge"] = pd.qcut(data.Age, q=4, labels=False)  #imput of a 1D array or series.
data["CatFare"] = pd.qcut(data.Fare, q=4, labels=False)

# some etxra feature engineering
data["Title"] = data.Name.apply(lambda x: re.search("([A-Z][a-z]+)\.", x).group(1))
sns.countplot(x='Title', data=data)
plt.xticks(rotation=45)
data["Title"] = data["Title"].replace({"Mme":"Mrs", "Mlle":"Miss", "Ms":"Miss"})
data["Title"] = data["Title"].replace(['Don', 'Dona', 'Rev', 'Dr','Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],"Special")
sns.countplot(x="Title", data=data)
# cabin NA for some people could mean they couldn't pay for a cabin and results in important info
data["has_cabin"] = ~data.Cabin.isnull()
data = data.drop(["Cabin", "Ticket","Name", "PassengerId", "SibSp", "Parch", "Age", "Fare"], axis=1)
data.head()
# transform all categorical columns to numerical columns, for machine learning algorithms
data = pd.get_dummies(data, drop_first=True)
data.head()
# 4. Make a decision tree
data_train = data.iloc[:891]
data_test = data.iloc[891:]
y = train_survived.values
X = data_train.values
test = data_test.values
max_depth = 3 # To stop splitting
min_samples_split = 5
data.info()
# Method 1 by using a train_test
# splitting train test in order to check the best performing max_depth size.
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.35, random_state=42, stratify=y)
#iterate over the max_depth to define to choose max_depth = 3
#dep = np.arange(1,6)
#train_accu = np.empty(len(dep))
#test_accu = np.empty(len(dep))

#for i, k in enumerate(dep):
#    my_tree = tree.DecisionTreeClassifier(max_depth=k)
#    my_tree.fit(X_train, y_train)
#    train_accu[i] = my_tree.score(X_train, y_train)
#    test_accu[i] =  my_tree.score(X_test, y_test)
    
#plt.plot(dep, train_accu, label='train accuracy')
#plt.plot(dep, test_accu, label = 'test accuracy')
#plt.xlabel('depth')
#plt.ylabel('accuracy')
#plt.show()

dep = np.arange(1,9)
param_grid = {'max_depth': dep}
my_tree = tree.DecisionTreeClassifier()
my_tree_cv = GridSearchCV(my_tree, param_grid=param_grid, cv=5)
my_tree_cv.fit(X, y)
print("Tuned Decision Tree Parameters: {}".format(my_tree_cv.best_params_))
print("Best score is {}".format(my_tree_cv.best_score_))
#my_tree = tree.DecisionTreeClassifier(max_depth = max_depth)
#my_tree.fit(X, y)
#my_tree.fit(X,y)
#my_tree.feature_importances_
my_tree_cv.score(X, y)
prediction = my_tree_cv.predict(test)


# 6. Test the Random forrest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, y)
prediction = random_forest.predict(test)
random_forest.score(X, y)

# 5. predict the test set
df_test['Survived'] = prediction
df_test[['PassengerId', 'Survived']].to_csv('mysolution4.csv', index=False)
