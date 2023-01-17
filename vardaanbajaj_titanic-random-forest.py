#  This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
train_df.head()
test_df.head()
print("No. of training examples: " + str(train_df.shape[0]))
print("No. of testing examples: " + str(test_df.shape[0]))
# check for missing values
train_df.isna().sum()
# calculating percent missing values for Age, Cabin, Embarked
print("Percentage missing age values: " + str((train_df['Age'].isna().sum()/train_df.shape[0])*100))
print("Percentage missing cabin values: " + str((train_df['Cabin'].isna().sum()/train_df.shape[0])*100))
print("Percentage missing embarked values: " + str((train_df['Embarked'].isna().sum()/train_df.shape[0])*100))
# Distribution of ages in data
age_distribution = dict(train_df["Age"].value_counts())
lists = sorted(age_distribution.items()) # sorted by key
x, y = zip(*lists)
plt.plot(x, y)
plt.show()
train_data = train_df.copy()
train_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True) # Since age distribution is a bit skewed towards the left side, it's better we use median of ages to fill the missing age values
train_data["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True) # Filling with most frequent occuring value
train_data.drop('Cabin', axis=1, inplace=True) # Dropping Cabin column since 77% cabin values are missing
train_data.isna().sum()
train_data.head()
# creating one hot encodings for Pclass, embarked, sex
training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])
training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)

final_train = training
final_train.head()
#Applying same changes to test set
test_df.isna().sum()
test_data = test_df.copy()
test_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
test_data["Fare"].fillna(train_df["Fare"].median(skipna=True), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing
final_test.head()
sc = StandardScaler()
final_train[["Age", "Fare"]] = sc.fit_transform(final_train[["Age", "Fare"]])
final_test[["Age", "Fare"]] = sc.fit_transform(final_test[["Age", "Fare"]])

final_train.head()
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
cols = ["Age", "SibSp", "Parch", "Fare", "Pclass_1", "Pclass_2", "Pclass_3", "Embarked_C", "Embarked_Q", "Embarked_S", "Sex_male"]
X = final_train[cols]
y = final_train['Survived']
model = RandomForestClassifier()

# selecting top 8 features
rfe = RFE(model, n_features_to_select = 8)
rfe = rfe.fit(X, y)
print('Top 8 most important features: ' + str(list(X.columns[rfe.support_])))
selected_features = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_3', 'Embarked_S', 'Sex_male']
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree (bagging)
bootstrap = [True, False]

# Creating random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
# First create the base model to tune
rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model 
# ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_3', 'Embarked_S', 'Sex_male']
rf_random.fit(final_train[selected_features], final_train['Survived'])
rf_random.best_params_
X = final_train[selected_features]
y = final_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)
# fitting both base random forest model and RandomizedSearchCV random forest model
base_model = RandomForestClassifier()
base_model.fit(X_train, y_train)
best_random = RandomForestClassifier(n_estimators = 2000,
 min_samples_split= 5,
 min_samples_leaf = 1,
 max_features = 'sqrt',
 max_depth = 100,
 bootstrap = True)
best_random.fit(X_train, y_train)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    accuracy = accuracy_score(test_labels, predictions) * 100
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))    
    return accuracy
base_accuracy = evaluate(base_model, X_test, y_test)

random_accuracy = evaluate(best_random, X_test, y_test)
from sklearn.model_selection import GridSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(final_train[selected_features], final_train['Survived'])
grid_search.best_params_
best_grid = RandomForestClassifier(n_estimators = 100,
 min_samples_split= 10,
 min_samples_leaf = 3,
 max_features = 3,
 max_depth = 100,
 bootstrap = True)
best_grid.fit(X_train, y_train)

grid_accuracy = evaluate(best_grid, X_test, y_test)
final_test['Survived'] = base_model.predict(final_test[selected_features])
final_test['PassengerId'] = test_df['PassengerId']

results = final_test[['PassengerId','Survived']]
results.to_csv("submission.csv", index=False)
