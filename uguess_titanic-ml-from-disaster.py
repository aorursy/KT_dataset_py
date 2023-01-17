# Import pandas that will help us read the provided csv files into dataframes
import pandas as pd
training_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')

# Lets view the training set
training_set.head(n = 10)
# Lets view the test set
test_set.head(n = 10)
# View Fare vs Pclass columns
training_set.iloc[:, [2, 9]].head(n = 10)
# Dropping insignificant/unnecessary columns
training_set = training_set.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis = 1)
test_set = test_set.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis = 1)
# Lets view the training set now
training_set.head(n = 10)
# Lets view the test set now
test_set.head(n = 10)
# Get statistical information about training set
training_set.describe(include = 'all')
# Get statistical information about test set
test_set.describe(include = 'all')
training_set['Age'] = training_set['Age'].fillna(training_set.mean()[0])
test_set['Age'] = test_set['Age'].fillna(test_set.mean()[0])
training_set['Embarked'] = training_set['Embarked'].fillna(training_set['Embarked'].mode()[0])
# Now we can see that Age and Embarked features are not missing any values
print(training_set['Age'].count())
print(test_set['Age'].count())
print(test_set['Embarked'].count())
# Import data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# draw a bar plot - Age-Group vs. survival
plt.subplots(1, 1, figsize = (15, 5))
age_bins = [0, 10, 20, 30 , 40, 50, 60, 70, 80, 90]
age_group = pd.cut(training_set['Age'], age_bins)
survived = training_set['Survived'].values
sns.barplot(x = age_group, y = survived)
plt.xlabel('Age-Group')
plt.ylabel('Survived')
plt.legend()
plt.show()
# draw a bar plot - Pclass vs. survival
plt.subplots(1, 1, figsize = (10, 5))
sns.barplot(x = 'Pclass', y = 'Survived', data = training_set)
plt.xlabel('Ticket class')
plt.ylabel('Survived')
plt.legend()
plt.show()
import numpy as np

# Since number of siblings (sibsp) and parents (parch) denote if passenger had a family onboard
# we are going to combine these columns to create a column called HasFamily
training_set['HasFamily'] = np.where(training_set['SibSp'] + training_set['Parch'] > 0, 1, 0)
test_set['HasFamily'] = np.where(test_set['SibSp'] + test_set['Parch'] > 0, 1, 0)
# Now we can drop SibSp and Parch columns
training_set = training_set.drop(['SibSp', 'Parch'], axis = 1)
test_set = test_set.drop(['SibSp', 'Parch'], axis = 1)

training_set.head(n=10)
test_set.head(n=10)
# draw a factor plot - HasFamily vs. survival
sns.factorplot(x = 'HasFamily', y = 'Survived', data = training_set)
plt.xlabel('Has Family')
plt.ylabel('Survived')
plt.legend()
plt.show()
# draw a bar plot - Sex vs. survival
sns.barplot(x = 'Sex', y = 'Survived', data = training_set)
plt.xlabel('Sex')
plt.ylabel('Survived')
plt.legend()
plt.show()
# Let's divide the data into features and label
features_train = training_set.iloc[:, 1:].values 
labels_train = training_set.iloc[:, 0].values
features_test = test_set.iloc[:, :].values
# label for test data is not provided

# lets use LabelEncoder to convert Sex data into numerical form
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
features_train[:, 1] = label_encoder.fit_transform(features_train[:, 1])
features_test[:, 1] = label_encoder.fit_transform(features_test[:, 1])
# Now encode Embarked feature
label_encoder = LabelEncoder()
features_train[:, 3] = label_encoder.fit_transform(features_train[:, 3])
features_test[:, 3] = label_encoder.fit_transform(features_test[:, 3])
# One Hot Encode Embarked feature
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(categorical_features=[3]) # Embarked Column index is 4
features_train = one_hot_encoder.fit_transform(features_train).toarray()
one_hot_encoder = OneHotEncoder(categorical_features=[3]) # Embarked Column index is 4
features_test = one_hot_encoder.fit_transform(features_test).toarray()
features_train = features_train[:, 1:] # dropping column at 0th index
features_test = features_test[:, 1:] # dropping column at 0th index
# Create a dataframe to store algorithms and their accuracies
algo_accuracy = pd.DataFrame(columns = ['Algorithm', 'Accuracy'])

# Logistic Regression
# Scaling features so that one features doesn't dominate the other
from sklearn.preprocessing import StandardScaler
# Standardize features by removing the mean and scaling to unit variance
# This basically gets the difference between the value and the mean of values in that column
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test) #no need to fit since training set already does this

# Fit Logistic Regression to the training data
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(features_train, labels_train)

# Predit the test result and calculate accuracy
labels_pred = log_reg.predict(features_test)
accuracy_log_reg = log_reg.score(features_train, labels_train) * 100
algo_accuracy.loc[len(algo_accuracy)] = ['Logistic Regression', accuracy_log_reg]
print(accuracy_log_reg)
# K-Nearest Neigbors
# No need to scale the data since its already done above.
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
KNN.fit(features_train, labels_train)

# Predit the test result and calculate accuracy
labels_pred = KNN.predict(features_test)
accuracy_KNN = KNN.score(features_train, labels_train) * 100
algo_accuracy.loc[len(algo_accuracy)] = ['K-Nearest Neighbors', accuracy_KNN]
print(accuracy_KNN)
# Support Vector Machines
# No need to scale the data since its already done above.
from sklearn.svm import SVC
svc = SVC(kernel = 'rbf')
svc.fit(features_train, labels_train)

# Predit the test result and calculate accuracy
labels_pred = svc.predict(features_test)
accuracy_svc = svc.score(features_train, labels_train) * 100
algo_accuracy.loc[len(algo_accuracy)] = ['Support Vector Machines', accuracy_svc]
print(accuracy_svc)
# Naive Bayes
# No need to scale the data since its already done above.
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(features_train, labels_train)

# Predit the test result and calculate accuracy
labels_pred = nb.predict(features_test)
accuracy_nb = nb.score(features_train, labels_train) * 100
algo_accuracy.loc[len(algo_accuracy)] = ['Naive Bayes', accuracy_nb]
print(accuracy_nb)
# Decision Trees
# No need to scale the data since its already done above.
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(criterion = 'entropy')
decision_tree.fit(features_train, labels_train)

# Predit the test result and calculate accuracy
labels_pred = decision_tree.predict(features_test)
accuracy_decision_tree = decision_tree.score(features_train, labels_train) * 100
algo_accuracy.loc[len(algo_accuracy)] = ['Decision Trees', accuracy_decision_tree]
print(accuracy_decision_tree)
# Random Forest
# No need to scale the data since its already done above.
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators = 20, criterion = 'entropy')
random_forest.fit(features_train, labels_train)

# Predit the test result and calculate accuracy
labels_pred = random_forest.predict(features_test)
accuracy_random_forest = random_forest.score(features_train, labels_train) * 100
algo_accuracy.loc[len(algo_accuracy)] = ['Random Forest', accuracy_random_forest]
print(accuracy_random_forest)
algo_accuracy.head(n=10)
# Decision Trees has the highest accuracy
decision_tree = DecisionTreeClassifier(criterion = 'entropy')
decision_tree.fit(features_train, labels_train)

# Predit the test result
decision_tree_preds = decision_tree.predict(features_test)

# Build a submission file
orig_test_set = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({
        "PassengerId": orig_test_set["PassengerId"],
        "Survived": decision_tree_preds
    })
submission.to_csv('titanic_preds.csv', index=False)