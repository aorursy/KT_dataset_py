# Import libraries for linear algebra and loading data
import numpy as np
import pandas as pd

# Load training and test data into dataframes
orig_training_set = pd.read_csv('../input/train.csv')
orig_test_set = pd.read_csv('../input/test.csv')
# View training data
orig_training_set.head(n=10)
# View test data
orig_test_set.head(n=10)
# Drop unnecessary columns from training and test set
training_set = orig_training_set.drop(['Name', 'PassengerId', 'Ticket'], axis = 1)
test_set = orig_test_set.drop(['Name', 'PassengerId', 'Ticket'], axis = 1)

print(training_set.columns)
print(test_set.columns)
# View statistical information about the training data
training_stats = training_set.describe(include='all')
print(training_stats)
# View statistical information about the test data
test_stats = test_set.describe(include='all')
print(test_stats)
# Lets focus on count index of these stats
training_stats.loc['count', :] 
test_stats.loc['count', :]
# Drop Cabin feature from training and test set
training_set = training_set.drop(['Cabin'], axis = 1)
test_set = test_set.drop(['Cabin'], axis = 1)

print(training_set.columns)
print(test_set.columns)
# Update variable that carries statistical information
training_stats = training_set.describe(include='all')
test_stats = test_set.describe(include='all')
# Import data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x=training_set.Pclass, y=training_set.Survived)
plt.ylabel('Passengers Survived(%)')
sns.barplot(x=training_set.Survived, y=training_set.Sex)
sns.kdeplot(
        training_set.loc[training_set['Survived'] == 0, 'Age'].dropna(), 
        color='red', 
        label='Did not survive')
sns.kdeplot(
        training_set.loc[training_set['Survived'] == 1, 'Age'].dropna(), 
        color='green', 
        label='Survived')
plt.xlabel('Age')
plt.ylabel('Passengers Survived(%)')
sns.pointplot(training_set.Embarked, training_set.Survived)
sns.regplot(x=training_set.SibSp, y=training_set.Survived, color='r')
sns.regplot(x=training_set.Parch, y=training_set.Survived, color='b')
sns.swarmplot(x=training_set.Survived, y=training_set.Fare)
# lets view the stats on training and test data again to do some analysis
print(training_stats)
print(test_stats)
sns.distplot(training_set['Age'].dropna(), bins=20, rug=True, kde=True)
training_set['Age'] = training_set.Age.fillna(training_stats.loc['mean', 'Age'])
test_set['Age'] = test_set.Age.fillna(test_stats.loc['mean', 'Age'])

training_stats = training_set.describe(include='all')
test_stats = test_set.describe(include='all')

print(training_stats.loc['count', 'Age'])
print(test_stats.loc['count', 'Age'])
sns.countplot(x=training_set.Embarked, palette="Greens_d");
from statistics import mode
mode_embarked = mode(training_set['Embarked'])
training_set['Embarked'] = training_set['Embarked'].fillna(mode_embarked)

training_stats = training_set.describe(include='all')

print(training_stats.loc['count', 'Embarked'])
# Understand the relation between empty Fare feature value and other features values
empty_fare = test_set[test_set['Fare'].isnull()]
print(empty_fare)
use_fare = test_set[(test_set['Pclass'] == 3) & 
                    (test_set['SibSp'] == 0) & 
                    (test_set['Parch'] == 0) &
                    (test_set['Embarked'] == 'S')]
test_set['Fare'] = test_set['Fare'].fillna(use_fare['Fare'].iloc[0]);
test_stats = test_set.describe(include='all')

print(test_stats.loc['count', 'Fare'])
# Engineer the Age data, drop the Age feature and add engineered categorical Age feature.
# If a passenger is less than or equal to 20, then Kid/Teenager, if greater than 20 and
# less than or equal to 40 then Young/Adult and so on...
training_set['Kid/Teenager'] = np.where(training_set['Age'] <= 20, 1, 0)
training_set['Young/Adult'] = np.where((training_set['Age'] > 20) & (training_set['Age'] <= 40), 1, 0)
training_set['Mature'] = np.where((training_set['Age'] > 40) & (training_set['Age'] <= 60), 1, 0)
training_set['Elderly'] = np.where(training_set['Age'] > 60, 1, 0)

test_set['Kid/Teenager'] = np.where(test_set['Age'] <= 20, 1, 0)
test_set['Young/Adult'] = np.where((test_set['Age'] > 20) & (test_set['Age'] <= 40), 1, 0)
test_set['Mature'] = np.where((test_set['Age'] > 40) & (test_set['Age'] <= 60), 1, 0)
test_set['Elderly'] = np.where(test_set['Age'] > 60, 1, 0)

# Now we can drop the Age column
training_set = training_set.drop(['Age'], axis=1)
test_set = test_set.drop(['Age'], axis=1)

# Lets view training data now
training_set.head(n=10)
# Lets view test data now
test_set.head(n=10)
training_set['HasFamily'] = np.where(training_set['SibSp'] + training_set['Parch'] > 0, 1, 0)
test_set['HasFamily'] = np.where(test_set['SibSp'] + test_set['Parch'] > 0, 1, 0)

# Now we can drop SibSp and Parch columns
training_set = training_set.drop(['SibSp', 'Parch'], axis=1)
test_set = test_set.drop(['SibSp', 'Parch'], axis=1)

# Lets view training data now
training_set.head(n=10)
# Lets view test data now
test_set.head(n=10)
# We can compare percentages between passengers, who has family, survived the disaster versus did not
hasfamily = len(training_set[(training_set['HasFamily'] == 1)])
fam_survived = (len(training_set[(training_set['HasFamily'] == 1) & (training_set['Survived'] == 1)]) / hasfamily) * 100
fam_didnotsurvive = (len(training_set[(training_set['HasFamily'] == 1) & (training_set['Survived'] == 0)]) / hasfamily) * 100

print ("{0:.2f}".format(fam_survived) + "% of passenger with family survived")
print ("{0:.2f}".format(fam_didnotsurvive) + "% of passenger with family did not survive")
# Now lets compare percentages between passengers, who do not have family, survived the disaster versus did not
nofamily = len(training_set[(training_set['HasFamily'] == 0)])
nofam_survived = (len(training_set[(training_set['HasFamily'] == 0) & (training_set['Survived'] == 1)]) / nofamily) * 100
nofam_didnotsurvive = (len(training_set[(training_set['HasFamily'] == 0) & (training_set['Survived'] == 0)]) / nofamily) * 100

print ("{0:.2f}".format(nofam_survived) + "% of passenger with no family survived")
print ("{0:.2f}".format(nofam_didnotsurvive) + "% of passenger with no family did not survive")
# Apply pandas qcut to Fare feature
#fare_groups = pd.qcut(training_set['Fare'], 4)
#fare_groups.unique()
# Lets assign values to these fare group feature as we did to age feature
#training_set['Fare-Group1'] = np.where(training_set['Fare'] <= 7.91, 1, 0)
#training_set['Fare-Group2'] = np.where((training_set['Fare'] > 7.91) & (training_set['Fare'] <= 14.454), 1, 0)
#training_set['Fare-Group3'] = np.where((training_set['Fare'] > 14.454) & (training_set['Fare'] <= 31), 1, 0)
#training_set['Fare-Group4'] = np.where(training_set['Fare'] > 31, 1, 0)

#test_set['Fare-Group1'] = np.where(test_set['Fare'] <= 7.91, 1, 0)
#test_set['Fare-Group2'] = np.where((test_set['Fare'] > 7.91) & (test_set['Fare'] <= 14.545), 1, 0)
#test_set['Fare-Group3'] = np.where((test_set['Fare'] > 14.454) & (test_set['Fare'] <= 31), 1, 0)
#test_set['Fare-Group4'] = np.where(test_set['Fare'] > 31, 1, 0)

# View the training data
#training_set.head(n=10)
# View the test data
#test_set.head(n=10)
# We can now drop the Fare feature out of training and test data
#training_set = training_set.drop(['Fare'], axis = 1)
#test_set = test_set.drop(['Fare'], axis = 1)
# Encode categorical feature - Sex
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
training_set['Sex'] = encoder.fit_transform(training_set['Sex'])

# Lets view the training data now
training_set.head(n=10)
# encode test data Sex feature
encoder = LabelEncoder()
test_set['Sex'] = encoder.fit_transform(test_set['Sex'])

# Lets view the training data now
test_set.head(n=10)
# Now lets encode Embarked feature
encoder = LabelEncoder()
training_set['Embarked'] = encoder.fit_transform(training_set['Embarked'])

# Lets view the training data now
training_set.head(n=10)
# encode test data Embarked feature
encoder = LabelEncoder()
test_set['Embarked'] = encoder.fit_transform(test_set['Embarked'])

# Lets view the training data now
test_set.head(n=10)
# Apply One-Hot encoding to Embarked feature in training data
training_set = pd.get_dummies(data=training_set, prefix=['Embarked'], columns=['Embarked'])
training_set.head(n=10)
# Apply One-Hot encoding to Embarked feature in test data
test_set = pd.get_dummies(data=test_set, prefix=['Embarked'], columns=['Embarked'])
test_set.head(n=10)
# Apply One-Hot Encoding to Pclass feature in training data
training_set = pd.get_dummies(data=training_set, prefix=['Pclass'], columns=['Pclass'])
training_set.head(n=10)
# Apply One-Hot Encoding to Pclass feature in training data
test_set = pd.get_dummies(data=test_set, prefix=['Pclass'], columns=['Pclass'])
test_set.head(n=10)
# Drop columns to avoid dummy variable trap in our training set
training_set = training_set.drop(['Embarked_2', 'Pclass_3', 'Elderly'], axis=1)
training_set.head(n=10)
# Drop columns to avoid dummy variable trap in our test set
test_set = test_set.drop(['Embarked_2', 'Pclass_3', 'Elderly'], axis=1)
test_set.head(n=10)
# All the columns in our training data expect Survived represent our features
features_train = training_set.loc[:, training_set.columns != 'Survived'].values
label_train = training_set.loc[:, training_set.columns == 'Survived'].values
features_test = test_set.values

# features is now ndarray type (Sparse matrix of shape) since our model (classifiers) take this type while fitting
print(features_train[0])
# labels
print(label_train[0])
# Apply feature scaling to our features_train
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)
print(features_train[0])
# First lets prepare few things to avoid redundant coding and for result visualization

# Create a dataframe to store algorithms and their accuracies
algo_accuracy = pd.DataFrame(columns = ['Algorithm', 'Accuracy'])

# Create a method that gets the mean accuracy score obtained by the classifer (model)
# And store the accuracy in algo_accuracy
def get_store_accuracy(classifier, clf_name, X, y):
    accuracy = classifier.score(X, y) * 100
    algo_accuracy.loc[len(algo_accuracy)] = [clf_name, "{0:.2f}".format(accuracy) + ' %']
# 1. K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(features_train, label_train.ravel())

# store accuracy
get_store_accuracy(knn, 'K-Nearest Neighbors', features_train, label_train)
# 2. Support Vector Machines
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(features_train, label_train.ravel())

# store accuracy
get_store_accuracy(svc, 'Support Vector Machines', features_train, label_train)
# 3. Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(features_train, label_train.ravel())

# store accuracy
get_store_accuracy(nb, 'Naive Bayes', features_train, label_train)
# 4. Decision Trees
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(criterion = 'entropy')
decision_tree.fit(features_train, label_train.ravel())

# store accuracy
get_store_accuracy(decision_tree, 'Decision Trees', features_train, label_train)
# 4. Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators = 20, criterion = 'entropy')
random_forest.fit(features_train, label_train.ravel())

# store accuracy
get_store_accuracy(random_forest, 'Random Forest', features_train, label_train)
# Lets view and compare the Algorithms and their accuracies
algo_accuracy.head(n=10)
# Use decision tree classifier to predict the test results
decision_tree_preds = decision_tree.predict(features_test)

# Build a submission file
submission = pd.DataFrame({
        "PassengerId": orig_test_set["PassengerId"],
        "Survived": decision_tree_preds
    })
submission.to_csv('Titanic_Predictions.csv', index=False)