# Credit must be given to Anisotropic's "Introduction to Ensembling/Stacking in Python" 
# and Yassine Ghouzam's "Titanic Top 4% ensembling model"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

from collections import Counter

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


sns.set(style='white', context='notebook', palette='deep')
# Load the training and testing data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# We need to save the Passenger Id's for submission because it will be dropped for training
IDtest = test['PassengerId'] 
train.head() # Preview the data
# Check for null values
print(train.isnull().sum())
print()
print(test.isnull().sum())
# Outlier detection using the Tukey method
# For each feature, we determine the interquartile range (Q3 - Q1) and define outliers as being
# outside (iqr-outlier_step, iqr+outlier_step)
def detect_outlier_indices(df, n, features):
    indices = []
    for f in features:
        Q1 = np.percentile(df[f], 25) # First quartile
        Q3 = np.percentile(df[f], 75) # Third quartile
        IQR = Q3 - Q1 # Interquartile range
        outlier_step = 1.5 * IQR
        # Determine outlier indices for feature
        outliers_for_feature = df[(df[f] < (Q1 - outlier_step)) | 
                                  (df[f] > (Q3 + outlier_step))].index
        indices.extend(outliers_for_feature)
    outlier_count = Counter(indices) # Dict: {item: count}
    return [index for index in outlier_count if outlier_count[index] > n]

outlier_indices = detect_outlier_indices(train, 2, ['Age', 'SibSp', 'Parch', 'Fare'])
train.iloc[outlier_indices] # View outliers to drop
# reset_index(drop=True) adds new index column without creating a new column from old indices
train = train.drop(outlier_indices).reset_index(drop=True)
train.size # 10 less samples
# Sex vs. Survived
graph = sns.factorplot(x='Sex', y='Survived', data=train, kind='bar')
graph = graph.set_ylabels('Survival Probability')
# Pclass vs. Survived
graph = sns.factorplot(x='Pclass', y='Survived', data=train, kind='bar')
graph = graph.set_ylabels('Survival Probability')
# SibSp vs. Survived
graph = sns.factorplot(x='SibSp', y='Survived', data=train, kind='bar')
graph = graph.set_ylabels('Survival Probability')
# Parch vs. Survived
graph = sns.factorplot(x='Parch', y='Survived', data=train, kind='bar')
graph = graph.set_ylabels('Survival Probability')
# Feature engineering
test['Fare'] = test['Fare'].fillna(test['Fare'].median()) # Removes 1 null value 
train['Embarked'] = train['Embarked'].fillna('S') # Removes 2 null values from embarked
# Combine train and test data into one dataset 
full_data = [train, test]

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    # Create new feature isAlone from family size
    dataset['isAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'isAlone'] = 1
    # Filling missing age values
    #age_mean = dataset['Age'].mean()
    #age_std = dataset['Age'].std()
    #age_null_count = dataset['Age'].isnull().sum()
    #age_null_random_list = np.random.randint(age_mean-age_std, age_mean+age_std, 
                                            #size=age_null_count)
    #dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    # Change the Age and Fare values from floats to int
    #dataset['Age'] = dataset['Age'].astype(int)
    dataset['Fare'] = dataset['Fare'].astype(int)
    # Mapping sex and embarked
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
    dataset['Embarked'] = dataset['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)

train.head()
# Plot a heatmap of feature correlations
plt.figure(figsize=(14, 12))
plt.title('Correlation of Features')
graph = sns.heatmap(train.corr(), cmap=plt.cm.bwr, annot=True,
                   fmt='.2f', square=True)
# Must fill missing age values 
# Age distribution
# Superimposing age densities
# Distribution shows peak between 0 and 5 corresponding to babies and young children

graph = sns.kdeplot(train['Age'][(train['Survived'] == 0) & (train['Age'].notnull())],
                    color='Red', shade=True)
graph = sns.kdeplot(train['Age'][(train['Survived'] == 1) & (train['Age'].notnull())],
                    color='Blue', shade=True)
graph.set_xlabel('Age')
graph.set_ylabel('Frequency')
graph = graph.legend(['Not Survived', 'Survived'])
# kde plot shows peak in survival rate corresponding to young age
# Furthermore, heatmap shows that age is negatively correlated with Pclass and FamilySize, so I will use these
# features to fill in missing values

age_nan_indices = list(dataset['Age'][np.isnan(dataset['Age'])].index)
for dataset in full_data:
    age_nan_indices = list(dataset['Age'][np.isnan(dataset['Age'])].index)
    for i in age_nan_indices:
        # Fill age with mean age of samples with similar Pclass and FamilySize
        age_pred = dataset['Age'][(dataset['Pclass'] == dataset.iloc[i]['Pclass']) & 
                                  (dataset['FamilySize'] == dataset.iloc[i]['FamilySize'])].mean()
        if np.isnan(age_pred):
            dataset['Age'].iloc[i] = dataset['Age'].mean()
        else:
            dataset['Age'].iloc[i] = age_pred
train['Name'].head()
# Extract title from name and create separate feature
def get_title(name):
    return name.split(',')[1].split('.')[0].strip()
# Create Title feature 
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Search for rare titles
plt.title('Title Counts')
graph = sns.countplot(x='Title', data=pd.concat([train, test]))
graph = plt.setp(graph.get_xticklabels(), rotation=45)
# Put all rare titles into a single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir',
                                                'Col', 'Capt', 'the Countess', 'Jonkheer',
                                                 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # Title mapping
    dataset['Title'] = dataset['Title'].map({'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 
                                             'Rare': 4}).astype(int)
# Men have low survival rate
graph = sns.factorplot(x='Title', y='Survived', data=train, kind='bar')
graph = graph.set_xticklabels(['Mr', 'Mrs', 'Miss', 'Master', 'Rare'])
# Replace cabin number by type 'X' if there is none
for dataset in full_data:
    # First letter of cabin indicates Desk which may be important
    dataset['Cabin'] = dataset['Cabin'].apply(lambda x: 'X' if pd.isnull(x) else x[0])
    
order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'X']
graph = sns.countplot(pd.concat([train, test])['Cabin'], order=order)
graph = sns.factorplot(x='Cabin', y='Survived', data=train, kind='bar', order=order)
graph = graph.set_ylabels('Survival Probability')
# Focus on whether or not passenger has a cabin
for dataset in full_data:
    dataset['HasCabin'] = dataset['Cabin'].apply(lambda x: 1 if x == 'X' else 0)
# Feature selection
features_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train = train.drop(features_to_drop, axis=1)
test = test.drop(features_to_drop, axis=1)
# Separate train and label features
Y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
seed = 42 # Reproducibility
# Stratification ensures that each fold is representative of all strata of the data
# Proportion of each class is approximately equal
# Generally better in terms of mean and variance compared to cross-validation
kfold = StratifiedKFold(n_splits=10)
classifier_names = ['RandomForest', 'AdaBoost', 'ExtraTrees', 'GradientBoosting', 
                    'LogisticRegression', 'KNeighbors', 'MLP', 'SVC', 'DecisionTree']
# Test different classifiers
classifiers = []
classifiers.append(RandomForestClassifier(random_state=seed))
classifiers.append(AdaBoostClassifier(random_state=seed))
classifiers.append(ExtraTreesClassifier(random_state=seed))
classifiers.append(GradientBoostingClassifier(random_state=seed))
classifiers.append(LogisticRegression(random_state=seed))
classifiers.append(KNeighborsClassifier())
classifiers.append(MLPClassifier(random_state=seed))
classifiers.append(SVC(random_state=seed))
classifiers.append(DecisionTreeClassifier(random_state=seed))

cv_results = []
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_train, y=Y_train, scoring='accuracy',
                                     cv=kfold))
# Display cross-validation results
for i, result in enumerate(cv_results):
    print('{}: {}%'.format(classifier_names[i], result.mean()*100))
# Hyperparameter tuning
# Choosing RandomForest AdaBoost, GradientBoosting, LinearDiscriminantAnalysis,
# and LogisticRegression classifiers for fine-tuning

# Random Forest Classifier
rf_clf = RandomForestClassifier()

rfc_param_grid = {'n_estimators': [100, 200, 500],
                'max_features': ['sqrt', 'log2', None],
                'min_samples_split': [2, 3, 5, 8],
                'min_samples_leaf': [2, 3, 5],
                'bootstrap': [True, False]}

gs_rfc = GridSearchCV(rf_clf, param_grid=rfc_param_grid, cv=kfold, scoring='accuracy', 
                      n_jobs=-1)
gs_rfc.fit(X_train, Y_train)

best_rf_clf = gs_rfc.best_estimator_

print(best_rf_clf)
print(gs_rfc.best_score_)
# AdaBoost Classifier
ada_clf = AdaBoostClassifier()

ada_param_grid = {'n_estimators': [100, 200, 500],
                'learning_rate': [0.1, 0.02, 0.5, 1],
                'algorithm': ['SAMME', 'SAMME.R']}

gs_ada = GridSearchCV(ada_clf, param_grid=ada_param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
gs_ada.fit(X_train, Y_train)

best_ada_clf = gs_ada.best_estimator_

print(best_ada_clf)
print(gs_ada.best_score_)
# ExtraTrees Classifier
extra_clf = ExtraTreesClassifier()

extra_param_grid = {'n_estimators': [100, 200, 500],
                   'max_features': ['sqrt', 'log2', None],
                   'min_samples_split': [2, 3, 5],
                   'min_samples_leaf': [2, 3, 5],
                   'bootstrap': [True, False]}

gs_extra = GridSearchCV(extra_clf, param_grid=extra_param_grid, cv=kfold, scoring='accuracy',
                       n_jobs=-1)
gs_extra.fit(X_train, Y_train)

best_extra_clf = gs_extra.best_estimator_

print(best_extra_clf)
print(gs_extra.best_score_)
# Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier()

gb_param_grid = {'loss': ['deviance'],
                'learning_rate': [0.001, 0.01, 0.02],
                'n_estimators': [200, 500, 800],
                'max_depth': [1, 3, 5, 8],
                'max_features': ['sqrt', 'log2', None]}

gs_gb = GridSearchCV(gb_clf, param_grid=gb_param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
gs_gb.fit(X_train, Y_train)

best_gb_clf = gs_gb.best_estimator_

print(best_gb_clf)
print(gs_gb.best_score_)
# Logistic Regression
logreg = LogisticRegression()

log_param_grid = {'C': [1.0],
                 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                 'max_iter': [10, 30, 50, 100]}

gs_log = GridSearchCV(logreg, param_grid=log_param_grid, cv=kfold, scoring='accuracy')
gs_log.fit(X_train, Y_train)

best_logreg = gs_log.best_estimator_

print(best_logreg)
print(gs_log.best_score_)
# Feature importances of tree based classifiers
names_classifiers = [('RandomForest', best_rf_clf), ('AdaBoost', best_ada_clf), 
                     ('GradientBoosting', best_gb_clf), ('ExtraTrees', best_extra_clf), 
                     ('LogisticRegression', best_logreg)]

fig, ax = plt.subplots(2, 2, figsize=(15, 15))

index = 0
for row in range(2):
    for col in range(2):
        name = names_classifiers[index][0]
        clf = names_classifiers[index][1]
        feature_importances = clf.feature_importances_
        indices = np.argsort(feature_importances)[::-1] # Least to most important features
        graph = sns.barplot(y=X_train.columns[indices], x=feature_importances[indices],
                           ax=ax[row][col])
        graph.set_xlabel('Relative Importance')
        graph.set_ylabel('Features')
        graph.set_title(name + ' feature importances')
        index += 1
# Title, sex, age, fare, and family size are most important features
# Might remove series of cabin features and create a 'HasCabin' feature
test_survived_rf_clf = pd.Series(best_rf_clf.predict(test), name='Rf')
test_survived_ada_clf = pd.Series(best_ada_clf.predict(test), name='Ada')
test_survived_gb_clf = pd.Series(best_gb_clf.predict(test), name='Gb')
test_survived_extra_clf = pd.Series(best_extra_clf.predict(test), name='Extra')
test_survived_logreg = pd.Series(best_logreg.predict(test), name='Logreg')

ensemble_results = pd.concat([test_survived_rf_clf, test_survived_ada_clf, test_survived_gb_clf,
                             test_survived_extra_clf, test_survived_logreg], axis=1)
# Compare the 5 classifiers with each other
# If the differences in predictions are small, then we can consider ensembling voting
plt.figure(figsize=(10, 10))
graph = sns.heatmap(ensemble_results.corr(), cmap=plt.cm.viridis, annot=True, square=True)
graph.set_title('Ensemble Results')
# Ensemble voting is useful because predictions are similar between classifiers
# Combines preditions from 5 classifiers
# 'soft' takes probability of each prediction into account
voting_clf = VotingClassifier(estimators=[('RandomForest', best_rf_clf), ('AdaBoost', best_ada_clf), 
                                          ('GradientBoosting', best_gb_clf), ('ExtraTrees', best_extra_clf), 
                                          ('LogisticRegression', best_logreg)], voting='soft')

voting_clf = voting_clf.fit(X_train, Y_train)
results = cross_val_score(voting_clf, X_train, y=Y_train, scoring='accuracy', cv=kfold)
print('Voting classifier: {}%'.format(results.mean()*100))
test_survived = voting_clf.predict(test)
submission = pd.DataFrame({'PassengerId': IDtest, 'Survived': test_survived})
submission.to_csv('titanic_ensemble_model.csv', index=False)