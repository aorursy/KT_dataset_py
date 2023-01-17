import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC



from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix



sns.set_style('darkgrid')

%matplotlib inline
edu_df = pd.read_csv('../input/xAPI-Edu-Data/xAPI-Edu-Data.csv', engine='python')
edu_df.head()
print("the shape of our dataset is {}".format(edu_df.shape))
edu_df.info()
edu_df.nunique()
edu_df['NationalITy'].unique()
edu_df['PlaceofBirth'].unique()
edu_df.drop('NationalITy', axis=1, inplace=True)
edu_df.columns
plt.figure(figsize=(14,8))

sns.countplot('PlaceofBirth', data=edu_df)

plt.xlabel('nationality')

plt.ylabel('number of students');
sorted_place = edu_df['PlaceofBirth'].value_counts().index
plt.figure(figsize=(14,8))

sns.countplot('PlaceofBirth', hue='gender', data=edu_df, order=sorted_place)

plt.xlabel('nationality')

plt.ylabel('number of students')

plt.legend(loc='upper right');
fig, ax = plt.subplots(1,2, figsize=(14,8))

sns.countplot('StageID', data=edu_df, ax=ax[0])

sns.countplot('StageID', hue='gender', data=edu_df, ax=ax[1]);
labels = edu_df['Topic'].unique()



plt.figure(figsize=(14,8))

f = sns.countplot('Topic', data=edu_df)

f.set_xticklabels(labels=labels, rotation=120);
fig, ax = plt.subplots(1,2, figsize=(14,8))

sns.countplot('Topic', hue='gender', data=edu_df, ax=ax[0])

sns.countplot('Topic', hue='StageID', data=edu_df, ax=ax[1])

ax[0].set_ylabel("number of students")

ax[1].set_ylabel("number of students")

ax[0].set_xticklabels(labels=labels, rotation=120)

ax[1].set_xticklabels(labels=labels, rotation=120)

ax[1].legend(loc='upper right');
topic_order = edu_df['Topic'].value_counts().index
fig, ax = plt.subplots(2,2, figsize=(12,8))

sns.countplot(edu_df.query('PlaceofBirth=="KuwaIT"')['Topic'], ax=ax[0,0], order=topic_order)

sns.countplot(edu_df.query('PlaceofBirth=="Jordan"')['Topic'], ax=ax[0,1], order=topic_order)

sns.countplot(edu_df.query('PlaceofBirth=="Iraq"')['Topic'], ax=ax[1,0], order=topic_order)

sns.countplot(edu_df.query('PlaceofBirth=="USA"')['Topic'], ax=ax[1,1], order=topic_order)

ax[0,0].set_xticklabels(labels=labels, rotation=120)

ax[0,1].set_xticklabels(labels=labels, rotation=120)

ax[1,0].set_xticklabels(labels=labels, rotation=120)

ax[1,1].set_xticklabels(labels=labels, rotation=120)

ax[0,0].set_title('Kuwait')

ax[0,1].set_title('Jordan')

ax[1,0].set_title('Iraq')

ax[1,1].set_title('USA')

plt.tight_layout();
fig, ax = plt.subplots(1, 2, figsize=(14,8))

sns.countplot('Class', data=edu_df, ax=ax[0])

sns.countplot('Class', hue='StudentAbsenceDays', data=edu_df, ax=ax[1]);
fig, ax = plt.subplots(2, 2, figsize=(14,8))

sns.barplot(x='Class', y='raisedhands', data=edu_df, ax=ax[0,0])

sns.barplot(x='Class', y='VisITedResources', data=edu_df, ax=ax[0,1])

sns.barplot(x='Class', y='Discussion', data=edu_df, ax=ax[1,0])

sns.barplot(x='Class', y='AnnouncementsView', data=edu_df, ax=ax[1,1])

plt.tight_layout();
plt.figure(figsize=(14,8))

sns.countplot('Class', hue='gender', data=edu_df);
plt.figure(figsize=(12,8))

sns.countplot('gender', hue='StudentAbsenceDays', data=edu_df);
fig, ax = plt.subplots(2, 2, figsize=(14,8))

sns.boxplot(x='gender', y='raisedhands', data=edu_df, ax=ax[0,0])

sns.boxplot(x='gender', y='VisITedResources', data=edu_df, ax=ax[0,1])

sns.boxplot(x='gender', y='Discussion', data=edu_df, ax=ax[1,0])

sns.boxplot(x='gender', y='AnnouncementsView', data=edu_df, ax=ax[1,1])

plt.tight_layout();
fig, ax = plt.subplots(1,3, figsize=(14,6))

sns.countplot('SectionID', hue='Class', data=edu_df, ax=ax[0])

sns.countplot('StageID', hue='Class', data=edu_df, ax=ax[1])

sns.countplot('Semester', hue='Class', data=edu_df, ax=ax[2]);
plt.figure(figsize=(14,8))

f = sns.countplot('Topic', hue='Class', data=edu_df)

f.set_xticklabels(labels=labels, rotation=120);
plt.figure(figsize=(14,8))

sns.countplot('Topic', hue='ParentschoolSatisfaction', data=edu_df);
X = edu_df.drop('Class', axis=1)

X.head()
y = edu_df['Class']
X = pd.get_dummies(X, drop_first=True)

X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("the shape of the training set is {}".format(X_train.shape))

print("the shape of the test set is {}".format(X_test.shape))
sns.countplot('Class', data=edu_df);
# consider n_neighbors in the range (1,50)

n_neighbors = range(50)

accuracy_list = []



for n in n_neighbors:

    knn = KNeighborsClassifier(n_neighbors=n+1)

    # fit on the training set

    knn.fit(X_train, y_train)

    # predict on the test set

    pred = knn.predict(X_test)

    # calculate the accuracy score on the test set

    accuracy = accuracy_score(y_test, pred)

    # store the value

    accuracy_list.append(accuracy)
plt.plot(range(1,51), accuracy_list)

plt.xlabel('number of neighbors')

plt.ylabel('accuracy score')
knn = KNeighborsClassifier(n_neighbors=12)

knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

knn_accuracy = accuracy_score(y_true=y_test, y_pred=knn_pred)

print("The accuracy score for KNN algorithm is {}".format(knn_accuracy))
# consider n_estimators in the range (10,200)

n_estimators = range(0,210,10)

accuracy_list = []



for n in n_estimators:

    rf = RandomForestClassifier(n_estimators=n+1)

    # fit on the training set

    rf.fit(X_train, y_train)

    # predict on the test set

    pred = rf.predict(X_test)

    # calculate the accuracy score on the test set

    accuracy = accuracy_score(y_test, pred)

    # store the value

    accuracy_list.append(accuracy)
plt.plot(range(0,210,10), accuracy_list)

plt.xlabel('number of trees')

plt.ylabel('accuracy score')
rf = RandomForestClassifier(n_estimators=110)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_true=y_test, y_pred=rf_pred)

print("The accuracy score for random forest algorithm is {}".format(rf_accuracy))
model_list = [LogisticRegression(max_iter=200), SVC()]



param_dict = [

              {'C': [0.1, 0.5, 1.0]},

              {'kernel': ['rbf', 'linear', 'sigmoid', 'poly']}

             ]
for model, param in zip(model_list, param_dict):

    grid_cv = GridSearchCV(estimator=model, param_grid=param, verbose=3)

    print("the current estimator is : {}".format(model))

    grid_cv.fit(X_train, y_train)

    print()

    print("the best score on the training set is : {} for the parameter {}".format(grid_cv.best_score_, 

                                                                                   grid_cv.best_params_))

    print()
print("The overall best estimator is {} with parameter {} and it produces a cross-validated score {} on the training set".

      format(grid_cv.best_estimator_, grid_cv.best_params_, grid_cv.best_score_))
pred = grid_cv.predict(X_test)
print(classification_report(y_true=y_test, y_pred=pred))