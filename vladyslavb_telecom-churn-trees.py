# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv')

df.head()
import matplotlib.pyplot as plt

import seaborn as sns
df.describe()
df.info()
df['churn'].value_counts(normalize = True)
df['churn'].value_counts(normalize = True).plot(kind = 'bar')
df["state"].value_counts()
df["area code"].value_counts()
df_1 = df.drop(["state","phone number"], axis = 1)
df_2 = df_1.drop(["churn", "area code"], axis = 1)
df_2["international plan"] = df_2['international plan'].map({"yes":1,"no":0})

df_2["voice mail plan"] = df_2['voice mail plan'].map({"yes":1,"no":0})

X = df_2



X
df_1["churn"] = df_1['churn'].map({False:0,True:1})

y = df_1['churn']
y
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 12)
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)
from sklearn.metrics import accuracy_score



y_pred = tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
from sklearn.model_selection import KFold

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
from sklearn.model_selection import GridSearchCV



tree_params_max_depth = {'max_depth': np.arange(2, 15)}

tree_grid = GridSearchCV(tree, tree_params_max_depth, cv=kf, scoring='accuracy')

tree_grid.fit(X_train, y_train)

tree_grid_cv_results_max_depth=tree_grid.cv_results_

tree_grid.best_params_
tree = DecisionTreeClassifier(max_depth=7)

tree_params_min_samples_split = {'min_samples_split': np.arange(2, 50)}

tree_grid = GridSearchCV(tree, tree_params_min_samples_split, cv=kf, scoring='accuracy')

tree_grid.fit(X_train, y_train)

tree_grid_cv_results_min_samples_split=tree_grid.cv_results_

tree_grid.best_params_
tree = DecisionTreeClassifier(max_depth=7, min_samples_split=15)

tree_params_min_samples_leaf = {'min_samples_leaf': np.arange(1, 50)}

tree_grid = GridSearchCV(tree, tree_params_min_samples_leaf, cv=kf, scoring='accuracy')

tree_grid.fit(X_train, y_train)

tree_grid_cv_results_min_samples_leaf=tree_grid.cv_results_

tree_grid.best_params_
tree = DecisionTreeClassifier(max_depth=7, min_samples_split=15, min_samples_leaf=2)

tree_params_max_features = {'max_features': np.arange(1, X.shape[1])}

tree_grid = GridSearchCV(tree, tree_params_max_features, cv=kf, scoring='accuracy') 

tree_grid.fit(X_train, y_train)

tree_grid_cv_results_max_features=tree_grid.cv_results_

tree_grid.best_params_
import matplotlib.pyplot as plt

fig, ((ax11,ax22),(ax33,ax44)) = plt.subplots(nrows=2, ncols=2, sharey=True,figsize=(10, 10))



ax11.plot(tree_params_max_depth['max_depth'], tree_grid_cv_results_max_depth['mean_test_score'])

ax11.set_xlabel('max_depth')

ax11.set_ylabel('Mean accuracy on test set')



ax22.plot(tree_params_min_samples_split['min_samples_split'], tree_grid_cv_results_min_samples_split['mean_test_score'])

ax22.set_xlabel('min_samples_split')



ax33.plot(tree_params_min_samples_leaf['min_samples_leaf'], tree_grid_cv_results_min_samples_leaf['mean_test_score'])

ax33.set_xlabel('min_samples_leaf')

ax33.set_ylabel('Mean accuracy on test set')



ax44.plot(tree_params_max_features['max_features'], tree_grid_cv_results_max_features['mean_test_score'])

ax44.set_xlabel('max_features')
best_tree = DecisionTreeClassifier(max_depth = 7, min_samples_split = 15, min_samples_leaf = 2, max_features = 16)

y_pred =best_tree.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)
from sklearn.tree import export_graphviz



export_graphviz(best_tree, out_file='best_tree.dot', feature_names=X.columns)

print(open('best_tree.dot').read())
features = dict(zip(range(len(X.columns)), X.columns))



# Важность признаков

importances = best_tree.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the tree

num_to_plot = len(X.columns)

feature_indices = [ind for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features[feature_indices[f]], importances[indices[f]])



plt.figure(figsize=(15,5))

plt.title("Feature importances")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),

               align="center")

ticks = plt.xticks(range(num_to_plot), 

                   feature_indices)

plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features[i]) for i in feature_indices]);
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)

accuracy_score(y_valid, y_pred)
rf = RandomForestClassifier()

rf_params_n_estimators = {'n_estimators': np.arange(50, 450, 50)}

rf_grid = GridSearchCV(rf, rf_params_n_estimators, cv=kf, scoring='accuracy')

rf_grid.fit(X_train, y_train)

rf_grid_cv_results_n_estimators = rf_grid.cv_results_

rf_grid.best_params_
rf = RandomForestClassifier(n_estimators = 100)

rf_params_max_depth = {'max_depth': np.arange(2, 15)}

rf_grid = GridSearchCV(rf, rf_params_max_depth, cv=kf, scoring='accuracy')

rf_grid.fit(X_train, y_train)

rf_grid_cv_results_max_depth = rf_grid.cv_results_

rf_grid.best_params_
rf = RandomForestClassifier(n_estimators = 100, max_depth = 13)

rf_params_min_samples_split = {'min_samples_split': np.arange(2, 50)}

rf_grid = GridSearchCV(rf, rf_params_min_samples_split, cv=kf, scoring='accuracy')

rf_grid.fit(X_train, y_train)

rf_grid_cv_results_min_samples_split = rf_grid.cv_results_

rf_grid.best_params_
rf = RandomForestClassifier(n_estimators = 100, max_depth = 13, min_samples_split = 2)

rf_params_min_samples_leaf = {'min_samples_leaf': np.arange(1, 50)}

rf_grid = GridSearchCV(rf, rf_params_min_samples_leaf, cv=kf, scoring='accuracy')

rf_grid.fit(X_train, y_train)

rf_grid_cv_results_min_samples_leaf = rf_grid.cv_results_

rf_grid.best_params_
rf = RandomForestClassifier(n_estimators = 100, max_depth = 13, min_samples_split = 2,min_samples_leaf = 1)

rf_params_max_features = {'max_features': np.arange(2, X.shape[1])}

rf_grid = GridSearchCV(rf, rf_params_max_features, cv=kf, scoring='accuracy')

rf_grid.fit(X_train, y_train)

rf_grid_cv_results_max_features = rf_grid.cv_results_

rf_grid.best_params_
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (5,5))



ax.plot(rf_params_n_estimators['n_estimators'], rf_grid_cv_results_n_estimators['mean_test_score'])

ax.set_xlabel('n_estimators')

ax.set_ylabel('Mean accuracy on test set')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (5,5))



ax.plot(rf_params_max_depth['max_depth'], rf_grid_cv_results_max_depth['mean_test_score'])

ax.set_xlabel('max_depth')

ax.set_ylabel('Mean accuracy on test set')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (5,5))



ax.plot(rf_params_min_samples_split['min_samples_split'], rf_grid_cv_results_min_samples_split['mean_test_score'])

ax.set_xlabel('min_samples_split')

ax.set_ylabel('Mean accuracy on test set')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (5,5))



ax.plot(rf_params_min_samples_leaf['min_samples_leaf'], rf_grid_cv_results_min_samples_leaf['mean_test_score'])

ax.set_xlabel('min_samples_leaf')

ax.set_ylabel('Mean accuracy on test set')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (5,5))



ax.plot(rf_params_max_features['max_features'], rf_grid_cv_results_max_features['mean_test_score'])

ax.set_xlabel('max_features')

ax.set_ylabel('Mean accuracy on test set')
best_rf = RandomForestClassifier(n_estimators = 100, max_depth = 13, min_samples_split = 2, min_samples_leaf = 1, max_features = 4)

y_pred = best_rf.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)
features = dict(zip(range(len(X.columns)), X.columns))



# Важность признаков

importances = best_rf.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = 10

feature_indices = [ind for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features[feature_indices[f]], importances[indices[f]])



plt.figure(figsize=(15,5))

plt.title("Feature importances")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),

               align="center")

ticks = plt.xticks(range(num_to_plot), 

                   feature_indices)

plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features[i]) for i in feature_indices]);