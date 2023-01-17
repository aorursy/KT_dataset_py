# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import GridSearchCV



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Загрузка данных

df = pd.read_csv('/kaggle/input/bank-marketing-dataset/bank.csv')

df.head()
df.keys()
#df.info()
import seaborn as sns

sns.catplot(x = "deposit", kind = "count", palette = "ch:.25", data = df)
# Выведем процентное соотношение

df['deposit'].value_counts(normalize=True)
# print(df['deposit'].shape) # Всего 11162 строк

df['deposit']
from sklearn. preprocessing import LabelEncoder



le = LabelEncoder()

le.fit(df.default)

df['default_le'] = le.transform(df.default)



le_1 = LabelEncoder()

le_1.fit(df.housing)

df['housing_le'] = le_1.transform(df.housing)



le_2 = LabelEncoder()

le_2.fit(df.housing)

df['loan_le'] = le_2.transform(df.loan)



le_3 = LabelEncoder()

le_3.fit(df.housing)

df['deposit_le'] = le_3.transform(df.deposit)



dct_1 = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec':12}

df['month_le'] = df['month'].map(dct_1)



df.head().T
df_1 = pd.read_csv('/kaggle/input/bank-marketing-dataset/bank.csv')

df_1['default'] = df['default_le']

df_1['housing'] = df['housing_le']

df_1['loan'] = df['loan_le']

df_1['deposit'] = df['deposit_le']

df_1['month'] = df['month_le']



# get_dummies

df_1 = pd.get_dummies(df_1, columns=['marital', 'education', 'contact', 'poutcome'])



df_1.head().T
#df_1.info()
#df_2 = df_1.drop(['job'], axis = 1)

df_2 = pd.get_dummies(df_1, columns=['job']) 



df_2.head().T
#df_2.info()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



# Создание X - вся таблица без target (deposit), а y - target (deposit).



y = df_2['deposit']

df_3 = df_2.drop('deposit', axis = 1)



X = df_3



X_new = scaler.fit_transform(X)

X_new
# Разбиение

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, 

                                                      y, 

                                                      test_size=0.25, 

                                                      random_state=20) 



#random_state. Controls the shuffling applied to the data before applying the split. 
print(X_train.shape, y_train.shape)

print( X_valid.shape, y_valid.shape)
# Обучение дерева решений

from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(max_depth = 3, random_state = 2019)

tree.fit(X_train, y_train)
# Визуализация

from sklearn.tree import export_graphviz



export_graphviz(tree, out_file='tree.dot', feature_names = X.columns)

print(open('tree.dot').read()) 
# Предсказания для валидационного множества

from sklearn.metrics import accuracy_score



y_pred = tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
from sklearn.model_selection import KFold

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
# # Кросс-валидация и подбор гиперпараметров



# tree_params = {'max_depth': np.arange(2, 38),

#                 'min_samples_split': np.arange(2, 38), 

#                 'min_samples_leaf': np.arange(2, 38), 

#                 'max_features': np.arange(2, 38)

#               }



# tree_grid = GridSearchCV(tree, tree_params, cv=kf, scoring='accuracy')

# tree_grid.fit(X_train, y_train)
# print("Estimator that was chosen by the search, i.e. estimator which gave highest score on the left out data:\n", 

#       tree_grid.best_estimator_)
# print("Score of best_estimator on the left out data:", tree_grid.best_score_)
# print("Parameter setting that gave the best results on the hold out data:\n", tree_grid.best_params_)
tree_param_max_depth = {'max_depth': np.arange(2, 38)}



tree_grid_param_1 = GridSearchCV(tree, tree_param_max_depth, cv=kf, scoring='accuracy')

tree_grid_param_1.fit(X_train, y_train)



print(tree_grid_param_1.best_score_, "\n", tree_grid_param_1.best_params_)
tree_param_min_samples_split = {'min_samples_split': np.arange(2, 38)}

tree_2 = DecisionTreeClassifier(max_depth = 15)

tree_grid_param_2 = GridSearchCV(tree_2, tree_param_min_samples_split, cv=kf, scoring='accuracy')

tree_grid_param_2.fit(X_train, y_train)



print(tree_grid_param_2.best_estimator_,"\n", tree_grid_param_2.best_score_, "\n", tree_grid_param_2.best_params_)
tree_param_min_samples_leaf = {'min_samples_leaf': np.arange(2, 38)}

tree_3 = DecisionTreeClassifier(max_depth = 15, min_samples_split = 37)

tree_grid_param_3 = GridSearchCV(tree_3, tree_param_min_samples_leaf, cv=kf, scoring='accuracy')

tree_grid_param_3.fit(X_train, y_train)



print(tree_grid_param_3.best_estimator_,"\n", tree_grid_param_3.best_score_, "\n", tree_grid_param_3.best_params_)
X_train.shape
tree_param_max_features = {'max_features': np.arange(2, 38)}

tree_4 = DecisionTreeClassifier(max_depth = 15, min_samples_split = 37, min_samples_leaf = 13)

tree_grid_param_4 = GridSearchCV(tree_4, tree_param_max_features, cv=kf, scoring='accuracy')

tree_grid_param_4.fit(X_train, y_train)



print(tree_grid_param_4.best_estimator_,"\n", tree_grid_param_4.best_score_, "\n", tree_grid_param_4.best_params_)
# Построим графики зависимости средней доли правильных ответов 

# (mean_test_score) от значений каждого из гиперпараметров.

import matplotlib.pyplot as plt



fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(8,13))



ax[0, 0].plot(tree_param_max_depth['max_depth'], tree_grid_param_1.cv_results_['mean_test_score'])

ax[0, 0].set_xlabel('max_depth')

ax[0, 0].set_ylabel('Mean accuracy on test set')



ax[0, 1].plot(tree_param_min_samples_split['min_samples_split'], tree_grid_param_2.cv_results_['mean_test_score'])

ax[0, 1].set_xlabel('min_samples_split')

ax[0, 1].set_ylabel('Mean accuracy on test set')



ax[1, 0].plot(tree_param_min_samples_leaf['min_samples_leaf'], tree_grid_param_3.cv_results_['mean_test_score'])

ax[1, 0].set_xlabel('min_samples_leaf')

ax[1, 0].set_ylabel('Mean accuracy on test set')



ax[1, 1].plot(tree_param_max_features['max_features'], tree_grid_param_4.cv_results_['mean_test_score'])

ax[1, 1].set_xlabel('max_features')

ax[1, 1].set_ylabel('Mean accuracy on test set')
best_tree = DecisionTreeClassifier(max_depth = 15, 

                                   max_features = 37, 

                                   min_samples_leaf = 13, 

                                   min_samples_split = 37)

best_tree.fit(X_train, y_train)
export_graphviz(best_tree, out_file='best_tree.dot', feature_names=X.columns)

print(open('best_tree.dot').read()) 
features = dict(zip(range(len(X.columns)), X.columns))



# Важность признаков

importances = best_tree.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = max(10, len(X.columns))

feature_indices = [ind for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features[feature_indices[f]], importances[indices[f]])
# Plot the impurity-based feature importances of the forest

plt.figure(figsize=(15,15))

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



#rf = RandomForestClassifier(n_estimators=100, random_state=2019, max_depth=6)



rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)



accuracy_score(y_valid, y_pred)
rf_param = {'n_estimators': [50, 100, 200, 300]

            #, 'max_depth':[5, 6, 7, 8, 9, 10]

           }

new_rf = GridSearchCV(rf, rf_param, cv=5, scoring='accuracy')

new_rf.fit(X_train, y_train)
new_rf.best_params_
new_rf.best_estimator_
new_rf.best_score_
rf_params_n_estimators = {'n_estimators': [50, 100, 200, 300]}

rf_grid_n_estimators = GridSearchCV(rf, rf_params_n_estimators, cv=kf, scoring='accuracy', n_jobs = -1)

rf_grid_n_estimators.fit(X_train, y_train)
print(rf_grid_n_estimators.best_score_, "\n", rf_grid_n_estimators.best_params_)
rf = RandomForestClassifier(n_estimators = 300)

rf_params_max_depth = {'max_depth': np.arange(2, 38)}

rf_grid_max_depth = GridSearchCV(rf, rf_params_max_depth, cv=kf, scoring='accuracy', n_jobs = -1)

rf_grid_max_depth.fit(X_train, y_train)
print(rf_grid_max_depth.best_estimator_,"\n", rf_grid_max_depth.best_score_, "\n", rf_grid_max_depth.best_params_)
rf = RandomForestClassifier(n_estimators = 300, max_depth=18)

rf_params_min_samples_split = {'min_samples_split': np.arange(2, 38)}

rf_grid_min_samples_split = GridSearchCV(rf, rf_params_min_samples_split, cv=kf, scoring='accuracy', n_jobs = -1)

rf_grid_min_samples_split.fit(X_train, y_train)
print(rf_grid_min_samples_split.best_estimator_,"\n", rf_grid_min_samples_split.best_score_, "\n", rf_grid_min_samples_split.best_params_)
rf = RandomForestClassifier(n_estimators = 300, max_depth = 18, min_samples_split = 4)

rf_params_min_samples_leaf = {'min_samples_leaf': np.arange(2, 38)}

rf_grid_min_samples_leaf = GridSearchCV(rf, rf_params_min_samples_leaf, cv=kf, scoring='accuracy', n_jobs = -1)

rf_grid_min_samples_leaf.fit(X_train, y_train)
print(rf_grid_min_samples_leaf.best_estimator_,"\n", rf_grid_min_samples_leaf.best_score_, "\n", rf_grid_min_samples_leaf.best_params_)
# Построим графики зависимости средней доли правильных ответов 

# (mean_test_score) от значений каждого из гиперпараметров.

import matplotlib.pyplot as plt



fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(8,13))



ax[0, 0].plot(rf_params_n_estimators['n_estimators'], rf_grid_n_estimators.cv_results_['mean_test_score'])

ax[0, 0].set_xlabel('n_estimators')

ax[0, 0].set_ylabel('Mean accuracy on test set')



ax[0, 1].plot(rf_params_max_depth['max_depth'], rf_grid_max_depth.cv_results_['mean_test_score'])

ax[0, 1].set_xlabel('max_depth')

ax[0, 1].set_ylabel('Mean accuracy on test set')



ax[1, 0].plot(rf_params_min_samples_split['min_samples_split'], rf_grid_min_samples_split.cv_results_['mean_test_score'])

ax[1, 0].set_xlabel('min_samples_split')

ax[1, 0].set_ylabel('Mean accuracy on test set')



ax[1, 1].plot(rf_params_min_samples_leaf['min_samples_leaf'], rf_grid_min_samples_leaf.cv_results_['mean_test_score'])

ax[1, 1].set_xlabel('min_samples_leaf')

ax[1, 1].set_ylabel('Mean accuracy on test set')
best_forest = RandomForestClassifier(max_depth = 18, 

                                     min_samples_leaf = 2, 

                                     min_samples_split = 4,

                                     n_estimators = 300)

best_forest.fit(X_train, y_train)
importances = best_forest.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = max(10, len(X.columns))

feature_indices = [ind for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features[feature_indices[f]], importances[indices[f]])
# Plot the impurity-based feature importances of the forest

plt.figure(figsize=(15,15))

plt.title("Feature importances")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),

               align="center")



ticks = plt.xticks(range(num_to_plot), 

                   feature_indices)



plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features[i]) for i in feature_indices]);
# Plot the impurity-based feature importances of the forest

plt.figure(figsize=(15,15))

plt.title("Feature importances")

bars = plt.bar(range(10), 

               importances[indices[:10]],

               color=([str(i/float(10+1)) for i in range(10)]),

               align="center")



ticks = plt.xticks(range(10), 

                   feature_indices)



plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features[i]) for i in feature_indices]);