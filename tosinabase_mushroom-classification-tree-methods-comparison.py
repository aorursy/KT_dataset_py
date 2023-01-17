import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, StratifiedKFold
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

df.head()
pd.concat({'unique values': df.apply(pd.unique), 'number of unique values': df.nunique()}, axis=1)
df.drop('veil-type', axis=1, inplace=True)
# les is the dict of LabelEncoder objects created for each dataframe column

les = {col: LabelEncoder() for col in df.columns}



for col in les:

    df[col] = les[col].fit_transform(df[col])

    

df.head()
target = df['class'].values

data = df.drop('class', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=17, shuffle=False)
kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state=17)



def plot_feature_importances(tree_grid, n_cols=10):

    f_imp = pd.DataFrame({'feature': list(df.drop('class', axis=1).columns), 

                          'importance': tree_grid.best_estimator_.feature_importances_}

                        ).sort_values('importance', ascending = False).reset_index()

    f_imp['importance_normalized'] = f_imp['importance'] / f_imp['importance'].sum()

    

    ax = plt.subplot()

    ax.barh(list(reversed(list(f_imp.index[:n_cols]))), 

            f_imp['importance_normalized'].head(n_cols), 

            align = 'center', edgecolor = 'k')

    ax.set_yticks(list(reversed(list(f_imp.index[:n_cols]))))

    ax.set_yticklabels(f_imp['feature'].head(n_cols))

    plt.show()



def train_by_gridsearch(train_set, test_set, clf, params, cv=kf, n_cols=10):

    tree_grid = GridSearchCV(clf, params, cv=cv)

    tree_grid.fit(train_set, test_set)

    

    plot_feature_importances(tree_grid, n_cols)

    

    return tree_grid.best_estimator_, tree_grid



def print_info(clf, greed):

    train_score = accuracy_score(clf.predict(X_train), y_train)

    test_score = accuracy_score(clf.predict(X_test), y_test)

    best_params = greed.best_params_

    print(f'Train Score = {train_score}')

    print(f'Test Score = {test_score}')

    print(f'Best Params:', best_params)
from sklearn.tree import DecisionTreeClassifier

dtc_params = {'max_depth': list(range(1, 11)),

              'min_samples_split': [2, 3, 4, 5],

              'min_samples_leaf': [2, 3, 4, 5],

              'max_leaf_nodes': [5, 10, 15, 20, 25, 30, 50],

             }

dtc, dtc_greed = train_by_gridsearch(X_train, y_train, clf=DecisionTreeClassifier(random_state=17), params=dtc_params)

print_info(dtc, dtc_greed)
from sklearn.ensemble import AdaBoostClassifier

abc_params = {'n_estimators': [50, 100, 150], 

              'base_estimator': [

                  DecisionTreeClassifier(max_depth=1), 

                  dtc

              ]}

abc, abc_greed = train_by_gridsearch(X_train, y_train, clf=AdaBoostClassifier(), params=abc_params)



print_info(abc, abc_greed)
from sklearn.ensemble import RandomForestClassifier

rfc_params = {'max_depth': list(range(2, 11)) + [None],

              'n_estimators': [15, 25, 50, 75, 100],

             }

rfc, rfc_greed = train_by_gridsearch(X_train, y_train, clf=RandomForestClassifier(random_state=17), params=rfc_params)



print_info(rfc, rfc_greed)
from sklearn.ensemble import GradientBoostingClassifier

gbc_params = {

    'max_depth': list(range(2, 11)) + [None],

    'n_estimators': [50, 75, 100, 150, 175, 200],

}

gbc, gbc_greed = train_by_gridsearch(X_train, y_train, clf=GradientBoostingClassifier(random_state=17), params=gbc_params)



print_info(gbc, gbc_greed)
from sklearn.ensemble import ExtraTreesClassifier

etc_params = {'max_depth': list(range(2, 11)) + [None],

             'max_leaf_nodes': [10, 15, 20, 30, 50],

             }

etc, etc_greed = train_by_gridsearch(X_train, y_train, clf=ExtraTreesClassifier(random_state=17), params=etc_params, n_cols=15)

print_info(etc, etc_greed)