# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier, export_graphviz
X = np.linspace(-2, 2, 7)

y = X ** 3



plt.scatter(X, y)

plt.xlabel(r'$x$')

plt.ylabel(r'$y$');
line_x = np.linspace(-2, 2)

editedLine = [np.mean(y) for x in line_x]



plt.scatter(X, y)

plt.plot(line_x, editedLine)
editedLine = [np.mean(y[X<0]) 

              if x<0 

              else np.mean(y[X>=0]) 

              for x in line_x]

plt.scatter(X, y)

plt.plot(line_x, editedLine)
def regression_var_criterion(X, y, t):

    X_left = X[X < t]

    y_left = y[X < t]

    X_right = X[X >= t]

    y_right = y[X >= t]

    return np.var(y) - len(X_left)*np.var(y_left)/ len(X) - len(X_right)*np.var(y_right)/ len(X) 



cons_threshold = np.linspace(-1.9, 1.9)

newThreshold = [regression_var_criterion(X, y, threshold) for threshold in cons_threshold]

plt.plot(cons_threshold, newThreshold)

# X - пороговое значение

# Y - Критерий регрессии
def branching(x, X, y):

    if x >= 1.5:

        return np.mean(y[X >= 1.5])

    if x < 1.5 and x >= 0:

        return np.mean(y[(X >= 0) & (X < 1.5)])

    if x >= -1.5 and x < 0:

        return np.mean(y[(X < 0) & (X >= -1.5)])

    else:

        return np.mean(y[X < -1.5])

    

    

branching = [branching(x, X, y) for x in line_x]



plt.scatter(X, y);

plt.plot(line_x, branching);
df = pd.read_csv('/kaggle/input/mlbootcamp5_train.csv', delimiter=";")

df.head()
df['full_years'] = (df['age']/365).astype('int')

dfr = pd.get_dummies(df, columns=['cholesterol', 'gluc']).drop(['cardio'],axis=1)

dfr.head()
target = df['cardio']

X_train, X_valid, y_train, y_valid = train_test_split(dfr.values, target.values, test_size=.3, random_state=17)
Train = DecisionTreeClassifier(max_depth=3, random_state=17).fit(X_train, y_train)
import graphviz



data = export_graphviz(Train, feature_names=dfr.columns, out_file=None, filled=True)

graphviz.Source(data)
acc = np.sum(Train.predict(X_train)==y_train)/y_train.size

acc
tree_params = {'max_depth': list(range(2, 11))}

tree_grid = GridSearchCV(DecisionTreeClassifier(random_state=17), tree_params, cv=4) 

tree_grid.fit(X_train, y_train)
plt.plot(tree_params['max_depth'], tree_grid.cv_results_['mean_test_score'])

# X - Максимальная глубина

# Y - Средняя точность
print("Лучший параметр:", tree_grid.best_params_)

print("Лучший результат:", tree_grid.best_score_)

tured = accuracy_score(y_valid, tree_grid.predict(X_valid))

print("Точность несогласных:", tured)
(tured / acc ) * 100 * acc
sub_df = pd.DataFrame(df.smoke.copy())

sub_df['male']  = df.gender - 1



sub_df['age_(40-50)'] = ((df.full_years >= 40)&(df.full_years < 50)).astype('int')

sub_df['age_(50-55)'] = ((df.full_years >= 50)&(df.full_years < 55)).astype('int')

sub_df['age_(55-60)'] = ((df.full_years >= 55)&(df.full_years < 60)).astype('int')

sub_df['age_(60-65)'] = ((df.full_years >= 60)&(df.full_years < 65)).astype('int')



sub_df['ap_hi_(120-140)'] = ((df.ap_hi >= 120)&(df.ap_hi < 140)).astype('int')

sub_df['ap_hi_(140-160)'] = ((df.ap_hi >= 140)&(df.ap_hi < 160)).astype('int')

sub_df['ap_hi_(160-180)'] = ((df.ap_hi >= 160)&(df.ap_hi < 180)).astype('int')



sub_df['chol=1'] = (df.cholesterol == 1).astype('int')

sub_df['chol=2'] = (df.cholesterol == 2).astype('int')

sub_df['chol=3'] = (df.cholesterol == 3).astype('int')



sub_df.head()
result = DecisionTreeClassifier(max_depth=3, random_state=17).fit(sub_df, target)



data = export_graphviz(result, feature_names=sub_df.columns, out_file=None, filled=True)

graphviz.Source(data)