# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import warnings

warnings.filterwarnings('ignore')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/mlcourse/telecom_churn.csv')

df.head()
df.info()
df.shape
df['International plan'] = df['International plan'].map({'Yes':1, "No": 0})

df['Voice mail plan'] = df['Voice mail plan'].map({'Yes':1, "No": 0})
df['Churn'] = df['Churn'].astype('int')
df.head()
state = df.pop('State')
X, y = df.drop('Churn', axis= 1), df['Churn']
X.shape, y.shape
from sklearn.model_selection import train_test_split
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size =.3, random_state= 42)
X_train.shape
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
pred_holdout = tree.predict(X_holdout)
pred_holdout.shape
accuracy_score(y_holdout, pred_holdout)
y.value_counts(normalize = True)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
params = {'max_depth': np.arange(2, 11),

          'min_samples_leaf': np.arange(1, 11)}
skf = StratifiedKFold(n_splits= 5, shuffle= True, random_state= 42)
best_tree = GridSearchCV(estimator=tree, param_grid=params, cv = skf, n_jobs=-1, verbose=1)
best_tree.fit(X_train, y_train)
best_tree.best_params_
best_tree.best_estimator_
best_tree.best_score_
pred_holdout_better = best_tree.predict(X_holdout)
accuracy_score(y_holdout, pred_holdout_better)
from sklearn.model_selection import cross_val_score

from tqdm import tqdm_notebook
cv_acc_by_depth = []

ho_acc_by_depth = []

max_depth_values = np.arange(2,10)

for max_depth in tqdm_notebook(max_depth_values):

    tree = DecisionTreeClassifier(random_state=42, max_depth= max_depth)

    val_scores = cross_val_score(estimator=tree, X=X_train, y=y_train, cv=skf)    

    cv_acc_by_depth.append(val_scores.mean())

    tree.fit(X_train, y_train)

    curr_pred = tree.predict(X_holdout)

    ho_acc_by_depth.append(accuracy_score(y_holdout, curr_pred))

    
cross_val_score(estimator=tree, X=X_train, y=y_train, cv=skf).mean()
plt.plot(max_depth_values, cv_acc_by_depth, label= 'CV', c='green')

plt.plot(max_depth_values, ho_acc_by_depth, label= 'Holdout', c='purple')

plt.legend();

plt.xlabel('Max Depth')

plt.ylabel('Accuracy')

plt.title('DT validation cureves for max-depth');

from sklearn.tree import export_graphviz
data = export_graphviz(decision_tree=best_tree.best_estimator_,

               out_file='tree.dot', filled=True,

               feature_names= df.drop('Churn' , axis=1).columns)
!ls *.dot
!cat tree.dot
import graphviz

data = export_graphviz(best_tree.best_estimator_,out_file=None,feature_names=df.drop('Churn', axis=1).columns,   

                         filled=True, rounded=True,  

                         special_characters=True)

graph = graphviz.Source(data)

graph