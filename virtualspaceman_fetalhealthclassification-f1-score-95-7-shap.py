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
path = '/kaggle/input/fetal-health-classification/fetal_health.csv'

df = pd.read_csv(path)
df.head()
print(df.fetal_health.value_counts())

'''

1) Normal
2) Suspect
3) Pathological

'''
df.severe_decelerations.value_counts()
df[df.severe_decelerations == 0.001]
df_corr = df.corr()
df_corr = df_corr.style.background_gradient(cmap='RdBu')
df_corr
labels = df['fetal_health'] - 1

df.drop(columns=['fetal_health'], inplace=True)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate, RandomizedSearchCV

from sklearn.metrics import auc, accuracy_score, confusion_matrix

import xgboost as xgb
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.25) 
class Experiment():
    def __init__(self, clf, scoring = ['accuracy'] ):
        self.clf = clf
        self.scoring = scoring
        self.scores= None
    
    
    def run(self, X_train, y_train, X_test, y_test, cv_splits=5, params_to_tune= {}):
        
        bspaces = 30
        print(bspaces*'*', f"Results for {self.clf.__class__.__name__}", bspaces*'*')
        if len(params_to_tune) > 0:
            search = RandomizedSearchCV(self.clf, params_to_tune, n_jobs=-1).fit(X_train, y_train)
            self.clf = search.best_estimator_
            
            print(f"Estimator best params: {search.best_params_}")
        
        scores = cross_validate(estimator=self.clf, 
                                X=X_train, y=y_train,
                                cv=cv_splits,
                                scoring=self.scoring)
        
        if isinstance(scores, dict):
            for metric in scores.keys():
                scores[metric] = np.mean(scores[metric])
        else:
            scores = np.mean(scores)
        
        self.scores = scores
        print(f"Avg. Validation scores: \n {scores}")
        
        print(3*bspaces*'*')
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint

scoring =  ['precision_micro', 'recall_micro', 'accuracy']

clf = DecisionTreeClassifier(random_state=42)

param_dist = {"max_depth": [3, 5, 8, 15, 2-0, 30, 50, 60, None],
              "max_features": randint(1, 30),
              "min_samples_leaf": randint(1, 30),
              "criterion": ["gini", "entropy"]}


exp_decision_tree = Experiment(clf,scoring)
exp_decision_tree.run(X_train, y_train, X_test, y_test, params_to_tune=param_dist)
from sklearn.ensemble import RandomForestClassifier

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 20]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 7, 10]

# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
param_dist = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


clf = RandomForestClassifier(random_state=42)
exp_random_forest = Experiment(clf,scoring)
exp_random_forest.run(X_train, y_train, X_test, y_test, params_to_tune=param_dist)
y_pred = exp_random_forest.clf.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred, average='micro')

# Transform to df for easier plotting
cm_df = pd.DataFrame(cm.astype(int),
                     index = ['Normal', 'Suspect', 'Pathological'], 
                     columns = ['Normal', 'Suspect', 'Pathological'])

plt.figure(figsize=(5.5,4), dpi=150)
sns.heatmap(cm_df, annot=True)
plt.title('Model \n F1-Score:{0:.3f}'.format(f1))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
!pip3 install shap
import shap

explainer = shap.TreeExplainer(exp_random_forest.clf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values[0], X_test)
shap.summary_plot(shap_values[1], X_test)
shap.summary_plot(shap_values[2], X_test)
