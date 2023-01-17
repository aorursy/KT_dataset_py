# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



warnings.filterwarnings('ignore')



df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
df.head(10)
df['Glucose'] = np.where(df['Glucose'] == 0, df['Glucose'].median(), df['Glucose'])

df['Insulin'] = np.where(df['Insulin'] == 0, df['Insulin'].median(), df['Insulin'])

df['SkinThickness'] = np.where(df['SkinThickness'] == 0, df['SkinThickness'].median(), df['SkinThickness'])
X = df.drop('Outcome', axis = 1)

y = df['Outcome']
pd.DataFrame(X, columns = df.columns[:-1])
print(X.head(10))
print(y.head(10))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state=0)
from sklearn.ensemble import RandomForestClassifier
rfClassifier = RandomForestClassifier(n_estimators= 10).fit(X_train, y_train)

prediction = rfClassifier.predict(X_test)
y.value_counts()
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print("Confusion Matrix\n", confusion_matrix(y_test, prediction))
print("\nAccuracy Score:", accuracy_score(y_test, prediction))
print("\nClassification Report\n",classification_report(y_test, prediction))
# Manual Hyperparameter Tuning

model = RandomForestClassifier(n_estimators=300, criterion='entropy', 

                               max_features='sqrt', min_samples_leaf=10, random_state=0).fit(X_train, y_train)

prediction = model.predict(X_test)

print("Confusion Matrix\n", confusion_matrix(y_test, prediction))

print("\nAccuracy Score:", accuracy_score(y_test, prediction))

print("\nClassification Report\n", classification_report(y_test, prediction))
#Randomized SearchCV

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num =10)]

max_features = ['auto', 'sqrt', 'log2']

max_depth = [int(x) for x in np.linspace(10, 1000, 100)]

min_samples_split = [1,3,4,5,7,9]

min_samples_leaf = [1,2,4,6,8,10]

random_grid = {'n_estimators':n_estimators, 'max_features':max_features, 'max_depth':max_depth,'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'criterion': ['entropy', 'gini']}

print(random_grid)
rf = RandomForestClassifier()
rf_randomcv = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, 

                                 random_state=100, n_jobs=-1)

rf_randomcv.fit(X_train, y_train)
rf_randomcv.best_params_
best_random_grid = rf_randomcv.best_estimator_
model_grid = best_random_grid.fit(X_train, y_train)

prediction = model_grid.predict(X_test)

print("Confusion Matrix\n", confusion_matrix(y_test, prediction))

print("\nAccuracy Score:", accuracy_score(y_test, prediction))

print("\nClassification Report\n", classification_report(y_test, prediction))
from sklearn.model_selection import GridSearchCV
param_grid = {'criterion': [rf_randomcv.best_params_['criterion']], 'max_depth': [rf_randomcv.best_params_['max_depth']],

                'max_features':[rf_randomcv.best_params_['max_features']], 

                'min_samples_leaf':[rf_randomcv.best_params_['min_samples_leaf'], rf_randomcv.best_params_['min_samples_leaf']+2, rf_randomcv.best_params_['min_samples_leaf']+4],

                'min_samples_split':[rf_randomcv.best_params_['min_samples_split']-2, rf_randomcv.best_params_['min_samples_split']-1, rf_randomcv.best_params_['min_samples_split'],

                                    rf_randomcv.best_params_['min_samples_split']+2, rf_randomcv.best_params_['min_samples_split']+1], 

                'n_estimators':[rf_randomcv.best_params_['n_estimators']-200, rf_randomcv.best_params_['n_estimators']-100, 

                               rf_randomcv.best_params_['n_estimators'],

                               rf_randomcv.best_params_['n_estimators']+100, rf_randomcv.best_params_['n_estimators']+200]}
param_grid
rf = RandomForestClassifier()
gridsearchcv = GridSearchCV(estimator=rf, param_grid=param_grid,cv=10, verbose=2, n_jobs=-1)

gridsearchcv.fit(X_train, y_train)
gridsearchcv.best_params_
gridsearchcv.best_estimator_
model_grid_search = gridsearchcv.best_estimator_.fit(X_train, y_train)

prediction = model_grid_search.predict(X_test)

print("Confusion Matrix\n", confusion_matrix(y_test, prediction))

print("\nAccuracy Score:", accuracy_score(y_test, prediction))

print("\nClassification Report\n", classification_report(y_test, prediction))
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
space = {'criterion': hp.choice('criterion', ['entropy', 'gini']), 

        'max_depth': hp.quniform('max_depth', 10,1200, 10),

        'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', 'None']),

        'min_samples_leaf': hp.uniform('min_samples_leaf', 0,0.5),

        'min_samples_split': hp.uniform('min_samples_split', 0,1),

        'n_estimators': hp.choice('n_estimators',[10,50,300,750,1200,1300,1500])}
def objective(space):

    model = RandomForestClassifier(criterion=space['criterion'], max_depth=space['max_depth'], max_features=space['max_features'],

                                   min_samples_leaf=space['min_samples_leaf'], min_samples_split=space['min_samples_split'],

                                   n_estimators=space['n_estimators']

                                  )

    accuracy_score = cross_val_score(model, X_train, y_train, cv = 5).mean()

    return {'loss': -accuracy_score, 'status':STATUS_OK}
from sklearn.model_selection import cross_val_score
trials = Trials()

best = fmin(fn=objective, space=space, algo = tpe.suggest, max_evals=80, trials=trials)
best
crit = {0:'entropy', 1:'gini'}

feat = {0:'auto', 1:'sqrt', 2:'log2', 3:'None'}

est = {0:10, 1:50, 2:300, 3:750, 4:1200, 5:1300, 6:1500}
print(crit[best['criterion']])
print(feat[best['max_features']])
print(est[best['n_estimators']])
trained_forest = RandomForestClassifier(criterion=crit[best['criterion']], max_depth=best['max_depth'], 

                                        max_features=feat[best['max_features']], min_samples_leaf=best['min_samples_leaf'],

                                       min_samples_split=best['min_samples_split'], n_estimators=est[best['n_estimators']]).fit(X_train,y_train)

predictions_forest = trained_forest.predict(X_test)

print("Confusion Matrix\n", confusion_matrix(y_test, predictions_forest))

print("\nAccuracy Score:", accuracy_score(y_test, predictions_forest))

print("\nClassification Report\n", classification_report(y_test, predictions_forest))

acc5 = accuracy_score(y_test, predictions_forest)
import numpy as np

from sklearn.model_selection import RandomizedSearchCV



n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num =10)]

max_features = ['auto', 'sqrt', 'log2']

max_depth = [int(x) for x in np.linspace(10, 1000, 100)]

min_samples_split = [2,5,10,14]

min_samples_leaf = [1,2,4,6,8]

param = {'n_estimators':n_estimators, 'max_features':max_features, 'max_depth':max_depth,'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'criterion': ['entropy', 'gini']}

print(param)
from tpot import TPOTClassifier
tpot_classifier = TPOTClassifier(generations=5, population_size=24, offspring_size=12, verbosity= 2,

                                early_stop=12, config_dict={'sklearn.ensemble.RandomForestClassifier':param},

                                cv=4, scoring='accuracy')
tpot_classifier.fit(X_train, y_train)
accuracy = tpot_classifier.score(X_test, y_test)

print(accuracy)
import optuna

import sklearn.svm
def objective(trials):

    classifier = trials.suggest_categorical('classifier', ['RandomForest', 'SVC'])

    if classifier == 'RandomForest':

        n_estimators = trials.suggest_int('n_estimators', 200, 2000, 10)

        max_depth = int(trials.suggest_float('max_depth', 10, 100, log = True))

        clf = sklearn.ensemble.RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)

    else:

        c = trials.suggest_float('svc_c', 1e-10, 1e10, log = True)

        clf = sklearn.svm.SVC(C=c, gamma='auto')

    return sklearn.model_selection.cross_val_score(clf, X_train, y_train, n_jobs=-1, cv=3).mean()

study = optuna.create_study(direction = 'maximize')

study.optimize(objective, n_trials=100)
trial = study.best_trial
print('Accuracy {}'.format(trial.value))

print('Best Hyperparameters: {}'.format(trial.params))
trial
study.best_params
rf = RandomForestClassifier(n_estimators = study.best_params['n_estimators'], max_depth = study.best_params['max_depth'])
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))