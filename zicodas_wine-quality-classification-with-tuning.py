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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
import warnings # supress warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.shape
df.columns
df.isnull().any()
df.describe()
sns.boxplot(df['residual sugar'])
from scipy import stats
z = np.abs(stats.zscore(df))
red_wines = df[(z < 3).all(axis=1)]
red_wines.shape
red_wines.describe()
sns.boxplot(red_wines['residual sugar'])
sns.distplot(red_wines['residual sugar'])
sns.countplot(x='quality', data=red_wines)
plt.subplots(figsize=(15, 10))
sns.heatmap(red_wines.corr(), annot = True, cmap = 'coolwarm')
red_wines['quality'].value_counts()
X = red_wines.drop('quality',axis=1)
y = red_wines['quality']
y.value_counts()
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
from collections import Counter
counter = Counter(y)
for k,v in counter.items():
	per = v / len(y) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
plt.bar(counter.keys(), counter.values())
plt.show()
X.shape,y.shape
X = StandardScaler().fit(X).transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)
print ('Train set:', X_train.shape, y_train.shape)
print ('Test set:', X_test.shape, y_test.shape)
# Instantiate the machine learning classifiers
log_model = LogisticRegression()
KNN_model = KNeighborsClassifier()
SVC_model = LinearSVC(dual=False)
Gauss_model = GaussianProcessClassifier()
Decision_model = DecisionTreeClassifier()
RandFor_model = RandomForestClassifier()
Ada_model = AdaBoostClassifier()
GaussNB_model = GaussianNB()
QDA_model = QuadraticDiscriminantAnalysis()
mlp_model = MLPClassifier()
# Define dictionary with performance metrics
scoring = {'accuracy':make_scorer(accuracy_score), 
           'precision':make_scorer(precision_score,average='macro'),
           'recall':make_scorer(recall_score,average='macro'), 
           'f1_score':make_scorer(f1_score,average='macro')}
log = cross_validate(log_model,X,y,cv=5,scoring=scoring)
knn = cross_validate(KNN_model,X,y,cv=5,scoring=scoring)
svc = cross_validate(SVC_model,X,y,cv=5,scoring=scoring)
gauss = cross_validate(Gauss_model,X,y,cv=5,scoring=scoring)
decision = cross_validate(Decision_model,X,y,cv=5,scoring=scoring)
random = cross_validate(RandFor_model,X,y,cv=5,scoring=scoring)
ada = cross_validate(Ada_model,X,y,cv=5,scoring=scoring)
gaussNB = cross_validate(GaussNB_model,X,y,cv=5,scoring=scoring)
qda = cross_validate(QDA_model,X,y,cv=5,scoring=scoring)
mlp = cross_validate(mlp_model,X,y,cv=5,scoring=scoring)
models_scores_table_new = pd.DataFrame({'Logistic Regression':[log['test_accuracy'].mean(),
                                                               log['test_precision'].mean(),
                                                               log['test_recall'].mean(),
                                                               log['test_f1_score'].mean()],
                                                           
                                      'KNeighbors Classifier':[knn['test_accuracy'].mean(),
															  knn['test_precision'].mean(),
                                                              knn['test_recall'].mean(),
                                                              knn['test_f1_score'].mean()],
                                       
                                      'SVC Classifer':[svc['test_accuracy'].mean(),
                                                       svc['test_precision'].mean(),
                                                       svc['test_recall'].mean(),
                                                       svc['test_f1_score'].mean()],
                                       
                                      'GaussianProcess Classifier':[gauss['test_accuracy'].mean(),
                                                                   gauss['test_precision'].mean(),
                                                                   gauss['test_recall'].mean(),
                                                                   gauss['test_f1_score'].mean()],
															  
									  
																   
                                      'DecisionTree Classifier':[decision['test_accuracy'].mean(),
                                                                   decision['test_precision'].mean(),
                                                                   decision['test_recall'].mean(),
                                                                   decision['test_f1_score'].mean()],
																   
									  'RandomForest Classifier':[random['test_accuracy'].mean(),
                                                                   random['test_precision'].mean(),
                                                                   random['test_recall'].mean(),
                                                                   random['test_f1_score'].mean()],
																   
									  'AdaBoost Classifier':[ada['test_accuracy'].mean(),
                                                                   ada['test_precision'].mean(),
                                                                   ada['test_recall'].mean(),
                                                                   ada['test_f1_score'].mean()],
																   
									  'GaussianNB':[gaussNB['test_accuracy'].mean(),
                                                                   gaussNB['test_precision'].mean(),
                                                                   gaussNB['test_recall'].mean(),
                                                                   gaussNB['test_f1_score'].mean()],
																   
									  'Quadratic Discriminant Analysis':[qda['test_accuracy'].mean(),
                                                                   qda['test_precision'].mean(),
                                                                   qda['test_recall'].mean(),
                                                                   qda['test_f1_score'].mean()],
                                      'MLP Classifer':[mlp['test_accuracy'].mean(),
                                                                   mlp['test_precision'].mean(),
                                                                   mlp['test_recall'].mean(),
                                                                   mlp['test_f1_score'].mean()]},
									  
                                       index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
models_scores_table_new
models_scores_table_new['Best Score'] = models_scores_table_new.idxmax(axis=1)
models_scores_table_new
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
RandFor_model.fit(X_train,y_train)
y_pred = RandFor_model.predict(X_test)
y_test.shape,y_pred.shape
confusion_matrix(y_test,y_pred)
multilabel_confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))
from pprint import pprint
print('Parameters currently in use:\n')
pprint(RandFor_model.get_params())
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)
# Use the random grid to search for best hyperparameters
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = RandFor_model, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random
# Fit the random search model
rf_random.fit(X_train,y_train)
rf_random.best_params_
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
base_model = RandomForestClassifier()
base_model.fit(X_train,y_train)
base_accuracy = evaluate(base_model,X_test,y_test)
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test,y_test)
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
from sklearn.model_selection import GridSearchCV


param_grid = {
    'bootstrap': [False],
    'max_depth': [60, 70, 80, 90, 100, 110],
    'max_features': ['auto'],
    'min_samples_leaf': [1,2,3],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [2000]
}

# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search
grid_search.fit(X_train, y_train)
grid_search.best_params_
best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_test, y_test)
print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
