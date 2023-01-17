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
df = pd.read_csv('../input/titanic/train.csv')
df.head()
df.info()
df['Sex'].value_counts()
cols_to_drop = ['Ticket', 'PassengerId', 'Cabin', 'Name']
df_drop = df.drop(cols_to_drop, axis = 1)
df_drop.head()
df_drop_allset = pd.get_dummies(df_drop, drop_first=True)
df_drop_allset.dropna(axis=0, inplace=True)
df_drop_allset.head()
y = df_drop_allset[['Survived']]
X = df_drop_allset.drop('Survived', axis=1)


display(y.head())
display(X.head())
print(X.info())
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
cat_col = ['Pclass','Sex_male','Embarked_Q','Embarked_S']
num_col = [i for i in X.columns if i not in cat_col]
feature_union = FeatureUnion([
      ('category', FunctionTransformer(lambda x: x[cat_col])),
      ('numeric', Pipeline([
        ('select', FunctionTransformer(lambda x: x[num_col])),
        ('scale', StandardScaler()),
        ('PCA', PCA())
        ])
      )
])

pca_grid = 'feature_select__numeric__PCA__n_components'
pca_grid_value = [1,2,3,4]
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()

dt_estimator = Pipeline([('feature_select',feature_union),('dt',tree)])

params_dt = {
    pca_grid: pca_grid_value, "dt__max_features": ["auto", "sqrt", "log2"], "dt__min_samples_split": [2,3,4], "dt__min_samples_leaf": [1,2,3,4],
}
from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()

naive_estimator = Pipeline([('feature_select',feature_union), ('naive', naive)])

params_naive = {
    pca_grid: pca_grid_value, 'naive__var_smoothing' :[1e-13, 1e-14, 1e-11, 1e-10]
}
from sklearn.neural_network import MLPClassifier

neural_net = MLPClassifier(solver='sgd', max_iter = 1000)

neural_net_estimator = Pipeline([('feature_select',feature_union), ('nn', neural_net)])

params_neural_net = {
    pca_grid: pca_grid_value, 'nn__hidden_layer_sizes' : [(100,), (50, 100), (10, 50, 100)], 'nn__activation': ['tanh', 'relu'],
    'nn__learning_rate' : [ 'constant', 'invscaling', 'adaptive']                                                                
}
dt_model = GridSearchCV(dt_estimator, params_dt, n_jobs=1, cv = 5)

naive_model = GridSearchCV(naive_estimator, params_naive, n_jobs=1, cv = 5)

neural_net_model = RandomizedSearchCV(neural_net_estimator, params_neural_net, n_jobs=1 , cv = 5)

dt_model.fit(X, y)
naive_model.fit(X, y.values.ravel())
neural_net_model.fit(X, y.values.ravel())
print(dt_model.best_params_)
print(naive_model.best_params_)
print(neural_net_model.best_params_)
best_dt = dt_model.best_estimator_
best_naive = naive_model.best_estimator_
best_nn = neural_net_model.best_estimator_
print(best_dt.score(X, y))
print(best_naive.score(X, y))
print(best_nn.score(X, y))
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
def evaluate(model, model_name, X = X, y = y, is_dt = False):
  label = y if is_dt else y.values.ravel()
  pred = model.predict(X)
  matrix = confusion_matrix(label, pred)
  print("Confusion matrix ::\n",matrix)

  tn, fp, fn, tp = matrix.ravel()
  
  print(f"true positive: {tp}, true negative: {tn}\nfalse positive: {fp}, false negative: {fn}")

  print(f'\n{model_name} ::')
  
  print(f'recall score : {recall_score(label, pred).round(4)}')
  print(f'precision score : {precision_score(label, pred).round(4)}')
  print(f'f1 score : {f1_score(label, pred).round(4)}')
    
  print('-'*15)
evaluate(best_dt, 'Decision Tree', is_dt = True)
evaluate(best_naive, 'Naive Bayes')
evaluate(best_nn, 'Neural Network')