# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = data, test_df
data.columns
for dataset in combine:
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset.drop(['Cabin'], axis=1, inplace=True)
    dataset['Embarked'].fillna(method='ffill', inplace=True)
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)
sex_mapping = {'male':0, 'female':1}
embarked_mapping = {'Q':0, 'S':1, 'C':2}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
    dataset = pd.get_dummies(dataset, columns=['Sex', 'Embarked'], drop_first=True)
for dataset in combine:
    dataset.drop(columns=['Name', 'Ticket'], axis=1, inplace=True)
data.drop(columns=['PassengerId'], axis=1, inplace=True)
test_df.columns
y = data['Survived'].values
data.drop('Survived', axis=1, inplace=True)
passengers_id = test_df['PassengerId']
X_test = test_df.drop('PassengerId', axis=1).values

data_scaler = Normalizer().fit(data.values)
X_train_scaled = data_scaler.transform(data.values)
X_test_scaled = data_scaler.transform(X_test)
X_test_scaled
params = {'kernel':['rbf', 'poly'],
         'degree': [x for x in range(1,15)],
         'C': [x for x in range(1,100)]}
svccv = RandomizedSearchCV(cv=5, estimator=SVC(), param_distributions=params)
svccv.fit(X_train_scaled, y)
svc = svccv.best_estimator_
svc.fit(X_train_scaled, y)
y_pred = svc.predict(X_test_scaled)
RFC = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=9, max_features=5, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=18, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
RFC.fit(X_train_scaled, y)
model = XGBClassifier(silent=False, 
                      scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=1000, 
                      reg_alpha = 0.3,
                      max_depth=4, 
                      gamma=10)
model.fit(X_train_scaled, y)
y_pred = model.predict(X_test_scaled)
y_pred.shape
submission_df = pd.DataFrame({'PassengerId':passengers_id, 'Survived':y_pred})
submission_df.to_csv('submission.csv', index=False)
model.score(X_train_scaled, y)
