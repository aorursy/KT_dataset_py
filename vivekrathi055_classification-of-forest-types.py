import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



from pandas.plotting import scatter_matrix

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

import warnings

warnings.filterwarnings("ignore")
dataset = pd.read_csv('../input/learn-together/train.csv')

y = dataset['Cover_Type']

X = dataset.drop('Cover_Type', axis=1)
X.head()
X.shape
X=X.drop(['Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points'], axis=1)
X.shape
y.head()
y.value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

models = []

#models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

#models.append(('LDA', LinearDiscriminantAnalysis()))

#models.append(('KNN', KNeighborsClassifier()))

#models.append(('CART', DecisionTreeClassifier()))

#models.append(('NB', GaussianNB()))

#models.append(('RF', RandomForestClassifier()))

#models.append(('XGB', XGBClassifier()))

#models.append(('LGBM', LGBMClassifier()))



#models.append(('SVM', SVC(gamma='auto')))
#results = []

#names = []

#for name, model in models:

#    kfold = model_selection.KFold(n_splits=10, random_state=0)

#    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

#    results.append(cv_results)

#    names.append(name)

#    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

#    print(msg)
#from sklearn.ensemble import RandomForestClassifier

#classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

#classifier.fit(X_train, y_train)

#print('Train Score : ', classifier.score(X_train, y_train))

#print('Test Score : ', classifier.score(X_test, y_test))
classifier1 = LGBMClassifier(learning_rate=0.3, n_estimators=600) 

classifier1.fit(X_train, y_train)

print('Train Score : ', classifier1.score(X_train, y_train))

print('Test Score : ', classifier1.score(X_test, y_test))
y_pred = classifier1.predict(X_test)

print('Accuracy Score: ', accuracy_score(y_test, y_pred))

print('Confusion Matrix: ',confusion_matrix(y_test, y_pred))

print('Classification Report : ',classification_report(y_test, y_pred))
classifier1.fit(X,y)

print('Train Score : ', classifier1.score(X, y))
test_data = pd.read_csv('../input/learn-together/test.csv')
test_data = test_data.drop(['Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points'], axis=1)
predictions = classifier1.predict(test_data)
submission = pd.DataFrame()

submission['Id'] = test_data['Id']

submission['Cover_Type'] = predictions
submission.to_csv('submission.csv', index=False)
#from sklearn.model_selection import GridSearchCV

#parameters = [{'learning_rate': [0.2,0.3,0.4], 'n_estimators': [500,550,600,650,700]}]

#grid_search = GridSearchCV(estimator = classifier1,

#                           param_grid = parameters,

#                           scoring = 'accuracy',

#                           cv = 10,

#                           n_jobs = -1)

#grid_search = grid_search.fit(X_train, y_train, 

#              eval_set=[(X_test, y_test)], 

#              early_stopping_rounds=10)

#best_accuracy = grid_search.best_score_

#best_parameters = grid_search.best_params_
#print(best_accuracy)

#print(best_parameters)
