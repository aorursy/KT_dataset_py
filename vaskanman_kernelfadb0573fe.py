# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

PATH="../input/"

print(os.listdir(PATH))



# Any results you write to the current directory are saved as output.
app_train = pd.read_csv(PATH + 'application_train.csv',)

app_test = pd.read_csv(PATH + 'application_test.csv',)

print ("формат обучающей выборки:", app_train.shape)

print ("формат тестовой выборки:", app_test.shape)
app_train.select_dtypes(include=[object]).apply(pd.Series.nunique, axis = 0)
app_train = pd.get_dummies(app_train)

app_test = pd.get_dummies(app_test)



print('Training Features shape: ', app_train.shape)

print('Testing Features shape: ', app_test.shape)
#сохраним лейблы, их же нет в тестовой выборке и при выравнивании они потеряются. 

train_labels = app_train['TARGET']



# Выравнивание - сохранятся только столбцы. имеющиеся в обоих датафреймах

app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)



print('Формат тренировочной выборки: ', app_train.shape)

print('Формат тестовой выборки: ', app_test.shape)



# Add target back in to the data

app_train['TARGET'] = train_labels
# создадим новый датафрейм для полиномиальных признаков

poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]

poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]



# обработаем отуствующие данные

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = 'median')



poly_target = poly_features['TARGET']



poly_features = poly_features.drop('TARGET', axis=1)



poly_features = imputer.fit_transform(poly_features)

poly_features_test = imputer.transform(poly_features_test)



from sklearn.preprocessing import PolynomialFeatures

                                  

# Создадим полиномиальный объект степени 3

poly_transformer = PolynomialFeatures(degree = 3)



# Тренировка полиномиальных признаков

poly_transformer.fit(poly_features)



# Трансформация признаков

poly_features = poly_transformer.transform(poly_features)

poly_features_test = poly_transformer.transform(poly_features_test)

print('Формат полиномиальных признаков: ', poly_features.shape)
poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]
# Датафрейм для новых фич 

poly_features = pd.DataFrame(poly_features, 

                             columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 

                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))



# Добавим таргет

poly_features['TARGET'] = poly_target



# рассчитаем корреляцию

poly_corrs = poly_features.corr()['TARGET'].sort_values()



# Отобразим признаки с наивысшей корреляцией

print(poly_corrs.head(10))

print(poly_corrs.tail(5))
# загрузим тестовые признаки в датафрейм

poly_features_test = pd.DataFrame(poly_features_test, 

                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 

                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))



# объединим тренировочные датафреймы

poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']

app_train_poly = app_train.merge(poly_features, on = 'SK_ID_CURR', how = 'left')



# объединим тестовые датафреймы

poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']

app_test_poly = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')



# Выровняем датафреймы

app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)



# Посмотрим формат

print('Тренировочная выборка с полиномиальными признаками: ', app_train_poly.shape)

print('Тестовая выборка с полиномиальными признаками: ', app_test_poly.shape)
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler



# Уберем таргет из тренировочных данных

if 'TARGET' in app_train:

    train = app_train.drop(labels = ['TARGET'], axis=1)

else:

    train = app_train.copy()

features = list(train.columns)



# копируем тестовые данные

test = app_test.copy()



# заполним недостающее по медиане

imputer = SimpleImputer(strategy = 'median')



# Нормализация

scaler = MinMaxScaler(feature_range = (0, 1))



# заполнение тренировочной выборки

imputer.fit(train)



# Трансофрмация тренировочной и тестовой выборок

train = imputer.transform(train)

test = imputer.transform(app_test)



# то же самое с нормализацией

scaler.fit(train)

train = scaler.transform(train)

test = scaler.transform(test)



print('Формат тренировочной выборки: ', train.shape)

print('Формат тестовой выборки: ', test.shape)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(train,train_labels,test_size=0.3,random_state=17)
X_train.shape, y_train.shape
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
c_values = np.logspace(-2, 3, 20)
grid1.best_score_
%%time

log_reg1 = LogisticRegression(C = 0.0001,solver="lbfgs",max_iter=2500)



c_values1 = np.logspace(-2, 3, 30)

parameters={'C':c_values1}

grid1 = GridSearchCV(log_reg1, parameters, scoring='roc_auc', cv=3)

grid1.fit(X_train, y_train)

print(grid1)

# summarize the results of the grid search

print(grid1.best_score_)

print (grid1.best_params_)
from datetime import datetime



now = datetime.now()
now #время было 4.16
%%time





# Создаем модель

log_reg = LogisticRegression(C = 0.99,solver="lbfgs",max_iter=2500)



parameters={'C':c_values}

grid = GridSearchCV(log_reg, parameters, cv=5)

grid.fit(X_train, y_train)

print(grid)

# summarize the results of the grid search

print(grid.best_score_)

print (grid.best_params_)

# Тренируем модель

#log_reg.fit(X_train, y_train)





#log_reg_pred_acc = log_reg.predict(X_test)

#log_reg_pred = log_reg.predict_proba(X_test)[:, 1]

#print ("Доля правильных ответов: ",accuracy_score(y_test,log_reg_pred_acc))

#print ("ROC-AUC: ",roc_auc_score(y_test,log_reg_pred))
from sklearn.linear_model import LogisticRegression



# Создаем модель

log_reg = LogisticRegression(C = 28.072,solver="lbfgs", max_iter=2500)



# Тренируем модель

log_reg.fit(train, train_labels)





log_reg_pred = log_reg.predict_proba(test)[:, 1]

# Предсказание на тестовых данных

#predictions = random_forest.predict_proba(test)[:, 1]



# Создание датафрейма для загрузки

submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = log_reg_pred



# Сохранение

submit.to_csv('up_linear_model.csv', index = False)
from sklearn.linear_model import LogisticRegressionCV 

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier



# Создадим классификатор

random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50)



# Тренировка на тернировочных данных

random_forest.fit(train, train_labels)



# Предсказание на тестовых данных

predictions = random_forest.predict_proba(test)[:, 1]



# Создание датафрейма для загрузки

submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = predictions



# Сохранение

submit.to_csv('random_forest_baseline.csv', index = False)
poly_features_names = list(app_train_poly.columns)



# Создание и тренировка объекта для заполнение недостающих данных

imputer = SimpleImputer(strategy = 'median')



poly_features = imputer.fit_transform(app_train_poly)

poly_features_test = imputer.transform(app_test_poly)



# Нормализация

scaler = MinMaxScaler(feature_range = (0, 1))



poly_features = scaler.fit_transform(poly_features)

poly_features_test = scaler.transform(poly_features_test)



random_forest_poly = RandomForestClassifier(n_estimators = 100, random_state = 50)



# Тренировка на полиномиальных данных

random_forest_poly.fit(poly_features, train_labels)



# Предсказания

predictions = random_forest_poly.predict_proba(poly_features_test)[:, 1]



# Датафрейм для загрузки

submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = predictions



# Сохранение датафрейма

submit.to_csv('random_forest_baseline_engineered.csv', index = False)
from lightgbm import LGBMClassifier



clf = LGBMClassifier()

clf.fit(train, train_labels)



predictions = clf.predict_proba(test)[:, 1]



# Датафрейм для загрузки

submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = predictions



# Сохранение датафрейма

submit.to_csv('lightgbm_baseline.csv', index = False)
submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = log_reg_pred



submit.head()
submit.to_csv('log_reg_baseline.csv', index = False)