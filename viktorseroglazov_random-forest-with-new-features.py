import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("dark")
plt.rcParams['figure.figsize'] = 8, 5
import pandas as pd
import random

import warnings
warnings.filterwarnings('ignore')
train_data_set = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
train_data_set.head(5)
#Lest consider number of unique values
train_data_set.groupby('workclass').size().plot(kind='bar')

train_data_set.groupby('education').size().plot(kind='bar')
train_data_set.groupby('relationship').size().plot(kind='bar')
train_data_set.groupby('sex').size().plot(kind='bar')
train_data_set.groupby('race').size().plot(kind='bar')
train_data_set.groupby('native-country').size().plot(kind='bar')
age_column = 'age'
workclass_column = 'workclass'
bottom_threshold_age = 0
top_threshold_age = 200
workclass_set = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
occupation_set = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', \
                 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct','Adm-clerical' \
                 'Farming-fishing', 'Transport-moving', 'Protective-serv', 'Armed-Forces']

country_set = ['United-States', 'Cambodia', 'England','Puerto-Rico', \
               'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', \
               'India', 'Greece', 'China', 'Cuba', 'Iran', 'Honduras', 'Vietnam', 'Philippines', 'Italy', \
               'Poland', 'Jamaica','Mexico', 'Portugal', 'Ireland', 'Dominican-Republic',
               'Laos','Taiwan', 'Haiti' , 'Columbia', 'Hungary', 'Guatemala', 'Yugoslavia', \
               'Nicaragua','Thailand', 'Scotland', 'Thailand', 'Yugoslavia', 
               'Trinadad&Tobago', 'Peru', 'Hong', 'Holand Netherlands' ]

def strip_particular_column(column_name, data):
    return data[column_name].apply(lambda x : x.strip())

def randomly_replace(undefined_value, list_for_random_choice):
    if undefined_value == '?':
        return random.choice(list_for_random_choice)
    
    return undefined_value

fill_unknown_dictionaries = {'workclass' : workclass_set,
                             'occupation': occupation_set,
                             'native-country' : country_set}
    
def fill_question_marks(train_data):
    for column_name, fill_list in fill_unknown_dictionaries.items():
        train_data[column_name] = strip_particular_column(column_name, train_data)
        train_data[column_name] = train_data[column_name].apply(lambda x : randomly_replace(x, fill_list))
    
    return train_data


train_data= fill_question_marks(train_data_set)
train_data.workclass.value_counts()
train_data.occupation.value_counts()
train_data_set = train_data
train_data_set.occupation.value_counts()
train_data_set.head(5)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score

train_data_set['status'] = (train_data_set['marital-status'] + '-' + train_data_set['relationship'])
train_data_set['capital'] = (train_data_set['capital-gain'] + train_data_set['capital-loss']) / 2
train_data_set['age-fnlwgt'] = (train_data_set['age'] + train_data_set['fnlwgt']) / 2
train_data_set['capital'] = (train_data_set['capital-gain'] + train_data_set['capital-loss']) / 2

target_feature = 'target'

feature_list = ['age', \
                 'age-fnlwgt', \
                 'capital', \
                 'status', \
                 'capital-gain', \
                 'workclass', \
                 'fnlwgt', \
                 'education', \
                 'education-num', \
                 'marital-status', \
                 'capital-loss', \
                 'relationship', \
                 'race', \
                 'sex', \
                 'occupation', \
                 'native-country', \
                 'hours-per-week']

category_column_list = ['education', \
                        'status', \
                        'workclass', \
                        'marital-status', \
                        'occupation', \
                        'relationship', \
                        'race', \
                        'sex', \
                        'native-country']

for column in category_column_list:
    train_data_set[column] = pd.factorize(train_data_set[column])[0]
train_data_set.head(5)    
test_slice_with_training_features = train_data_set[feature_list]

forest = RandomForestClassifier(criterion='gini', \
                                      n_estimators=1200, \
                                      max_depth=15, \
                                      n_jobs=-1, \
                                      min_samples_leaf = 1, \
                                      min_samples_split = 5, \
                                      random_state=0)

model_forest = forest.fit(test_slice_with_training_features, train_data_set[target_feature])
feat_importances_random_forest = pd.Series(model_forest.feature_importances_, index=test_slice_with_training_features.columns)
feat_importances_random_forest.nlargest(14).plot(kind='barh')
plt.show()
#I have adjusted it manually
n_estimators = [1300, 1200, 1600, 2000]
max_depth = [10, 13, 15, 20]
min_samples_split = [4, 5, 3, 2]
min_samples_leaf = [1, 2, 5] 
max_features = [2, 3, 4]

params_dict = dict(n_estimators = n_estimators,
                   min_samples_leaf = min_samples_leaf,
                   max_depth = max_depth,  
                   min_samples_split = min_samples_split, 
                   max_features = max_features)



grid_forest = GridSearchCV(model_forest, \
                     params_dict, \
                     cv = 3, \
                     verbose = 1, \
                     n_jobs = -1)
grid_forest = grid_forest.fit(test_slice_with_training_features, train_data_set[target_feature])
grid_forest.best_params_
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn import metrics

optimized_forest_model = RandomForestClassifier(criterion='gini', \
                                      n_estimators=1600, \
                                      max_features = 9, \
                                      max_depth=15, \
                                      n_jobs=-1, \
                                      min_samples_leaf = 1, \
                                      min_samples_split = 2, \
                                      random_state=20)

cv = StratifiedKFold(n_splits=40, random_state=123, shuffle=True)
results = pd.DataFrame(columns=['training_score', 'test_score'])

best_forest_model = None
best_score = 0

x_train, x_test, y_train, y_test = train_test_split(train_data_set[features_list], train_data_set['target'], test_size=0.3, random_state=135)

X = train_data_set[features_list]
Y = train_data_set['target']

for (train, test), i in zip(cv.split(X, Y), range(20)):
    current_model = optimized_forest_model.fit(X.iloc[train], Y.iloc[train])
    prediction = current_model.predict(x_test)
    current_score = accuracy_score(y_test, prediction)
    if current_score > best_score:
        best_forest_model = current_model
        best_score = current_score
    print(f'step {i} score {accuracy_score(y_test, prediction)}')

#forest.fit(x_train, y_train)

prediction = best_forest_model.predict_proba(x_test)
prediction = prediction[:, 1]
print(f'max = {best_score}')
print(f'final accuracy {accuracy_score(y_test, best_model.predict(x_test))}')
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', \
                              max_depth=13, \
                              max_features=6, \
                              min_samples_leaf=0.001, \
                              random_state=20)

tree_params = {'max_depth': range(1,15),
               'max_features': range(1,8),
               'min_samples_leaf':np.linspace(0.0001, 1, 100, endpoint=True),
               'random_state': range(1, 30)
              }

tree_grid_optimizer = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1, verbose=True)
tree_grid_optimizer.fit(train_data_set[features_list], train_data_set['target'])
print(f'best params {tree_grid.best_params_}')
print(f'best score {tree_grid.best_score_}')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

x_train, x_test, y_train, y_test = train_test_split(train_data_set[features_list], train_data_set['target'], test_size=0.3, random_state=135)
scaler = StandardScaler()

knn_classifier = KNeighborsClassifier(n_neighbors=10, weights='uniform')
knn_model = knn_classifier.fit(scaler.fit_transform(x_train), y_train)

scaled_calculation_score = scaler.fit_transform(x_train[features_list])

cv = StratifiedKFold(n_splits=10, random_state=123, shuffle=True)
results = pd.DataFrame(columns=['training_score', 'test_score'])
    
X = train_data_set[features_list]
y = train_data_set['target']
best_model = None
best_score = 0
for (train, test), i in zip(cv.split(X, y), range(10)):
    current_model = knn_model.fit(X.iloc[train], y.iloc[train])
    prediction = current_model.predict(x_test)
    current_score = accuracy_score(y_test, prediction)
    if current_score > best_score:
        best_model = current_model
        best_score = current_score
    print(f'step {i} score {accuracy_score(y_test, prediction)}')
    
knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', best_model)])

knn_params = {'knn__n_neighbors': range(1, 20),
              'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
               'knn__weights': ['distance','uniform']}

knn_grid_searcher = GridSearchCV(knn_pipe, knn_params, cv=5, n_jobs=-1, verbose=True)
knn_grid_searcher.fit(train_data_set[features_list], train_data_set['target'])
print(f'knn best params {knn_grid_searcher.best_params_}')
print(f'knn best score {knn_grid_searcher.best_score_}')

knn_predict_result = best_model.predict(scaled_calculation_score)
accuracy_score(data_set_for_score['target'], knn_predict_result)

test_data_set = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')

test_data_set['status'] = (test_data_set['marital-status'] + '-' + test_data_set['relationship'])
test_data_set['capital'] = (test_data_set['capital-gain'] + test_data_set['capital-loss']) / 2
test_data_set['age-fnlwgt'] = (test_data_set['age'] + test_data_set['fnlwgt']) / 2
test_data_set['capital'] = (test_data_set['capital-gain'] + test_data_set['capital-loss']) / 2


for column in category_column_list:
    test_data_set[column] = pd.factorize(test_data_set[column])[0]
test_data_set.head(5)    
predicted_result = best_forest_model.predict_proba(test_data_set[features_list])
test_data_set['uid'].reset_index(drop=True, inplace=True)
prepared_to_csv = pd.concat([test_data_set['uid'], pd.Series(predicted_result[:,1], name='target')], axis=1)
prepared_to_csv.to_csv('submit.csv', index=False)