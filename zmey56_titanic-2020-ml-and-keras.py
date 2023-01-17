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
# import libraries
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from pprint import pprint
import xgboost
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from xgboost import XGBClassifier
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# print cross validation metrics
def get_cv_roc_pr_rc(clf, skf, X, y):
    roc_auc = []
    pr_auc = []
    precision = []
    recall = []
    models = []
    for train, test in skf.split(X, y):
        models.append(clf.fit(X.iloc[train,:], y.iloc[train]))
        y_pred = clf.predict(X.iloc[test,:])
        y_pred_prob = clf.predict_proba(X.iloc[test,:])[:,1]
        roc_auc.append(roc_auc_score(y.iloc[test], y_pred_prob))
        pr_auc.append(average_precision_score(y.iloc[test], y_pred_prob))
        precision.append(precision_score(y.iloc[test], y_pred))
        recall.append(recall_score(y.iloc[test], y_pred))

    print ('ROC-AUC:', np.mean(roc_auc))
    print ('PR-AUC:', np.mean(pr_auc))
    print ('Precission:', np.mean(precision))
    print ('Recall:', np.mean(recall))
    
    return roc_auc, pr_auc, precision, recall, models
# print cross validation metrics two
def get_cv_roc_pr_rc_2(clf, skf, X, y):
    roc_auc = []
    pr_auc = []
    precision = []
    recall = []
    models = []
    for train, test in skf.split(X, y):
        models.append(clf.fit(X.iloc[train,:], y.iloc[train]))
        y_pred = clf.predict(X.iloc[test,:])
        #y_pred_prob = clf.predict_proba(X.iloc[test,:])[:,1]
        roc_auc.append(roc_auc_score(y.iloc[test], y_pred))
        pr_auc.append(average_precision_score(y.iloc[test], y_pred))
        precision.append(precision_score(y.iloc[test], y_pred))
        recall.append(recall_score(y.iloc[test], y_pred))

    print ('ROC-AUC:', np.mean(roc_auc))
    print ('PR-AUC:', np.mean(pr_auc))
    print ('Precission:', np.mean(precision))
    print ('Recall:', np.mean(recall))
    
    return roc_auc, pr_auc, precision, recall, models
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
# Concatenate training and test sets
data = pd.concat([train.drop(['Survived'], axis=1), test])
fig, axarr = plt.subplots(1, 2, figsize=(12,6))
a = sns.countplot(train['Sex'], ax=axarr[0]).set_title('Passengers count by sex')
axarr[1].set_title('Survival rate by sex')
b = sns.barplot(x='Sex', y='Survived', data=train, ax=axarr[1]).set_ylabel('Survival rate')
# survival rate by class
train.groupby('Pclass').Survived.mean()
fig, axarr = plt.subplots(1,2,figsize=(12,6))
a = sns.countplot(x='Pclass', hue='Survived', data=train, ax=axarr[0]).set_title('Survivors and deads count by class')
axarr[1].set_title('Survival rate by class')
b = sns.barplot(x='Pclass', y='Survived', data=train, ax=axarr[1]).set_ylabel('Survival rate')
# survival rate by class
train.groupby(['Pclass', 'Sex']).Survived.mean()
plt.title('Survival rate by sex and class')
g = sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train).set_ylabel('Survival rate')
fig, axarr = plt.subplots(1,2,figsize=(12,6))
axarr[0].set_title('Age distribution')
f = sns.distplot(train['Age'], color='g', bins=40, ax=axarr[0])
axarr[1].set_title('Age distribution for the two subpopulations')
g = sns.kdeplot(train['Age'].loc[train['Survived'] == 1], 
                shade= True, ax=axarr[1], label='Survived').set_xlabel('Age')
g = sns.kdeplot(train['Age'].loc[train['Survived'] == 0], 
                shade=True, ax=axarr[1], label='Not Survived')
plt.figure(figsize=(8,5))
g = sns.swarmplot(y='Sex', x='Age', hue='Survived', data=train).set_title('Survived by age and sex')
plt.figure(figsize=(8,5))
h = sns.swarmplot(x='Pclass', y='Age', hue='Survived', data=train).set_title('Survived by age and class')
train.Fare.describe()
fig, axarr = plt.subplots(1,2,figsize=(12,6))
f = sns.distplot(train.Fare, color='g', ax=axarr[0]).set_title('Fare distribution')
fare_ranges = pd.qcut(train.Fare, 4, labels = ['Low', 'Mid', 'High', 'Very high'])
axarr[1].set_title('Survival rate by fare category')
g = sns.barplot(x=fare_ranges, y=train.Survived, ax=axarr[1]).set_ylabel('Survival rate')
plt.figure(figsize=(8,5))
# I excluded the three outliers with fare > 500 from this plot
a = sns.swarmplot(x='Sex', y='Fare', hue='Survived', data=train.loc[train.Fare<500]).set_title('Survived by fare and sex')
# passenger with free ticket or mistake?
train.loc[train.Fare==0]
fig, axarr = plt.subplots(1,2,figsize=(12,6))
sns.countplot(train['Embarked'], ax=axarr[0]).set_title('Passengers count by boarding point')
p = sns.countplot(x = 'Embarked', hue = 'Survived', data = train, 
                  ax=axarr[1]).set_title('Survivors and deads count by boarding point')
g = sns.countplot(data=train, x='Embarked', hue='Pclass').set_title('Pclass count by embarking point')
# Extract Title from Name, store in column and plot barplot
data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);
# Substitute titles and plot barplot
data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);
fig, axarr = plt.subplots(1,2,figsize=(12,6))
a = sns.countplot(train['SibSp'], ax=axarr[0]).set_title('Passengers count by SibSp')
axarr[1].set_title('Survival rate by SibSp')
b = sns.barplot(x='SibSp', y='Survived', data=train, ax=axarr[1]).set_ylabel('Survival rate')
fig, axarr = plt.subplots(1,2,figsize=(12,6))
a = sns.countplot(train['Parch'], ax=axarr[0]).set_title('Passengers count by Parch')
axarr[1].set_title('Survival rate by Parch')
b = sns.barplot(x='Parch', y='Survived', data=train, ax=axarr[1]).set_ylabel('Survival rate')
# Create column of number of Family members onboard
data['Fam_Size'] = data.Parch + data.SibSp + 1
data['Fam_type'] = pd.cut(data.Fam_Size, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])
# Extract the first two letters
data['Ticket_lett'] = data.Ticket.apply(lambda x: x[:2])
# Calculate ticket length
data['Ticket_len'] = data.Ticket.apply(lambda x: len(x))
# fill NaN
data['Age'] = data['Age'].interpolate(limit_direction='both', method='linear')
data['Fare'] = data['Fare'].interpolate(limit_direction='both', method='linear')
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].value_counts().index[0])
# Binning numerical columns
data['CatAge'] = pd.qcut(data.Age, q=4, labels=False )
# remove unneeded columns
data = data.drop(['PassengerId', 'Name', 'SibSp', 'Parch',
       'Ticket', 'Cabin', 'Fam_Size', 'Age'], axis = 1)
data.info()
# Transform into binary variables
data_dun = pd.get_dummies(data, drop_first=True)
data_dun.shape
# extract Y values
y_train = train['Survived']
# create Stratified K-Folds cross-validator
skf = StratifiedKFold(n_splits=5)
# split to train and test
data_dun_train = data_dun[:891]
data_dun_test = data_dun[891:]
# random under-sampling
ros = RandomUnderSampler(random_state=0)
ros.fit(data_dun_train, y_train)
X_resampled, y_resampled = ros.fit_sample(data_dun_train, y_train)
lr = LogisticRegression(max_iter = 10000, penalty="l2")
grid_values = {'C': [0.001,0.01,0.1,1,10,100,1000]} 
model_lr = GridSearchCV(lr, param_grid=grid_values)
# first experiment with under-sampling
model_lr.fit(X_resampled, y_resampled)
model_lr.best_params_
lr_1 = LogisticRegression(C = 1, max_iter = 10000, penalty="l2")
roc_auc, pr_auc, precision, recall, models = get_cv_roc_pr_rc(lr_1, skf, X_resampled, y_resampled)
# second experiment without under-sampling
model_lr.fit(data_dun_train, y_train)
model_lr.best_params_
lr_2 = LogisticRegression(C = 10, max_iter = 10000, penalty="l2")
roc_auc, pr_auc, precision, recall, models = get_cv_roc_pr_rc(lr, skf, data_dun_train, y_train)
random_forest = RandomForestClassifier()
n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(random_forest, hyperF, cv = 5, verbose = 1, 
                      n_jobs = -1)
bestF = gridF.fit(data_dun_train, y_train)
gridF.best_params_
random_forest = RandomForestClassifier(max_depth = 25, min_samples_leaf = 1, min_samples_split = 15, n_estimators = 500)
roc_auc, pr_auc, precision, recall, models = get_cv_roc_pr_rc(random_forest, skf, data_dun_train, y_train)
params = {
    "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],
    "alpha" : [0.0001, 0.001, 0.01, 0.1],
    "penalty" : ["l2", "l1", "none"]
}

model = SGDClassifier(max_iter=5, tol=None)
clf = GridSearchCV(model, param_grid=params)
bestF_sgd = clf.fit(data_dun_train, y_train)
bestF_sgd.best_score_
bestF_sgd.best_estimator_
sgd_best = SGDClassifier(alpha=0.001, loss='log', max_iter=5, penalty='l1', tol=None)
roc_auc, pr_auc, precision, recall, models = get_cv_roc_pr_rc_2(sgd_best, skf, data_dun_train, y_train)
X_train, X_test, y_train_2, y_test = train_test_split(data_dun_train, y_train, test_size=0.25)
knn = KNeighborsClassifier() 

k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]

for k in k_choices:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train_2)
    y_pred_prob = knn.predict_proba(X_test)[:,1]
    #print(k)
    print('k = ', k, ' ROC-AUC:', roc_auc_score(y_test, y_pred_prob))
knn_best = KNeighborsClassifier(n_neighbors=15) 
roc_auc, pr_auc, precision, recall, models = get_cv_roc_pr_rc(knn_best, skf, data_dun_train, y_train)
gaussian = GaussianNB() 
roc_auc, pr_auc, precision, recall, models = get_cv_roc_pr_rc(gaussian, skf, data_dun_train, y_train)
params = {
    "max_iter" : [1, 5, 10, 50, 100, 1000],
    "alpha" : [0.0001, 0.001, 0.01, 0.1],
    "penalty" : ["l2", "l1", "none"]
}

model = Perceptron(tol=None)
per = GridSearchCV(model, param_grid=params)
per_best = per.fit(data_dun_train, y_train)
per_best.best_score_
per_best.best_params_
perceptron = Perceptron(alpha=0.001, max_iter=1000, penalty='l1', tol=None)
roc_auc, pr_auc, precision, recall, models = get_cv_roc_pr_rc_2(perceptron, skf, data_dun_train, y_train)
decision_tree = DecisionTreeClassifier() 
# Create lists of parameter for Decision Tree Classifier
criterion = ['gini', 'entropy']
max_depth = [4,6,8,12]

# Create a dictionary of all the parameter options 
# Note has you can access the parameters of steps of a pipeline by using '__’
parameters = dict(criterion=['gini', 'entropy'],
                  max_depth = [1,2,4,6,8,12],
                  min_samples_split = [2, 5, 10, 15, 20, 25, 30, 35, 100],
                  min_samples_leaf = [1, 2, 5, 10])

# Conduct Parameter Optmization With Pipeline
# Create a grid search object
clf_dt = GridSearchCV(decision_tree, parameters)

# Fit the grid search
clf_dt.fit(data_dun_train, y_train)

# View The Best Parameters
print('Best Criterion:', clf_dt.best_estimator_.get_params()['criterion'])
print('Best max_depth:', clf_dt.best_estimator_.get_params()['max_depth'])
print('Best min samples split:', clf_dt.best_estimator_.get_params()['min_samples_split'])
print('Best max depth:', clf_dt.best_estimator_.get_params()['min_samples_leaf'])

# Use Cross Validation To Evaluate Model
CV_Result = cross_val_score(clf_dt, data_dun_train, y_train, cv=5, n_jobs=-1)
print(); print(CV_Result)
print(); print(CV_Result.mean())
print(); print(CV_Result.std())
clf_dt.best_params_
decision_tree_best = DecisionTreeClassifier(criterion='gini',max_depth=4,min_samples_leaf=1,min_samples_split=30) 
roc_auc, pr_auc, precision, recall, models = get_cv_roc_pr_rc(decision_tree_best, skf, data_dun_train, y_train)
# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
    'n_estimators': range(8, 20),
    'max_depth': range(6, 10),
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'learning_rate': [.4, .45, .5, .55, .6],
    'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
    'gamma': [0, 0.25, 0.5, 1.0],
    'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
}

# Instantiate the regressor: gbm
gbm = XGBClassifier(n_estimators=10)

# Perform random search: grid_mse
xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 
                                    estimator = gbm, scoring = "accuracy", 
                                    verbose = 1, n_iter = 50, cv = 4)


# Fit randomized_mse to the data
xgb_random.fit(data_dun_train, y_train)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", xgb_random.best_params_)
print("Best accuracy found: ", xgb_random.best_score_)
best_score = xgb_random.best_score_
best_params = xgb_random.best_params_
print("Best score: {}".format(best_score))
print("Best params: ")
for param_name in sorted(best_params.keys()):
    print('%s: %r' % (param_name, best_params[param_name]))
xgb_best = XGBClassifier(colsample_bylevel=1.0, colsample_bytree=0.7,gamma=1.0,learning_rate=0.6,
                         max_depth=7,min_child_weight=0.5,n_estimators=17,reg_lambda=5.0,
                         subsample=0.7)
roc_auc, pr_auc, precision, recall, models = get_cv_roc_pr_rc(xgb_best, skf, data_dun_train, y_train)
random_forest.fit(data_dun_train, y_train)
y_pred = pd.DataFrame(random_forest.predict(data_dun_test), columns=['Survived'])
sub = pd.concat([test.PassengerId, y_pred], axis=1)
sub = sub.set_index(['PassengerId'])
sub.to_csv('titanic_RF_1.csv')
y_pred = pd.DataFrame(xgb_best.predict(data_dun_test), columns=['Survived'])
sub = pd.concat([test.PassengerId, y_pred], axis=1)
sub = sub.set_index(['PassengerId'])
sub.to_csv('titanic_XGB_1.csv')
import keras
from keras.models import Sequential
from keras.layers import Dense
from numpy.random import seed
import tensorflow
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
ss = StandardScaler()
data_train_scaler = ss.fit_transform(data_dun_train)
data_test_scaler = ss.fit_transform(data_dun_test)
model = Sequential()
len(data_train_scaler[1])
model.add(Dense(512, activation = 'relu', input_dim = 73))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(data_train_scaler, y_train, batch_size = 10, epochs = 100)
y_pred = pd.DataFrame(model.predict(data_test_scaler), columns=['Survived'])
sub = pd.concat([test.PassengerId, y_pred], axis=1)
sub['Survived'].values[sub['Survived'].values > 0.5] = 1
sub['Survived'].values[sub['Survived'].values <= 0.5] = 0
sub['Survived'] = sub['Survived'].astype(int)
sub = sub.set_index(['PassengerId'])
sub.to_csv('titanic_keras_2.csv')
data_train_scaler.shape[1]
# create model
tmodel = Sequential()
tmodel.add(Dense(input_dim=data_train_scaler.shape[1], units=128,
                 kernel_initializer='normal', bias_initializer='zeros'))
tmodel.add(Activation('relu'))

for i in range(0, 8):
    tmodel.add(Dense(units=64, kernel_initializer='normal',
                     bias_initializer='zeros'))
    tmodel.add(Activation('relu'))
    tmodel.add(Dropout(.25))

tmodel.add(Dense(units=1))
tmodel.add(Activation('linear'))

tmodel.compile(loss='mean_squared_error', optimizer='rmsprop')
tmodel.fit(data_train_scaler, y_train, epochs=600, verbose=2)
y_pred = pd.DataFrame(tmodel.predict(data_test_scaler), columns=['Survived'])
sub = pd.concat([test.PassengerId, y_pred], axis=1)
sub['Survived'].values[sub['Survived'].values > 0.5] = 1
sub['Survived'].values[sub['Survived'].values <= 0.5] = 0
sub['Survived'] = sub['Survived'].astype(int)
sub = sub.set_index(['PassengerId'])
sub.to_csv('titanic_keras_exp_1.csv')
def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_all = concat_df(df_train, df_test)

df_train.name = 'Training Set'
df_test.name = 'Test Set'
df_all.name = 'All Set' 

dfs = [df_train, df_test]

print('Number of Training Examples = {}'.format(df_train.shape[0]))
print('Number of Test Examples = {}\n'.format(df_test.shape[0]))
print('Training X Shape = {}'.format(df_train.shape))
print('Training y Shape = {}\n'.format(df_train['Survived'].shape[0]))
print('Test X Shape = {}'.format(df_test.shape))
print('Test y Shape = {}\n'.format(df_test.shape[0]))
print(df_train.columns)
print(df_test.columns)
df_all['Fare'] = pd.qcut(df_all['Fare'], 13)
df_all['Age'] = pd.qcut(df_all['Age'], 10)
df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1

fig, axs = plt.subplots(figsize=(20, 20), ncols=2, nrows=2)
plt.subplots_adjust(right=1.5)

sns.barplot(x=df_all['Family_Size'].value_counts().index, y=df_all['Family_Size'].value_counts().values, ax=axs[0][0])
sns.countplot(x='Family_Size', hue='Survived', data=df_all, ax=axs[0][1])

axs[0][0].set_title('Family Size Feature Value Counts', size=20, y=1.05)
axs[0][1].set_title('Survival Counts in Family Size ', size=20, y=1.05)

family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
df_all['Family_Size_Grouped'] = df_all['Family_Size'].map(family_map)

sns.barplot(x=df_all['Family_Size_Grouped'].value_counts().index, y=df_all['Family_Size_Grouped'].value_counts().values, ax=axs[1][0])
sns.countplot(x='Family_Size_Grouped', hue='Survived', data=df_all, ax=axs[1][1])

axs[1][0].set_title('Family Size Feature Value Counts After Grouping', size=20, y=1.05)
axs[1][1].set_title('Survival Counts in Family Size After Grouping', size=20, y=1.05)

for i in range(2):
    axs[i][1].legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 20})
    for j in range(2):
        axs[i][j].tick_params(axis='x', labelsize=20)
        axs[i][j].tick_params(axis='y', labelsize=20)
        axs[i][j].set_xlabel('')
        axs[i][j].set_ylabel('')

plt.show()
df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')
df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df_all['Is_Married'] = 0
df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1
fig, axs = plt.subplots(nrows=2, figsize=(20, 20))
sns.barplot(x=df_all['Title'].value_counts().index, y=df_all['Title'].value_counts().values, ax=axs[0])

axs[0].tick_params(axis='x', labelsize=10)
axs[1].tick_params(axis='x', labelsize=15)

for i in range(2):    
    axs[i].tick_params(axis='y', labelsize=15)

axs[0].set_title('Title Feature Value Counts', size=20, y=1.05)

df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')

sns.barplot(x=df_all['Title'].value_counts().index, y=df_all['Title'].value_counts().values, ax=axs[1])
axs[1].set_title('Title Feature Value Counts After Grouping', size=20, y=1.05)

plt.show()
import string
def extract_surname(data):    
    
    families = []
    
    for i in range(len(data)):        
        name = data.iloc[i]

        if '(' in name:
            name_no_bracket = name.split('(')[0] 
        else:
            name_no_bracket = name
            
        family = name_no_bracket.split(',')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]
        
        for c in string.punctuation:
            family = family.replace(c, '').strip()
            
        families.append(family)
            
    return families

df_all['Family'] = extract_surname(df_all['Name'])
df_train = df_all.loc[:890]
df_test = df_all.loc[891:]
dfs = [df_train, df_test]
# Creating a list of families and tickets that are occuring in both training and test set
non_unique_families = [x for x in df_train['Family'].unique() if x in df_test['Family'].unique()]
non_unique_tickets = [x for x in df_train['Ticket'].unique() if x in df_test['Ticket'].unique()]

df_family_survival_rate = df_train.groupby('Family')['Survived', 'Family','Family_Size'].median()
df_ticket_survival_rate = df_train.groupby('Ticket')['Survived', 'Ticket','Ticket_Frequency'].median()

family_rates = {}
ticket_rates = {}

for i in range(len(df_family_survival_rate)):
    # Checking a family exists in both training and test set, and has members more than 1
    if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:
        family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]

for i in range(len(df_ticket_survival_rate)):
    # Checking a ticket exists in both training and test set, and has members more than 1
    if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:
        ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]
mean_survival_rate = np.mean(df_train['Survived'])

train_family_survival_rate = []
train_family_survival_rate_NA = []
test_family_survival_rate = []
test_family_survival_rate_NA = []

for i in range(len(df_train)):
    if df_train['Family'][i] in family_rates:
        train_family_survival_rate.append(family_rates[df_train['Family'][i]])
        train_family_survival_rate_NA.append(1)
    else:
        train_family_survival_rate.append(mean_survival_rate)
        train_family_survival_rate_NA.append(0)
        
for i in range(len(df_test)):
    if df_test['Family'].iloc[i] in family_rates:
        test_family_survival_rate.append(family_rates[df_test['Family'].iloc[i]])
        test_family_survival_rate_NA.append(1)
    else:
        test_family_survival_rate.append(mean_survival_rate)
        test_family_survival_rate_NA.append(0)
        
df_train['Family_Survival_Rate'] = train_family_survival_rate
df_train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA
df_test['Family_Survival_Rate'] = test_family_survival_rate
df_test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA

train_ticket_survival_rate = []
train_ticket_survival_rate_NA = []
test_ticket_survival_rate = []
test_ticket_survival_rate_NA = []

for i in range(len(df_train)):
    if df_train['Ticket'][i] in ticket_rates:
        train_ticket_survival_rate.append(ticket_rates[df_train['Ticket'][i]])
        train_ticket_survival_rate_NA.append(1)
    else:
        train_ticket_survival_rate.append(mean_survival_rate)
        train_ticket_survival_rate_NA.append(0)
        
for i in range(len(df_test)):
    if df_test['Ticket'].iloc[i] in ticket_rates:
        test_ticket_survival_rate.append(ticket_rates[df_test['Ticket'].iloc[i]])
        test_ticket_survival_rate_NA.append(1)
    else:
        test_ticket_survival_rate.append(mean_survival_rate)
        test_ticket_survival_rate_NA.append(0)
        
df_train['Ticket_Survival_Rate'] = train_ticket_survival_rate
df_train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
df_test['Ticket_Survival_Rate'] = test_ticket_survival_rate
df_test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA
