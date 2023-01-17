# Basic Libraries

import warnings
import re

# Data Analysis and Wrangling

import pandas as pd
import numpy as np
import random as rnd

# Visualization

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn-pastel')
sns.set_style('darkgrid')

# Machine Learning

import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_train.head()
#Inspect Columns and Values from Dataset:

print('Dataset Size : {}'.format(df_train.shape))
print('\nDataset Features: {}'.format(df_train.columns))
print('Dataset Main Info:\n')
df_train.info()
print('_'*40)
df_train.info()
df_train.describe()
df_train.describe(include=['O'])
#Understand the Survival Rates:

df_train['Survived'].value_counts().plot(kind='bar',figsize=(10,5), color = 'darkslateblue')

_= plt.title('Survived Feature')
_= plt.xlabel('Survived?')
_= plt.ylabel('Occurencies')

print('Class 0 (No): {}'.format(round(df_train['Survived'].value_counts()[0]/
                                      df_train['Survived'].value_counts().sum()*100,2)))
print('Class 1 (Yes): {}'.format(round(df_train['Survived'].value_counts()[1]/
                                       df_train['Survived'].value_counts().sum()*100,2)))
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
grid = sns.FacetGrid(df_train, col='Survived')
grid.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
grid = sns.FacetGrid(df_train, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
grid = sns.FacetGrid(df_train, row='Embarked', col='Survived', height=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
#Define Enconde Function - For All Categorical Features:

def Encode_Class(df, name):
    df = pd.concat([df, pd.get_dummies(df[name], prefix=name)], axis=1)
    df.drop(labels=name, axis=1, inplace=True)
    return df
#Understand the Survival Rates:

df_train['Survived'].value_counts().plot(kind='bar',figsize=(10,5), color = 'darkslateblue')

_= plt.title('Survived Feature')
_= plt.xlabel('Survived')
_= plt.ylabel('Occurencies')

print('Class 0 (No): {}'.format(round(df_train['Survived'].value_counts()[0]/
                                      df_train['Survived'].value_counts().sum()*100,2)))
print('Class 1 (Yes): {}'.format(round(df_train['Survived'].value_counts()[1]/
                                       df_train['Survived'].value_counts().sum()*100,2)))
#Inspect a few Cabin Examples:

df_train['Cabin'][df_train['Cabin'].isnull() == False].sample(5)
#Normalize the Missing Data from Cabin Feature:

df_train.fillna('N', inplace=True)
#Create Deck Feature:

df_train['Deck'] = df_train['Cabin'].str.slice(0,1)

df_train = Encode_Class(df_train,'Deck')

df_train.sample(5)
#Inspect a few Name Examples:

df_train['Name'].sample(5)
#Define Title Function:

def Get_Title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    
    if title_search:
        return title_search.group(1)
    return ""
#Apply Function and Inspect Results:

pd.crosstab(df_train['Name'].apply(Get_Title),df_train['Sex'])
#Define Name/Title transformation Function:

def Name_into_Titles(df, name):
    df['Title'] = df[name].apply(Get_Title)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 
                                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    df.drop(labels=name, axis=1, inplace=True)
    
    df = Encode_Class(df, 'Title')
        
    return df
df_train = Name_into_Titles(df_train, 'Name')

df_train.sample(5)
#Understand the Classes Relations:

df_train['Pclass'].value_counts().plot(kind='bar',figsize=(10,5), color ='darkslateblue')

_= plt.title('Pclass Feature')
_= plt.xlabel('Class')
_= plt.ylabel('Occurencies')

print('Class 1 (First): {}'.format(round(df_train['Pclass'].value_counts()[1]/
                                      df_train['Pclass'].value_counts().sum()*100,2)))
print('Class 2 (Second): {}'.format(round(df_train['Pclass'].value_counts()[2]/
                                       df_train['Pclass'].value_counts().sum()*100,2)))
print('Class 3 (Third): {}'.format(round(df_train['Pclass'].value_counts()[3]/
                                       df_train['Pclass'].value_counts().sum()*100,2)))
#Apply Function on Dataset:

df_train = Encode_Class(df_train, 'Pclass')

df_train.sample(5)
#Inspect Sex Feature:

df_train['Sex'].value_counts().plot(kind='bar',figsize=(10,5), color = 'darkslateblue')

_= plt.title('Sex Feature')
_= plt.xlabel('Class')
_= plt.ylabel('Occurencies')

print('Male: {}'.format(round(df_train['Sex'].value_counts()['male']/
                                      df_train['Sex'].value_counts().sum()*100,2)))
print('Female: {}'.format(round(df_train['Sex'].value_counts()['female']/
                                       df_train['Sex'].value_counts().sum()*100,2)))
#Transform Sex Feature into Categorical:

df_train = Encode_Class(df_train,'Sex')

df_train.sample(5)
#Replace NaN Values of the Age Feature:

df_train = df_train.replace('N', np.nan)
df_train['Age'].fillna(df_train['Age'].median())
#Inspect Age Feature:

df_train['Age'].plot(kind='hist',figsize=(10,5), color = 'darkslateblue')

_= plt.title('Age Histogram')
_= plt.xlabel('Age')
_= plt.ylabel('Occurencies')
#Create Bins for Age Feature:

df_train['Age_Bin'] = pd.cut(df_train['Age'], bins=[0,12,20,40,120], labels=['Children','Young','Adult','Elder'])

df_train = Encode_Class(df_train,'Age_Bin')

df_train.sample(5)
#Create FamilySize Feature:

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1

df_train['IsAlone'] = 0
df_train.loc[df_train['FamilySize']==1, 'IsAlone']=1

df_train.sample(5)
#Exclude Ticket Feature:

df_train.drop(labels=['Ticket'], axis=1, inplace=True)

df_train.sample(5)
#Inspect Fare Feature:

df_train['Fare'].plot(kind='hist',bins=3, figsize=(10,5), color = 'darkslateblue')

_= plt.title('Fare Histogram')
_= plt.xlabel('Fare')
_= plt.ylabel('Occurencies')
#Create Bins for Age Feature:

df_train['Fare_Bin'] = pd.cut(df_train['Fare'], bins=3, labels=['Low','Median','High'])

df_train = Encode_Class(df_train,'Fare_Bin')

df_train.sample(5)
#Inspect Embarked Feature:

df_train['Embarked'].value_counts()
#Transform Missing Data from the Embarked Feature:

df_train['Embarked'].fillna('S', inplace=True)

df_train = Encode_Class(df_train,'Embarked')

df_train.sample(5)
#Inspect the Titanic Dataset:

print(df_train.columns)
df_train.sample(5)
#Clear Titanic Dataset:

df_train = df_train[['Pclass_1','Pclass_2','Pclass_3',
                         'Title_Master','Title_Miss','Title_Mr', 'Title_Mrs', 'Title_Rare',
                         'Sex_female', 'Sex_male',
                         'Age_Bin_Children', 'Age_Bin_Young', 'Age_Bin_Adult', 'Age_Bin_Elder',
                         'FamilySize', 'IsAlone',
                         'Fare_Bin_Low', 'Fare_Bin_Median',
                         'Fare_Bin_High',
                         'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E','Deck_F', 'Deck_G', 'Deck_N', 'Deck_T',
                         'Embarked_C','Embarked_Q', 'Embarked_S',
                         'Survived'
                        ]]

df_train.sample(5)
#Scale FamilySize Feature:

df_train['FamilySize'] = StandardScaler().fit_transform(np.array(df_train['FamilySize']).reshape(-1,1))
#Define X and Y Features:

X = df_train.iloc[:,:-1]
y = df_train.iloc[:,-1]
#Split Train and Test Samples:

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#Prepare Variables to Store and Display Results:

model_matrix = {}

def Display_Model_Results(model_name, model, y_test, X_test):
    
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(20,10))
    fig.subplots_adjust(hspace =.2, wspace=.2)
       
    pred = model.predict(X_test)
    pred_proba_pos = model.predict_proba(X_test)[:,1]
    
    fpr, tpr, thresholds_forest = roc_curve(y_test.values,pred_proba_pos)

    print('{0} Model: {1}%'.format(model_name, round(roc_auc_score(y_test, pred_proba_pos)*100,2)))

    _= ax[0].plot(fpr, tpr, linewidth=2)
    _= ax[0].plot([0, 1], [0, 1], 'k--')

    _= ax[0].set_xlim(0,1)
    _= ax[0].set_ylim(0,1)

    _= ax[0].set_xlabel('False Positive Rate', fontsize=16)
    _= ax[0].set_ylabel('True Positive Rate', fontsize=16)
    _= ax[0].set_title('ROC Curve', fontsize=20)

    #Plot Confusion Matrix:

    sns.heatmap(confusion_matrix(y_test, pred),square=True, annot=True, linewidths=.5, cbar=False, robust=True,
                cmap='Purples', annot_kws={'size': 25}, fmt='g', ax=ax[1])

    _= ax[1].set_xticks([0.5,1.5])
    _= ax[1].set_yticks([0.5,1.5])

    _= ax[1].set_ylabel('True Results', fontsize=16)
    _= ax[1].set_xlabel('Predicted Results', fontsize=16)

    _= ax[1].set_title('Confusion Matrix', fontsize=20)
    
    model_matrix[model_name] = [round(roc_auc_score(y_test, pred_proba_pos)*100,2), model]
#Initiate Model and Inspect Base Parameters:

LogReg = LogisticRegression()

LogReg.get_params()
#Create Grid Search Parameters based on Parameters:

random_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
               'multi_class': ['auto', 'ovr', 'multinominal'],
               'penalty': ['l1', 'l2', 'elasticnet', 'none'],
               'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
               }
#Set GridSearch Using RandomizedSearchCV:

kfold = KFold(n_splits=10, random_state=42)

model_random = RandomizedSearchCV(estimator=LogReg,
                                  param_distributions=random_grid,
                                  n_iter=100,
                                  cv=kfold,
                                  verbose=1,
                                  random_state=42,
                                  scoring="accuracy",
                                  n_jobs=4)
#Fit Model with RandomizedSearchGrid:

start_time = pd.Timestamp.today()

model_random.fit(X_train,y_train)

end_time = pd.Timestamp.today()

print("Executed in: {}".format(end_time - start_time))
#Choose the Best Model Calculated:

model_name = 'LogisticRegression'

best_model = model_random.best_estimator_
#Inspect Results:

Display_Model_Results(model_name, best_model, y_validate, X_validate)
#Initiate Model and Inspect Base Parameters:

KNN = KNeighborsClassifier()

KNN.get_params()
#Create Grid Search Parameters based on Parameters:

random_grid = {'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
               'leaf_size': [30, 40, 50, 60, 70, 80, 90, 100],
               'p': [1, 2],
               'n_neighbors': [4, 5, 6, 7, 8]
               }
#Set GridSearch Using RandomizedSearchCV:

kfold = KFold(n_splits=10, random_state=42)

model_random = RandomizedSearchCV(estimator=KNN,
                                  param_distributions=random_grid,
                                  n_iter=100,
                                  cv=kfold,
                                  verbose=1,
                                  random_state=42,
                                  scoring="accuracy",
                                  n_jobs=4)
#Fit Model with RandomizedSearchGrid:

start_time = pd.Timestamp.today()

model_random.fit(X_train,y_train)

end_time = pd.Timestamp.today()

print("Executed in: {}".format(end_time - start_time))
#Choose the Best Model Calculated:

model_name = 'K-Nearest Neighbours (KNN)'

best_model = model_random.best_estimator_
#Inspect Results:

Display_Model_Results(model_name, best_model, y_validate, X_validate)
#Initiate Model and Inspect Base Parameters:

Decision_Tree = DecisionTreeClassifier()

Decision_Tree.get_params()
#Create Grid Search Parameters based on Parameters:

random_grid = {'max_features': ['log2', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'criterion':['gini','entropy']}
#Set GridSearch Using RandomizedSearchCV:

kfold = KFold(n_splits=10, random_state=42)

model_random = RandomizedSearchCV(estimator=Decision_Tree,
                                  param_distributions=random_grid,
                                  n_iter=100,
                                  cv=kfold,
                                  verbose=1,
                                  random_state=42,
                                  scoring="accuracy",
                                  n_jobs=4)
#Fit Model with RandomizedSearchGrid:

start_time = pd.Timestamp.today()

model_random.fit(X_train,y_train)

end_time = pd.Timestamp.today()

print("Executed in: {}".format(end_time - start_time))
#Choose the Best Model Calculated:

model_name = 'Decision Tree'

best_model = model_random.best_estimator_
#Inspect Results:

Display_Model_Results(model_name, best_model, y_validate, X_validate)
#Initiate Model and Inspect Base Parameters:

Random_Forest = RandomForestClassifier()

Random_Forest.get_params()
#Create Grid Search Parameters based on Parameters:

random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
               'max_features': ['log2', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False],
               'criterion':['gini','entropy']}
#Set GridSearch Using RandomizedSearchCV:

kfold = KFold(n_splits=10, random_state=42)

model_random = RandomizedSearchCV(estimator=Random_Forest,
                                  param_distributions=random_grid,
                                  n_iter=100,
                                  cv=kfold,
                                  verbose=1,
                                  random_state=42,
                                  scoring="accuracy",
                                  n_jobs=4)
#Fit Model with RandomizedSearchGrid:

start_time = pd.Timestamp.today()

model_random.fit(X_train,y_train)

end_time = pd.Timestamp.today()

print("Executed in: {}".format(end_time - start_time))
#Choose the Best Model Calculated:

model_name = 'Random_Forest'

best_model = model_random.best_estimator_
#Inspect Results:

Display_Model_Results(model_name, best_model, y_validate, X_validate)
#Initiate Model and Inspect Base Parameters:

SGD = SGDClassifier()

SGD.get_params()
#Create Grid Search Parameters based on Parameters:

random_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1],
               'loss': ['hinge','log', 'modified_huber', 'epsilon_insensitive']
              }
#Set GridSearch Using RandomizedSearchCV:

kfold = KFold(n_splits=10, random_state=42)

model_random = RandomizedSearchCV(estimator=SGD,
                                  param_distributions=random_grid,
                                  n_iter=100,
                                  cv=kfold,
                                  verbose=1,
                                  random_state=42,
                                  scoring="accuracy",
                                  n_jobs=4)
#Fit Model with RandomizedSearchGrid:

start_time = pd.Timestamp.today()

model_random.fit(X_train,y_train)

end_time = pd.Timestamp.today()

print("Executed in: {}".format(end_time - start_time))
#Choose the Best Model Calculated:

model_name = 'Stochastic Gradient Descent'

best_model = model_random.best_estimator_
#Inspect Results:

Display_Model_Results(model_name, best_model, y_validate, X_validate)
#Initiate Model and Inspect Base Parameters:

AdaBoost_clf = AdaBoostClassifier()

AdaBoost_clf.get_params()
#Create Grid Search Parameters based on Parameters:

random_grid = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400],
               'learning_rate': [0.01, 0.05, 0.1,0.3,1,1.3, 1.5,1.7] ,
               'algorithm': ['SAMME', 'SAMME.R']}
#Set GridSearch Using RandomizedSearchCV:

kfold = KFold(n_splits=10, random_state=42)

model_random = RandomizedSearchCV(estimator=AdaBoost_clf,
                                  param_distributions=random_grid,
                                  n_iter=100,
                                  cv=kfold,
                                  verbose=1,
                                  random_state=42,
                                  scoring="accuracy",
                                  n_jobs=4)
#Fit Model with RandomizedSearchGrid:

start_time = pd.Timestamp.today()

model_random.fit(X_train,y_train)

end_time = pd.Timestamp.today()

print("Executed in: {}".format(end_time - start_time))
#Choose the Best Model Calculated:

model_name = 'AdaBoost'

best_model = model_random.best_estimator_
#Inspect Results:

Display_Model_Results(model_name, best_model, y_validate, X_validate)
#Initiate Model and Inspect Base Parameters:

Bagging_clf = BaggingClassifier()

Bagging_clf.get_params()
#Create Grid Search Parameters based on Parameters:

random_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
               'max_samples': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
               'max_features': [1, 2, 3, 4, 5, 6, 7]
               }
#Set GridSearch Using RandomizedSearchCV:

kfold = KFold(n_splits=10, random_state=42)

model_random = RandomizedSearchCV(estimator=Bagging_clf,
                                  param_distributions=random_grid,
                                  n_iter=100,
                                  cv=kfold,
                                  verbose=1,
                                  random_state=42,
                                  scoring="accuracy",
                                  n_jobs=4)
#Fit Model with RandomizedSearchGrid:

start_time = pd.Timestamp.today()

model_random.fit(X_train,y_train)

end_time = pd.Timestamp.today()

print("Executed in: {}".format(end_time - start_time))
#Choose the Best Model Calculated:

model_name = 'Bagging Classifier'

best_model = model_random.best_estimator_
#Inspect Results:

Display_Model_Results(model_name, best_model, y_validate, X_validate)
#Initiate Model and Inspect Base Parameters:

GradientBoosting_clf = GradientBoostingClassifier()

GradientBoosting_clf.get_params()
#Create Grid Search Parameters based on Parameters:

random_grid = {'n_estimators':[500, 1000, 1500, 2000, 2500], 
              'learning_rate':[0.01, 0.03, 0.05, 0.07, 0.09],
              'min_samples_split':[2,4,6,8,10],
              'min_samples_leaf':[3,5,7,9,11]}
#Set GridSearch Using RandomizedSearchCV:

kfold = KFold(n_splits=10, random_state=42)

model_random = RandomizedSearchCV(estimator=GradientBoosting_clf,
                                  param_distributions=random_grid,
                                  n_iter=100,
                                  cv=kfold,
                                  verbose=1,
                                  random_state=42,
                                  scoring="accuracy",
                                  n_jobs=4)
#Fit Model with RandomizedSearchGrid:

start_time = pd.Timestamp.today()

model_random.fit(X_train,y_train)

end_time = pd.Timestamp.today()

print("Executed in: {}".format(end_time - start_time))
#Choose the Best Model Calculated:

model_name = 'GradientBoosting'

best_model = model_random.best_estimator_
#Inspect Results:

Display_Model_Results(model_name, best_model, y_validate, X_validate)
#Initiate Model and Inspect Base Parameters:

xgb_clf = XGBClassifier(objective='binary:logistic',silent=True, nthread=1)

xgb_clf.get_params()
#Create Grid Search Parameters based on Parameters:

random_grid = {'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05],
               'n_estimators' : [100, 300, 600, 900, 1000],
               'min_child_weight': [1, 5, 10],
               'gamma': [0.5, 1, 1.5, 2, 5] ,
               'subsample': [0.8, 0.9, 1.0] ,
               'colsample_bytree': [0.3, 0.5, 0.7, 0.8],
               'max_depth': [3, 4, 5]
              }
#Set GridSearch Using RandomizedSearchCV:

kfold = KFold(n_splits=10, random_state=42)

model_random = RandomizedSearchCV(estimator=xgb_clf,
                                  param_distributions=random_grid,
                                  n_iter=100,
                                  cv=kfold,
                                  verbose=1,
                                  random_state=42,
                                  scoring="accuracy",
                                  n_jobs=4)
#Fit Model with RandomizedSearchGrid:

start_time = pd.Timestamp.today()

model_random.fit(X_train,y_train)

end_time = pd.Timestamp.today()

print("Executed in: {}".format(end_time - start_time))
#Choose the Best Model Calculated:

model_name = 'XGBoost'

best_model = model_random.best_estimator_
#Inspect Results:

Display_Model_Results(model_name, best_model, y_validate, X_validate)
#Plot Main Results from Models:

results = [value[0] for (key, value) in model_matrix.items()]
names = [key for (key, value) in model_matrix.items()]
x_bar = list(range(0,len(names)))

_= plt.figure(figsize=(20,10))

_= plt.bar([key for (key, value) in model_matrix.items()],[value[0] for (key, value) in model_matrix.items()], color='darkslateblue', alpha=0.7)

for i in range(len(x_bar)):
    _= plt.annotate(results[i],(x_bar[i] , results[i]+1), size=15, color='darkblue', horizontalalignment='center')

_= plt.ylim((75,100))

_= plt.title('%Score - Models')
