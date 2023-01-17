import pandas as pd

import numpy as np

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import re

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.naive_bayes import GaussianNB

import warnings

warnings.filterwarnings('ignore') 

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, cross_val_score
# Loading the test and training set

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



# Combining both sets

full_data = [train, test]
# Finding shape of each set

print('''Training set shape: {}

Test set shape: {}'''.format(train.shape, test.shape))
# Exploring variables of each set

for dataset in full_data:

    info = dataset.info()

    print(info)
# Creating a title column from the name variable

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

    unique_title_count = pd.DataFrame(dataset['Title'].value_counts())

    print(unique_title_count)
# Some titles are the same so we will place them together

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



# We have quite a few titles uncommon titles (<10) so we will replace them with 'Other'

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Dr', 'Rev', 'Major', 'Col', 'Lady', 'Capt', 'Countess', 'Jonkheer', 'Sir', 'Don', 'Dr', 'Dona'], 'Other')

    print(dataset['Title'].value_counts())
# Creating family size variable

for dataset in full_data:

    dataset['FamilySize'] = 1 + dataset['SibSp'] + dataset['Parch']
pd.DataFrame(train)
corr_matrix = train.corr()

corr_matrix['Survived'].sort_values(ascending=False)
sns.heatmap(train[['Survived', 'Pclass','Sex',  'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']].corr(),annot=True, fmt = ".2f", cmap = "plasma")
# Probability of survival depending on sex

sns.factorplot(data=train, x='Sex', y='Survived', kind='bar', palette='winter')
train[["Sex","Survived"]].groupby('Sex').mean()
# Probability of survival depending on title

sns.factorplot(data=train, x='Title', y='Survived', kind='bar', palette='winter')
title_surv_df = train[["Title","Survived"]].groupby('Title').mean().sort_values(by='Survived', ascending=False)

title_surv_df.style.background_gradient(subset=['Survived'], cmap='Greens')
fare_under_150 = train[train['Fare'] < 150]



sns.boxplot(y=fare_under_150['Fare'], x=train['Survived'], palette='winter')
fare_surv_df = train[["Survived","Fare"]].groupby('Survived').mean().sort_values(by='Survived', ascending=False)

pd.DataFrame(fare_surv_df['Fare'].rename('Mean Fare Paid'))
# Probability of survival depending on location embarked from

sns.factorplot(data=train, x='Embarked', y='Survived', kind='bar', palette='winter')
emb_surv_df = train[["Embarked","Survived"]].groupby('Embarked').mean().sort_values(by='Survived', ascending=False)

emb_surv_df
# Probability of survival depending on family size

sns.factorplot(data=train, x='FamilySize', y='Survived', kind='bar', palette='winter')
fam_surv_df = pd.crosstab(train['Survived'], train['FamilySize'])

fam_surv_df
sns.factorplot(x='Survived', y = 'Age',data = train, kind='violin', palette='winter')
# Finding null values

def null_percentage(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = round(data.isnull().sum().sort_values(ascending = False)/len(data)*100,2)

    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])



print('''---Training Set Nulls--- \n {} \n

---Test Set Nulls--- \n {}'''.format(null_percentage(train), null_percentage(test)))
# Now we have checked for nulls we can split the training set into X and y

X_train = train.drop(['Survived'], axis=1).copy()

y_train = train['Survived']



full_data_X = [X_train, test]
# Making a copy for test set so that we have the passengerIDs for submission

test_copy = test.copy()



# Dropping unwanted columns

for dataset in full_data_X:

    dataset.drop(['PassengerId', 'SibSp', 'Parch', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# Splitting cols into numerical and categorical

numerical_cols = ['Age', 'Fare', 'FamilySize']

categorical_cols = ['Pclass', 'Sex', 'Embarked', 'Title']



print('Numerical Columns: {}\n'.format(numerical_cols))

print('Categorical Columns: {}\n'.format(categorical_cols))
# Numerical column transformer

num_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='mean')),

    ('scaler', StandardScaler())

])



# Categorical column transformer

cat_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore')),

])



# Preprocessing pipeline

preprocessor = ColumnTransformer(

    transformers=[

        ('num', num_transformer, numerical_cols),

        ('cat', cat_transformer, categorical_cols)

    ])





# Fitting the data and transforming the training & test set

X_train_preprocessed = preprocessor.fit_transform(X_train)

test_preprocessed = preprocessor.fit_transform(test)



pd.DataFrame(X_train_preprocessed)
X_train = pd.DataFrame(X_train_preprocessed)
# Kfold cross validation to validate possible models

k_fold = StratifiedKFold(n_splits=10)

random_state = 0
# Logistic Regression

log_reg = LogisticRegression(random_state=random_state)

log_reg.fit(X_train, y_train)



# Using cross validation

log_reg_cv = cross_val_score(log_reg, X_train, y = y_train, scoring = "accuracy", cv = k_fold, n_jobs=5)

log_reg_cv_mean = log_reg_cv.mean()

log_reg_cv_mean
# K-NN

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)



# Using cross validation

knn_cv = cross_val_score(knn, X_train, y = y_train, scoring = "accuracy", cv = k_fold, n_jobs=5)

knn_cv_mean = knn_cv.mean()

knn_cv_mean
# Kernel SVM

k_svm = SVC(kernel='rbf', random_state=0)

k_svm.fit(X_train, y_train)



# Using cross validation

k_svm_cv = cross_val_score(k_svm, X_train, y = y_train, scoring = "accuracy", cv = k_fold, n_jobs=5)

k_svm_cv_mean = k_svm_cv.mean()

k_svm_cv_mean
# Naive Bayes

naive_bayes = GaussianNB()

naive_bayes.fit(X_train, y_train)



# Using cross validation

naive_bayes_cv = cross_val_score(naive_bayes, X_train, y = y_train, scoring = "accuracy", cv = k_fold, n_jobs=5)



naive_bayes_cv_mean = naive_bayes_cv.mean()

naive_bayes_cv_mean
# Random Forest

rf = RandomForestClassifier(n_estimators=100, random_state=0)

rf.fit(X_train, y_train)



# Using cross validation

rf_cv = cross_val_score(rf, X_train, y = y_train, scoring = "accuracy", cv = k_fold, n_jobs=-1)

print(rf_cv.mean())



# Grid Search

param_grid = [{'bootstrap': [True, False],

 'max_depth': [10, 20, 30, 40, 50, None],

 'max_features': ['auto', 'sqrt'],

 'min_samples_leaf': [1, 2, 4],

 'min_samples_split': [5, 10, 12],

 'n_estimators': [100, 200, 300, 350]},

]



rf_2 = RandomForestClassifier(random_state=random_state)



grid_search = GridSearchCV(rf_2, param_grid, cv=5, scoring='neg_mean_squared_error')



grid_search.fit(X_train, y_train)
rf_best_params = grid_search.best_params_

rf_best_params
# Using best parameters for random forest

rf3 = RandomForestClassifier(bootstrap= True,

 max_depth= 20,

 max_features= 'auto',

 min_samples_leaf= 1,

 min_samples_split= 10,

 n_estimators= 300,

random_state=random_state)

rf3.fit(X_train, y_train)



# Using cross validation

rf3_cv = cross_val_score(rf3, X_train, y = y_train, scoring = "accuracy", cv = k_fold, n_jobs=5)

rf3_cv_mean = rf3_cv.mean()

rf3_cv_mean
# XGBoost

xgb = XGBClassifier()

xgb.fit(X_train, y_train)



# Using cross validation

xgb_cv = cross_val_score(xgb, X_train, y = y_train, scoring = "accuracy", cv = k_fold, n_jobs=5)

xgb_cv_mean = xgb_cv.mean()

xgb_cv_mean
# Creating dataframe showing accuracy scores of each model

compare = {'Model': ['Logistic Regression', 'K-NN', 'Kernel SVM', 'Naive Bayes', 'Random Forest', 'XGBoost'],

          'Cross validation score': [log_reg_cv_mean, knn_cv_mean, k_svm_cv_mean, naive_bayes_cv_mean, rf3_cv_mean, xgb_cv_mean]}

compare_df = pd.DataFrame(data=compare)

 

# Visualising best scores

compare_df.style.bar(subset=['Cross validation score'], color='#d65f5f')
# Using our best model (Random Forest) to predict on test set

rf3_test_predictions = rf3.predict(test_preprocessed)
submission = pd.DataFrame({'PassengerId':test_copy['PassengerId'],'Survived':rf3_test_predictions})



#Visualising the first 5 rows

submission.head()
#Converting DataFrame to a csv file

#This is saved in the same directory as your notebook

titanic_predictions = 'Titanic Predictions 2.csv'



submission.to_csv(titanic_predictions,index=False)



print('Saved file: ' + titanic_predictions)