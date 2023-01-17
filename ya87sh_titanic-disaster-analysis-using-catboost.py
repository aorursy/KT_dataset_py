# Base Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Setting the seaborn style

sns.set_style("darkgrid")



# SciKit Learn Library

import sklearn



# For Spliting the Dataset

from sklearn.model_selection import train_test_split



# For Scaling the numeric predictors

from sklearn.preprocessing import StandardScaler



# For GridSearch()

from sklearn.model_selection import GridSearchCV



# For Creating K-folds

from sklearn.model_selection import KFold



# For Classification Report

from sklearn.metrics import classification_report



from sklearn import metrics



# supress warnings

import warnings 

warnings.filterwarnings('ignore')





# CatBoost library

import catboost

# Importing data for Analysis

submission = pd.read_csv("../input/titanic/test.csv")

validation = pd.read_csv("../input/titanic/test.csv")

dataset = pd.read_csv("../input/titanic/train.csv")
# Dropping irrelevant columns form train and test set

dataset = dataset.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1)

validation = submission.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1)
# Let's view the dataset

dataset.head()
dataset.info()
# Countplot: to check the balance of Survived Class

sns.countplot(dataset['Survived'])

plt.title("Survived Class Histogram")

plt.show



# Displaying the Counts

dataset['Survived'].value_counts()
# Countplot: Survived vs Sex

sns.countplot(dataset['Sex'], hue=dataset['Survived'])

plt.title("Sex: Survived vs Dead")

plt.show()
# Countplot: Survived vs Passenger Class

sns.countplot(dataset['Pclass'], hue=dataset['Survived'])

plt.title("Passenger Class: Survived vs Dead")

plt.show()
# Countplot: Survived vs Embarked



# Increasing figure size

plt.figure(figsize=(12,6))



# Initiating first figure

plt.figure(1)



# Countplot - Embarked: Survive vs Dead

plt.subplot(1,2,1)

sns.countplot(dataset['Embarked'], hue=dataset['Survived'])

plt.title("Embarked: Survived vs Dead")



# Barplot - Embarked Survival Rate

plt.subplot(1,2,2)

sns.barplot(x = 'Embarked', y = 'Survived', data = dataset)

plt.title("Embarked Survival Rate")

plt.show()
# Spliting Dataset into Train and Test



from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(dataset, train_size = 0.7, test_size = 0.3, random_state = 0)
# Increasing figure size

plt.figure(figsize=(12,6))



# Initiating first figure

plt.figure(1)



# Boxplot - Age

plt.subplot(1,2,1)

sns.boxplot(y = df_train['Age'])

plt.title("Age - Train")



# Boxplot - Fare

plt.subplot(1,2,2)

sns.boxplot(y = df_train['Fare'])

plt.title("Fare - Train")



plt.show()
# Increasing figure size

plt.figure(figsize=(12,6))



# Initiating first figure

plt.figure(1)



# Boxplot - Age

plt.subplot(1,2,1)

sns.boxplot(y = df_test['Age'])

plt.title("Age - Test")



# Boxplot - Fare

plt.subplot(1,2,2)

sns.boxplot(y = df_test['Fare'])

plt.title("Fare - Test")



plt.show()
df_train['Age'].describe()
df_train['Fare'].describe()
df_test['Age'].describe()
df_test['Fare'].describe()
# Removing Outliers from Age

Q1 = df_train['Age'].quantile(0.25)

Q3 = df_train['Age'].quantile(0.75)

IQR = Q3 - Q1



# Removing Outliers from Train

df_train['Age'] = df_train['Age'].drop(df_train[(df_train['Age'] < (Q1 - 1.5 * IQR)) | (df_train['Age'] > (Q3 + 1.5 * IQR))].index)
# Removing Outliers from Fare

Q1 = df_train['Fare'].quantile(0.25)

Q3 = df_train['Fare'].quantile(0.75)

IQR = Q3 - Q1



# Removing Outliers from Train

df_train['Fare'] = df_train['Fare'].drop(df_train[(df_train['Fare'] < (Q1 - 1.5 * IQR)) | (df_train['Fare'] > (Q3 + 1.5 * IQR))].index)
# Removing Outliers from Age

Q1 = df_test['Age'].quantile(0.25)

Q3 = df_test['Age'].quantile(0.75)

IQR = Q3 - Q1



# Removing Outliers from Test

df_test['Age'] = df_test['Age'].drop(df_test[(df_test['Age'] < (Q1 - 1.5 * IQR)) | (df_test['Age'] > (Q3 + 1.5 * IQR))].index)
# Removing Outliers from Fare

Q1 = df_test['Fare'].quantile(0.25)

Q3 = df_test['Fare'].quantile(0.75)

IQR = Q3 - Q1



# Removing Outliers from Train

df_test['Fare'] = df_test['Fare'].drop(df_test[(df_test['Fare'] < (Q1 - 1.5 * IQR)) | (df_test['Fare'] > (Q3 + 1.5 * IQR))].index)
# Increasing figure size

plt.figure(figsize=(12,6))



# Initiating first figure

plt.figure(1)



# Boxplot - Age

plt.subplot(1,2,1)

sns.boxplot(y = df_train['Age'])

plt.title("Age")



# Boxplot - Fare

plt.subplot(1,2,2)

sns.boxplot(y = df_train['Fare'])

plt.title("Fare")



plt.show()
# Increasing figure size

plt.figure(figsize=(12,6))



# Initiating first figure

plt.figure(1)



# Boxplot - Age

plt.subplot(1,2,1)

sns.boxplot(y = df_test['Age'])

plt.title("Age")



# Boxplot - Fare

plt.subplot(1,2,2)

sns.boxplot(y = df_test['Fare'])

plt.title("Fare")



plt.show()
# Printing how many missing values are there in each column in Train

df_train.isnull().sum()
# Printing how many missing values are there in each column in Test

df_test.isnull().sum()
# Imputing missing values in Age and Fare with Mean of Columns - Train

df_train['Age'].fillna(df_train['Age'].mean(), inplace=True)

df_train['Fare'].fillna(df_train['Fare'].mean(), inplace=True)
# Imputing missing values in Age and Fare with Mean of Columns - Test

df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)

df_test['Fare'].fillna(df_test['Fare'].mean(), inplace=True)
# For Embarked we will be dropping the rows with missing values

df_train = df_train.dropna(axis = 0,subset=['Embarked'])

df_test = df_test.dropna(axis = 0,subset=['Embarked'])
# First we will create Column family_size basis SibSp and Parch

df_train['family_size'] = df_train.SibSp + df_train.Parch+1

df_test['family_size'] = df_test.SibSp + df_test.Parch+1

validation['family_size'] = validation.SibSp + validation.Parch+1
# From family_size we will derive family_group

def family_group(size):

    b = ''

    if (size <= 1):

        b = 'alone'

    elif (size <= 4):

        b = 'small'

    else:

        b = 'large'

    return b



# Creating family_group by its size (in both dataset and Submission)

df_train['family_group'] = df_train['family_size'].map(family_group)

df_test['family_group'] = df_test['family_size'].map(family_group)

validation['family_group'] = validation['family_size'].map(family_group)



# Drop family_size

df_train.drop(columns=['family_size'], inplace = True)

df_test.drop(columns=['family_size'], inplace = True)

validation.drop(columns=['family_size'], inplace = True)
# Function to bin Passengers basis their Age

def age_group(age):

    ag_grp= ''

    

    if age <= 8:

        ag_grp = "child"

    elif age <= 19:

        ag_grp = "teenager"

    elif age <= 40:

        ag_grp = "Adult"

    elif age <= 60:

        ag_grp = "Senior"

    else:

        ag_grp = "Old"

    return ag_grp



df_train['age_group'] = df_train['Age'].map(age_group)

df_test['age_group'] = df_test['Age'].map(age_group)

validation['age_group'] = submission['Age'].map(age_group)



# Drop Age columns

df_train.drop(columns=['Age'], inplace = True)

df_test.drop(columns=['Age'], inplace = True)

validation.drop(columns=['Age'], inplace = True)
# Let's look our dataset again

df_train.head()
# First, let's understand which columns need treatment

df_train.info()
# Scaling Numerical Features

from sklearn.preprocessing import StandardScaler



# Creating StandardScaler Object

sc = StandardScaler()



# Selecting Numeric Columns

numeric_vars = ['Pclass','SibSp','Parch','Fare']



# Scaling Train Dataframe

df_train[numeric_vars] = sc.fit_transform(df_train[numeric_vars])



# Transforming Test Dataframe

df_test[numeric_vars] = sc.fit_transform(df_test[numeric_vars])
# Assigning Predictors and Response in X and y

y_train = df_train.pop('Survived')

X_train = df_train



y_test = df_test.pop('Survived')

X_test = df_test
# Collecting 'Column Indexes' of all the Categorical Columns in X_train

categorical_features_indices = np.where(X_train.dtypes == np.object)[0]



# Printing Categorical Features

categorical_features_indices
# Initialising Base CatBoost Model



# 1. Base Model

cb = catboost.CatBoostClassifier(loss_function='Logloss',

                         eval_metric='Logloss',

                         boosting_type='Ordered', # use permutations

                         random_seed=2405, 

                         use_best_model=True,

                         one_hot_max_size = 6,

                         silent=True)







# 2. Fitting the Model

cb.fit(X_train,y_train,cat_features=categorical_features_indices, eval_set=(X_test, y_test))



# 3. Initial Prediction of Results

y_pred = cb.predict(X_test)



# 4. Predicting Probabilites

y_pred_proba = cb.predict_proba(X_test)



# 5. Printing Classification Report

print(classification_report(y_test, y_pred))

# Printing ROC-AUC score

from sklearn import metrics

metrics.roc_auc_score(y_test,y_pred_proba[:,1])



fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba[:,1])

roc_auc = metrics.auc(fpr, tpr)



plt.figure(figsize=(12,8))

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)

plt.legend(loc = 'lower right')



plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
# Creating a Parameter Grid to be Tuned

param_grid = {'depth':[1,3,5,10],

              'iterations':[100,200,300],

              'learning_rate':[0.1,0.03,0.01], 

              'l2_leaf_reg':[1,3,5],

              'border_count':[5,10,20]

          }



# Creating Hyper-param tuned model with 5 models

#from sklearn.model_selection import GridSearchCV



model_cv = GridSearchCV(estimator = cb, 

                        param_grid = param_grid,

                        cv = 5, 

                        verbose = 1) 
# Fitting the Hyperparameter-tuned Model

#model_cv.fit(X_train,y_train,cat_features=categorical_features_indices,eval_set=(X_test, y_test))
# Getting list of best parameters

#model_cv.best_params_
# Making the Final CatBoost model

cb_final = catboost.CatBoostClassifier(

                         loss_function='Logloss',

                         eval_metric='Logloss',

                         boosting_type='Ordered', # use permutations

                         random_seed=240, 

                         use_best_model=True,

                         one_hot_max_size = 5,

                         silent=True,

                         depth = 3,

                         iterations = 300,

                         learning_rate = 0.03, 

                         l2_leaf_reg = 5,

                         border_count = 5

                        )
# Fitting the Final Model, Final Prediction and Classification report

cb_final.fit(X_train,y_train,cat_features=categorical_features_indices, eval_set=(X_test, y_test))



# Final Prediction of Results

y_pred = cb_final.predict(X_test)



# Final Prediction of Probabilities

y_pred_proba = cb_final.predict_proba(X_test)



# Printing Classification Report

print(classification_report(y_test, y_pred))
# Printing ROC-AUC score

from sklearn import metrics

metrics.roc_auc_score(y_test,y_pred_proba[:,1])



fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba[:,1])

roc_auc = metrics.auc(fpr, tpr)



plt.figure(figsize=(12,8))

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)

plt.legend(loc = 'lower right')



plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
# Predicting final results

y_pred = cb_final.predict(validation)



# Creating final Submission

submission = pd.DataFrame({

        "PassengerId": submission["PassengerId"],

        "Survived": y_pred

    })
#submission.to_csv('mycsvfile.csv',index=False)