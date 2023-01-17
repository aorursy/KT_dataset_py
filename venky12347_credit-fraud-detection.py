# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Generic library

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df=pd.read_csv('/kaggle/input/credit-card-fraud/credit_train.csv')

test_df=pd.read_csv('/kaggle/input/credit-card-fraud/credit_test.csv')
train_df.head(3)
print("Train size : rows",train_df.shape[0]," and columns",train_df.shape[1])

print("Test size : rows",test_df.shape[0]," and columns",test_df.shape[1])
train_df.columns
test_df.columns
train_df.columns.difference(test_df.columns)
train_df["source"] = "train"

test_df["source"] = "test"

df = pd.concat([train_df,test_df])
df.dtypes
df.describe()
df.isnull().sum().max()
# The classes are heavily skewed we need to solve this issue later.

print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
colors = ["#0101DF", "#DF0101"]

sns.countplot('Class', data=df, palette=colors)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)

from sklearn.preprocessing import StandardScaler, RobustScaler



# RobustScaler is less prone to outliers.



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))





df.drop(['Amount'], axis=1, inplace=True)
scaled_amount = df['scaled_amount']





df.drop(['scaled_amount'], axis=1, inplace=True)

df.insert(0, 'scaled_amount', scaled_amount)



# Amount is Scaled!



df.head()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import KFold, StratifiedKFold



print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')



X = df.drop('Class', axis=1)

y = df['Class']



sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)



for train_index, test_index in sss.split(X, y):

    print("Train:", train_index, "Test:", test_index)

    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]

    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]



# We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.

# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)



# Check the Distribution of the labels





# Turn into an array

original_Xtrain = original_Xtrain.values

original_Xtest = original_Xtest.values

original_ytrain = original_ytrain.values

original_ytest = original_ytest.values



# See if both the train and test label distribution are similarly distributed

train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)

test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

print('-' * 100)



print('Label Distributions: \n')

print(train_counts_label/ len(original_ytrain))

print(test_counts_label/ len(original_ytest))
df['Class'] = df['Class'].str.replace("'","")
df['Class'].value_counts()
# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.



# Lets shuffle the data before creating the subsamples



df = df.sample(frac=1)



# amount of fraud classes 492 rows.

fraud_df = df.loc[df['Class'] == 1]

non_fraud_df = df.loc[df['Class'] == 0][:492]



normal_distributed_df = pd.concat([fraud_df, non_fraud_df])



# Shuffle dataframe rows

new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df = df

new_df.head()
fraud_df = df.loc[df['Class'] == 1]

non_fraud_df = df.loc[df['Class'] == 0][:492]
normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
new_df = normal_distributed_df.sample(frac=1, random_state=42)
new_df=df

new_df.head()
print('Distribution of the Classes in the subsample dataset')

print(new_df['Class'].value_counts()/len(new_df))







sns.countplot('Class', data=new_df, palette=colors)

plt.title('Equally Distributed Classes', fontsize=14)

plt.show()
# Undersampling before cross validating (prone to overfit)

X = new_df.drop(['Class','source'], axis=1)

y = new_df['Class']
# Our data is already scaled we should split our training and test sets

from sklearn.model_selection import train_test_split



# This is explicitly used for undersampling.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Turn the values into an array for feeding the classification algorithms.

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
# Let's implement simple classifiers

# Classifier Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import collections



classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}
#our scores are getting even high scores even when applying cross validation.

from sklearn.model_selection import cross_val_score

for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train, y_train, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV





# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}







grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train, y_train)

# We automatically get the logistic regression with the best parameters.

log_reg = grid_log_reg.best_estimator_



knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}



grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)

grid_knears.fit(X_train, y_train)

# KNears best estimator

knears_neighbors = grid_knears.best_estimator_



# Support Vector Classifier

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(X_train, y_train)



# SVC best estimator

svc = grid_svc.best_estimator_



# DecisionTree Classifier

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,7,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(X_train, y_train)



# tree best estimator

tree_clf = grid_tree.best_estimator_
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, RandomizedSearchCV





print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))

print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))



# List to append the score and then find the average

accuracy_lst = []

precision_lst = []

recall_lst = []

f1_lst = []

auc_lst = []



# Classifier with optimal parameters

# log_reg_sm = grid_log_reg.best_estimator_

log_reg_sm = LogisticRegression()









rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)





# Implementing SMOTE Technique 

# Cross Validating the right way

# Parameters

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

for train, test in sss.split(original_Xtrain, original_ytrain):

    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) # SMOTE happens during Cross Validation not before..

    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])

    best_est = rand_log_reg.best_estimator_

    prediction = best_est.predict(original_Xtrain[test])

    

    accuracy_lst.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))

    precision_lst.append(precision_score(original_ytrain[test], prediction))

    recall_lst.append(recall_score(original_ytrain[test], prediction))

    f1_lst.append(f1_score(original_ytrain[test], prediction))

    auc_lst.append(roc_auc_score(original_ytrain[test], prediction))

    

print('---' * 45)

print('')

print("accuracy: {}".format(np.mean(accuracy_lst)))

print("precision: {}".format(np.mean(precision_lst)))

print("recall: {}".format(np.mean(recall_lst)))

print("f1: {}".format(np.mean(f1_lst)))

print('---' * 45)
labels = ['No Fraud', 'Fraud']

smote_prediction = best_est.predict(original_Xtest)

print(classification_report(original_ytest, smote_prediction, target_names=labels))


y_score = best_est.decision_function(original_Xtest)
average_precision = average_precision_score(original_ytest, y_score)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))