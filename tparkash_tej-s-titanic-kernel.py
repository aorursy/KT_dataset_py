# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

df = pd.read_csv("../input/train.csv")

df = df.drop(columns=["PassengerId", "Cabin", "Ticket"], axis=1)
df['Salutation'] = df['Name'].apply(lambda x : x.split(',')[1].split()[0])

df = df.drop(columns=["Name"], axis=1)
def impute_na(df, variable):

    # random sampling

    df[variable+'_random'] = df[variable]        

    # extract the random sample to fill the na

    random_sample = df[variable].dropna().sample(df[variable].isnull().sum(), random_state=0, replace=True)    

    # pandas needs to have the same index in order to merge datasets

    random_sample.index = df[df[variable].isnull()].index 

    df.loc[df[variable].isnull(), variable+'_random'] = random_sample

    df.drop(columns=[variable], axis=1, inplace=True)

impute_na(df, 'Embarked')
df = pd.get_dummies(df, columns=['Sex', 'Embarked_random', 'Pclass', 'SibSp', 'Parch', 'Salutation'], drop_first=True)
df = df.drop(columns=["Salutation_Mlle.", "Salutation_Mme.", "Salutation_Don.","Salutation_Sir.","Salutation_Jonkheer.", "Salutation_the", "Salutation_Lady."], axis=1)
# Function to have Weight of evidence encoding

'''def function_woe(df, variable, target):

    prob_df = df.groupby([variable])[target].mean()

    prob_df = pd.DataFrame(prob_df)

    df.dropna(subset=[variable], inplace=True)

    prob_df.loc[prob_df[target] == 0, target] = 0.00001

    prob_df['Non_target'] = 1-prob_df[target]

    prob_df['WoE'] = np.log(prob_df[target]/prob_df['Non_target'])

    #print(prob_df)

    woe_labels = prob_df['WoE'].to_dict()

    df[variable+"_woe"] = df[variable].map(woe_labels)

    df.drop(columns=[variable], axis=1, inplace=True)

    return woe_labels

s_map_woe = function_woe(df, 'Sex', 'Survived')

e_map_woe = function_woe(df, 'Embarked_random', 'Survived')

#c_map_woe = function_woe(df, 'Cabin_random', 'Survived')

p_map_woe = function_woe(df, 'Pclass', 'Survived')

#sp_map_woe = function_woe(df, 'SibSp', 'Survived')

#pa_map_woe = function_woe(df, 'Parch', 'Survived')'''
# Frequncy encoding would be helpful here, as suvived chances could be determined by Cabin with most frequeny

#count_dict = df.Cabin_random.value_counts().to_dict()

#df['Cabin_count'] =  df['Cabin_random'].map(count_dict)
train = df.iloc[:,2].values.reshape(-1, 1)
# lets do scaling on required column

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

train = sc.fit_transform(train)
df_test = pd.concat([df, 

                    pd.DataFrame(train, columns=['Fare_s']),], axis=1)

df = df_test

df.drop(columns=['Fare'], axis=1, inplace=True)
df.shape
# create a test and train split

train = df[df['Age'].notnull()] # WHERE AGE IS NOT NULL

test = df[df['Age'].isnull()]  # WHERE AGE IS NULL

# Segregate X_train and Y_train

X_train = train.iloc[:, 2:28].values

Y_train = train.iloc[:, 1].values

X_test = test.iloc[:, 2:28].values
X_test.shape
# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

rf_test = RandomForestRegressor(n_estimators = 350, random_state = 0)

rf_test.fit(X_train, Y_train)
Y_result1 = rf_test.predict(X_test)

Y_result1 = np.floor(Y_result1)

Y_result1.astype(int)
# create a series and attached to original df

dataset = pd.Series(Y_result1)

dataset.index = df[df['Age'].isnull()].index

df['Age_lr'] = df['Age']

df.loc[df['Age'].isnull(), "Age_lr"] = dataset
df = df.drop(columns=["Age"], axis=1)
#df = df.drop(columns=["Age_lr"], axis=1)

df.head()
df['Age_test'] = sc.transform((df.Age_lr).values.reshape(-1, 1))
df = df.drop(columns=['Age_lr'], axis=1)

final_df = df
models = []

acc = []

precision = []

recall = []

f1 = []

final_df.head()
# lets scale Estimated salary

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



X = final_df.iloc[:, 1:35].values

y = final_df.iloc[:, 0].values



X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 0)

lr.fit(X_train, y_train)

models.append('Logistic regression')
print(lr.coef_, lr.intercept_, lr.n_iter_)

y_pred = lr.predict(X_test)

from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, precision_score, recall_score)

print('Confusion Matrix for LR: \n',confusion_matrix(y_test, lr.predict(X_test)))

print('Accuracy for LR: \n',accuracy_score(y_test, lr.predict(X_test)))

acc.append(accuracy_score(y_test, lr.predict(X_test)))

print('Precision for LR: \n',precision_score(y_test, lr.predict(X_test)))

precision.append(precision_score(y_test, lr.predict(X_test)))

print('Recall for LR: \n',recall_score(y_test, lr.predict(X_test)))

recall.append(recall_score(y_test, lr.predict(X_test)))

print('f1_score for LR: \n',f1_score(y_test, lr.predict(X_test)))

f1.append(f1_score(y_test, lr.predict(X_test)))
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier(criterion = 'entropy', random_state=0)

dt.fit(X_train, y_train)

models.append('Descision tree')
y_pred = dt.predict(X_test)

print('Confusion Matrix for DTrees: \n',confusion_matrix(y_test, dt.predict(X_test)))

print('Accuracy for DTrees: \n',accuracy_score(y_test, dt.predict(X_test)))

acc.append(accuracy_score(y_test, dt.predict(X_test)))

print('Precision for DTrees: \n',precision_score(y_test, dt.predict(X_test)))

precision.append(precision_score(y_test, dt.predict(X_test)))

print('Recall for DTrees: \n',recall_score(y_test, dt.predict(X_test)))

recall.append(recall_score(y_test, dt.predict(X_test)))

print('f1_score for DTrees: \n',f1_score(y_test, dt.predict(X_test)))

f1.append(f1_score(y_test, dt.predict(X_test)))

#dt.predict_proba(X_test)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 350, criterion = 'entropy', 

                                    random_state = 0)

rf.fit(X_train, y_train)

models.append('Random Forest')
y_pred = rf.predict(X_test)

print('Confusion Matrix for RF: \n',confusion_matrix(y_test, rf.predict(X_test)))

print('Accuracy for RF: \n',accuracy_score(y_test, rf.predict(X_test)))

acc.append(accuracy_score(y_test, rf.predict(X_test)))

print('Precision for RF: \n',precision_score(y_test, rf.predict(X_test)))

precision.append(precision_score(y_test, rf.predict(X_test)))

print('Recall for RF: \n',recall_score(y_test, rf.predict(X_test)))

recall.append(recall_score(y_test, rf.predict(X_test)))

print('f1_score for RF: \n',f1_score(y_test, rf.predict(X_test)))

f1.append(f1_score(y_test, rf.predict(X_test)))
from sklearn.svm import SVC

svc = SVC(kernel = 'rbf', random_state = 0, probability=True)

svc.fit(X_train, y_train)

models.append('SVC')
y_pred = svc.predict(X_test)

print('Confusion Matrix for SVM: \n',confusion_matrix(y_test, svc.predict(X_test)))

print('Accuracy for SVM: \n',accuracy_score(y_test, svc.predict(X_test)))

acc.append(accuracy_score(y_test, svc.predict(X_test)))

print('Precision for SVM: \n',precision_score(y_test, svc.predict(X_test)))

precision.append(precision_score(y_test, svc.predict(X_test)))

print('Recall for SVM: \n',recall_score(y_test, svc.predict(X_test)))

recall.append(recall_score(y_test, svc.predict(X_test)))

print('f1_score for SVM: \n',f1_score(y_test, svc.predict(X_test)))

f1.append(f1_score(y_test, svc.predict(X_test)))

#svc.predict_proba(X_test)
# Fitting AdaBoost Classification to the Training set

from sklearn.ensemble import AdaBoostClassifier

adb = AdaBoostClassifier(base_estimator=dt, n_estimators=50, 

                         algorithm='SAMME.R', random_state=40)

adb.fit(X_train, y_train)

models.append('AdaBoost')
print('Confusion Matrix for ADB: \n',confusion_matrix(y_test, adb.predict(X_test)))

print('Accuracy for ADB: \n',accuracy_score(y_test, adb.predict(X_test)))

acc.append(accuracy_score(y_test, adb.predict(X_test)))

print('Precision for ADB: \n',precision_score(y_test, adb.predict(X_test)))

precision.append(precision_score(y_test, adb.predict(X_test)))

print('Recall for ADB: \n',recall_score(y_test, adb.predict(X_test)))

recall.append(recall_score(y_test, adb.predict(X_test)))

print('f1_score for ADB: \n',f1_score(y_test, adb.predict(X_test)))

f1.append(f1_score(y_test, adb.predict(X_test)))
# Fitting Voting Classifier Classification to the Training set

from sklearn.ensemble import VotingClassifier

vc = VotingClassifier(estimators=[('Logistic Regression',lr),

                                   ('SVM',svc),

                                   ('Decision Tree',dt),

                                   ('Random Forest',rf),

                                   ('AdaBoost',adb)], 

                                   voting='soft', flatten_transform=True)

vc.fit(X_train, y_train)

models.append('Average Ensemble')
print('Confusion Matrix for VC: \n',confusion_matrix(y_test, vc.predict(X_test)))

print('Accuracy for VC: \n',accuracy_score(y_test, vc.predict(X_test)))

acc.append(accuracy_score(y_test, vc.predict(X_test)))

print('Precision for VC: \n',precision_score(y_test, vc.predict(X_test)))

precision.append(precision_score(y_test, vc.predict(X_test)))

print('Recall for VC: \n',recall_score(y_test, vc.predict(X_test)))

recall.append(recall_score(y_test, vc.predict(X_test)))

print('f1_score for VC: \n',f1_score(y_test, vc.predict(X_test)))

f1.append(f1_score(y_test, vc.predict(X_test)))
model_dict = {'Models': models,

             'Accuracies': acc,

             'Precision': precision,

             'Recall': recall,

             'f1-score': f1}
model_dict = {'Models': models,

             'Accuracies': acc,

             'Precision': precision,

             'Recall': recall,

             'f1-score': f1}
mydf = pd.DataFrame(model_dict)

mydf.head()

#model_dict
from sklearn.model_selection import GridSearchCV

parameters = {"n_estimators": [100, 200, 300],

              "criterion":['gini','entropy'],

              "max_depth": [8, 16, 32],

              "min_samples_split": [10, 20, 30],

              "min_samples_leaf": [10, 20, 35],

              "min_weight_fraction_leaf": [0.1, 0.05, 0.005]}

grid_search = GridSearchCV(estimator = rf,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_

best_parameters
# Fitting Final Model on training set

from sklearn.ensemble import RandomForestClassifier

tunedRF = RandomForestClassifier(n_estimators = best_parameters["n_estimators"],

                                 criterion = best_parameters["criterion"],

                                 max_depth = best_parameters["max_depth"],

                                 min_samples_split = best_parameters["min_samples_split"],

                                 min_samples_leaf = best_parameters["min_samples_leaf"],

                                 min_weight_fraction_leaf = best_parameters["min_weight_fraction_leaf"])

tunedRF.fit(X_train, y_train)
models.append('Tuned RF')

print('Confusion Matrix for Tuned RF: \n',confusion_matrix(y_test, tunedRF.predict(X_test)))

print('Accuracy for Tuned RF: \n',accuracy_score(y_test, tunedRF.predict(X_test)))

acc.append(accuracy_score(y_test, tunedRF.predict(X_test)))

print('Precision for Tuned RF: \n',precision_score(y_test, tunedRF.predict(X_test)))

precision.append(precision_score(y_test, tunedRF.predict(X_test)))

print('Recall for Tuned RF: \n',recall_score(y_test, tunedRF.predict(X_test)))

recall.append(recall_score(y_test, tunedRF.predict(X_test)))

print('f1_score for Tuned RF: \n',f1_score(y_test, tunedRF.predict(X_test)))

f1.append(f1_score(y_test, tunedRF.predict(X_test)))
df_test = pd.read_csv("../input/test.csv")

df_test = df_test.drop(columns=["PassengerId", "Cabin", "Ticket"], axis=1)

df_test['Salutation'] = df_test["Name"].apply(lambda x: x.split(',')[1].split()[0])

df_test = df_test.drop(columns=["Name"], axis=1)
impute_na(df_test, 'Fare')
df_test = pd.get_dummies(df_test, columns=['Sex', 'Embarked', 'Pclass', 'SibSp', 'Parch', 'Salutation'], drop_first=True)
df_test.head()
train = df_test.iloc[:,1].values.reshape(-1, 1)

train = sc.fit_transform(train)

df_test = pd.concat([df_test, 

                    pd.DataFrame(train, columns=['Fare_s']),], axis=1)

df_test.drop(columns=['Fare_random'], axis=1, inplace=True)
df_test.shape
test = df_test[df_test['Age'].isnull()]  # WHERE AGE IS NULL

X_test = test.iloc[:, 1:27].values
X_test.shape
Y_result1 = rf_test.predict(X_test)

Y_result1 = np.floor(Y_result1)

Y_result1.astype(int)
dataset = pd.Series(Y_result1)

dataset.index = df_test[df_test['Age'].isnull()].index

df_test['Age_lr'] = df_test['Age']

df_test.loc[df_test['Age'].isnull(), "Age_lr"] = dataset
df_test.head()
df_test['Age_test'] = sc.transform((df_test.Age_lr).values.reshape(-1, 1))

df_test = df_test.drop(columns=['Age_lr'], axis=1)

df_test = df_test.drop(columns=['Age'], axis=1)
X_test = df_test.iloc[:, 0:28].values

X_test.shape
Y_pred = tunedRF.predict(X_test)
Y_pred
df_sub = pd.DataFrame({'PassengerId':range(892, 1310), 

                       'Survived': Y_pred})

df_sub.head()
df_sub.to_csv('gender_submission.csv', index=False)