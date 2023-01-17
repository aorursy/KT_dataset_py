# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the read-only "../input/" directory

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#!pip install lightgbm

from numpy import mean, std

import seaborn as sns

from matplotlib import *

from matplotlib import pyplot as plt



from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score



from xgboost                          import XGBClassifier

from catboost                         import CatBoostClassifier

from lightgbm                         import LGBMClassifier

from sklearn.ensemble                 import VotingClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

train_data = pd.read_csv('/kaggle/input/hcareanalytics/train.csv')

test_data = pd.read_csv('/kaggle/input/hcareanalytics/test.csv')

#sample_submission = pd.read_csv('/kaggle/input/topicmodel/healthcare/sample_submission.csv')

train_data.columns = train_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

test_data.columns = test_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
print('Train Data Shape: ', train_data.shape)

print('Test Data Shape: ', test_data.shape)

train_data.head()
train_data.dtypes
train_data.isnull().sum()
train_data.nunique()
train_data.columns
# Unique values for all the columns

for col in train_data.columns[~(train_data.columns.isin(['case_id', 'patientid', 'admission_deposit']))].tolist():

    print(" Unique Values --> " + col, ':', len(train_data[col].unique()), ': ', train_data[col].unique())
i = 1

for column in train_data.columns[~(train_data.columns.isin(['case_id', 'patientid', 'admission_deposit']))].tolist():

    plt.figure(figsize = (60, 10))

    plt.subplot(4, 4, i)

    sns.barplot(x = train_data[column].value_counts().index, y = train_data[column].value_counts())

    i += 1

    plt.show()
sns.boxplot(x = 'visitors_with_patient', data = train_data, orient = 'v' )

sns.despine()
plt.figure(figsize = (20, 6))

sns.barplot(x = train_data.groupby(['severity_of_illness'])['visitors_with_patient'].value_counts().index, y = train_data.groupby(['severity_of_illness'])['visitors_with_patient'].value_counts())

plt.xticks(rotation = 90)

sns.despine()
train_data.isnull().sum()
train_data['city_code_patient'] = train_data['city_code_patient'].fillna(17.0)

test_data['city_code_patient'] = test_data['city_code_patient'].fillna(17.0)

train_data = train_data.fillna('NaN')

test_data = test_data.fillna('NaN')





for column in train_data.columns[~(train_data.columns.isin(['case_id', 'stay']))].tolist():



    le = LabelEncoder()



    if column == 'city_code_patient':

        train_data['city_code_patient'] = train_data['city_code_patient'].astype('str')

        test_data['city_code_patient'] = test_data['city_code_patient'].astype('str')

        train_data['city_code_patient'] = le.fit_transform(train_data['city_code_patient'])

        test_data['city_code_patient'] = le.transform(test_data['city_code_patient'])

    

    elif column == 'bed_grade':

        bedGrade = {1: '1',2: '2', 3: '3', 4: '4', np.nan: '5'}

        train_data['bed_grade'] = train_data['bed_grade'].map(bedGrade)

        test_data['bed_grade'] = test_data['bed_grade'].map(bedGrade)

        train_data['bed_grade'] = train_data['bed_grade'].fillna('NaN')

        test_data['bed_grade'] = test_data['bed_grade'].fillna('NaN')

    

    else:

        train_data[column] = le.fit_transform(train_data[column])

        test_data[column] = le.fit_transform(test_data[column])
train_data.head()
train_data.isnull().sum()
ss = StandardScaler()



for column in train_data.columns[~(train_data.columns.isin(['case_id', 'stay']))].tolist():

    train_data[[column]] = ss.fit_transform(train_data[[column]])

    test_data[[column]] = ss.fit_transform(test_data[[column]])
# Partitioning the features and the target



X = train_data[train_data.columns[~(train_data.columns.isin(['case_id', 'stay']))].tolist()].values

y = train_data['stay'].values
# kfold, scores = KFold(n_splits = 6, shuffle = True, random_state = 22), list()

# for train, test in kfold.split(X):

#     X_train, X_test = X[train], X[test]

#     y_train, y_test = y[train], y[test]

    

#     model = LGBMClassifier(random_state = 0, max_depth = 6, n_estimators = 200, bagging_fraction=0.9, feature_fraction=0.9, subsample_freq = 2,importance_type = "gain",verbosity = -1, max_bin = 60,num_leaves = 300,boosting_type = 'dart',learning_rate=0.1, scale_pos_weight=2.5)

#     model.fit(X_train, y_train)

#     preds = model.predict(X_test)

#     score = accuracy_score(y_test, preds)

#     scores.append(score)

#     print('Validation Accuracy:', score)

# print("Average Validation Accuracy: ", sum(scores)/len(scores))
# #### using kfold gridserch



# from sklearn.model_selection import RandomizedSearchCV 

# from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve, cross_val_score

# lgb = LGBMClassifier(random_state =1, shuffle = True)

# kfold = StratifiedKFold(n_splits=6)



# lgb_param_grid_best = {'learning_rate':[0.095], 

#                   'reg_lambda':[0.4],

#                   'gamma': [1],

#                   'subsample': [0.6],

#                   'max_depth': [6],

#                   'n_estimators': [1000]

#               }



# gs_lgb = GridSearchCV(lgb, param_grid = lgb_param_grid_best, cv=kfold, n_jobs= -1, verbose = 1)



# gs_lgb.fit(X,y)



# lgb_best = gs_lgb.best_estimator_

# print(f'LGB GridSearch best params: {gs_lgb.best_params_}')

# print(f'LGB GridSearch best score: {gs_lgb.best_score_}')



# predictions = gs_lgb.predict(test_data[test_data.columns[~(test_data.columns.isin(['case_id']))].tolist()].values)

# submission = pd.DataFrame({'case_id': test_data['case_id'], 'Stay': predictions.ravel()})

# submission.to_csv('av_healthcare_v1.csv', index = False)

# submission.head()
lgb1 = LGBMClassifier(  bagging_fraction=1, feature_fraction=0.9, subsample_freq = 2,importance_type = "gain",verbosity = -1, max_bin = 60,num_leaves = 300,boosting_type = 'dart',learning_rate=0.15, n_estimators=494, max_depth=5, scale_pos_weight=2.5)





lgb1.fit(X,y)

#y_pred_lgb1 = lgb1.predict(test).astype(int)   

print("Accuracy_Score:",lgb1.score(X,y))

 

predictions = lgb1.predict(test_data[test_data.columns[~(test_data.columns.isin(['case_id']))].tolist()].values)

submission = pd.DataFrame({'case_id': test_data['case_id'], 'Stay': predictions.ravel()})

submission.to_csv('av_healthcare_v1.csv', index = False)

submission.head()



# Accuracy_Score: 0.4392629020405856   best till now
submission.to_csv('av_healthcare_v1.csv', index = False)

submission.head()
# kfold, scores = KFold(n_splits = 6, shuffle = True, random_state = 0), list()

# for train, test in kfold.split(X):

#     X_train, X_test = X[train], X[test]

#     y_train, y_test = y[train], y[test]

    

#     model = LGBMClassifier(learning_rate=0.1,random_state = 0, max_depth = 6, n_estimators = 200, verbose = 100)

#     model.fit(X_train, y_train)

#     preds = model.predict(X_test)

#     score = accuracy_score(y_test, preds)

#     scores.append(score)

#     print('Validation Accuracy:', score)

# print("Average Validation Accuracy: ", sum(scores)/len(scores))
# predictions = model.predict(test_data[test_data.columns[~(test_data.columns.isin(['case_id']))].tolist()].values)

# submission = pd.DataFrame({'case_id': test_data['case_id'], 'Stay': predictions.ravel()})

# submission.to_csv('av_healthcare_v1.csv', index = False)

# submission.head()