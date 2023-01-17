# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

% matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../../LendingClub/Data"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#import sqlite3

#conn = sqlite3.connect('../input/database.sqlite')

#c = conn.cursor()
# Get data



# Disable display truncation 

pd.set_option('display.max_columns', 0)

pd.set_option('display.max_rows', 500)



df = pd.read_csv('../input/loan.csv', low_memory = False)

df.head()
# Plot Loan Status

plt.figure(figsize= (12,6))

plt.ylabel('Loan Status')

plt.xlabel('Count')

df['loan_status'].value_counts().plot(kind = 'barh', grid = True)

plt.show()
# Calculate Good and Bad Loan Status Ratio

good_loan =  len(df[(df.loan_status == 'Fully Paid') |

                    (df.loan_status == 'Current') | 

                    (df.loan_status == 'Does not meet the credit policy. Status:Fully Paid')])

print ('Good/Bad Loan Ratio: %.2f%%'  % (good_loan/len(df)*100))
df.loan_status.value_counts()
# Data Dimension

df.shape
# Drop these features for now

df.drop([    'id',

             'member_id',

             'emp_title',

             'title',

             'url',

             'zip_code',

             'verification_status',

             'home_ownership',

             'issue_d',

             'earliest_cr_line',

             'last_pymnt_d',

             'next_pymnt_d',

             'desc',

#             'pymnt_plan',

#             'initial_list_status',

#             'addr_state',

             'last_credit_pull_d', 

                                    ], axis=1, inplace=True)
# Show records number

df.count().sort_values()
# Drop columns with less than 25% data.

lack_of_data_idx = [x for x in df.count() < 887379*0.25]

df.drop(df.columns[lack_of_data_idx], 1, inplace=True)
# After Deletion

df.info()
print (df.mths_since_last_delinq.min(), df.mths_since_last_delinq.max())

print(df.mths_since_last_delinq.mean())

print(df.mths_since_last_delinq.median())
df.mths_since_last_delinq = df.mths_since_last_delinq.fillna(df.mths_since_last_delinq.median())
df.dropna(inplace=True)
df.info()
# Calculate Good and Bad Loan Status Ratio

good_loan =  len(df[(df.loan_status == 'Fully Paid') |

                    (df.loan_status == 'Current') | 

                    (df.loan_status == 'Does not meet the credit policy. Status:Fully Paid')])

print ('Good/Bad Loan Ratio: %.2f%%'  % (good_loan/len(df)*100))
# create an bad/good loan indicator feature

df['good_loan'] = np.where((df.loan_status == 'Fully Paid') |

                        (df.loan_status == 'Current') | 

                        (df.loan_status == 'Does not meet the credit policy. Status:Fully Paid'), 1, 0)
# Hot encode some categorical features 

columns = ['term', 'grade', 'sub_grade', 'emp_length', 'purpose', 'application_type','addr_state',

           'pymnt_plan', 'initial_list_status']



for col in columns:

    tmp_df = pd.get_dummies(df[col], prefix=col)

    df = pd.concat((df, tmp_df), axis=1)
# drop attributes that we hot-encoded

df.drop(['loan_status',

           'term',

           'grade',

           'sub_grade',

           'emp_length',

           'addr_state',

           'initial_list_status',

           'pymnt_plan',

           'purpose',

           'application_type'], axis=1, inplace=True)
# Rename some features to concur w/ some algorithms

df = df.rename(columns= {'emp_length_< 1 year':'emp_length_lt_1 year',

                         'emp_length_n/a':'emp_length_na'})
# Due to resource limitation, we limit data to only the first 10,000 records.

df = df[:10000]
df.head()
# Split Train/Test data

from sklearn.model_selection import train_test_split



y = df['good_loan']

X = df.ix[:, df.columns != 'good_loan']



X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=44)
# Bring in evaluator

import sklearn.metrics as mt

from sklearn.model_selection import cross_val_score



# Flatten Data

from sklearn.preprocessing import StandardScaler, RobustScaler



#std_scaler = StandardScaler()

rob_scaler = RobustScaler()



#X_train_S = std_scaler.fit_transform(X_train)

#X_test_S = std_scaler.transform(X_test)



# Use robust scaler to reduce outliers

X_train_R = rob_scaler.fit_transform(X_train)

X_test_R = rob_scaler.transform(X_test)
from sklearn.svm import SVC
# Weighted prediction feature

y_0 = len(y_train[y_train == 0])/len(y_train)

y_1 = 1 - y_0
svm_clf = SVC(class_weight={0:y_1, 1:y_0})

svm_clf.fit(X_train_R, y_train)



svm_predictions = svm_clf.predict(X_test_R) # Save prediction





#print(svm_clf.score(X_test_R, y_test))

scores = cross_val_score(svm_clf, X_test_R, y_test, cv=5)

print(scores)

print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))



print(mt.classification_report(y_test, svm_predictions))

print(mt.confusion_matrix(y_test, svm_predictions))
from imblearn.over_sampling import SMOTE



sm = SMOTE(k_neighbors=10, random_state=44, kind = 'svm')

X_res_train, y_res_train = sm.fit_sample(X_train_R, y_train)
svm_sm_clf = SVC()

svm_sm_clf.fit(X_res_train, y_res_train)



svm_sm_predictions = svm_clf.predict(X_test_R)



#print(svm_sm_clf.score(X_test_R, y_test))

scores = cross_val_score(svm_sm_clf, X_test_R, y_test, cv=5)

print(scores)

print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))



print(mt.classification_report(y_test, svm_sm_predictions))

print(mt.confusion_matrix(y_test, svm_sm_predictions))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 20)

rf.fit(X_train, y_train)

       

rf_predictions = rf.predict(X_test)



#print(rf.score(X_test, y_test))

scores = cross_val_score(rf, X_test, y_test, cv=5)

print(scores)

print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))



print(mt.classification_report(y_test, rf_predictions))

print(mt.confusion_matrix(y_test, rf_predictions))
from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb.fit(X_train, y_train)



xgb_predictions = xgb.predict(X_test)

                            

#print(xgb.score(X_test, y_test))

scores = cross_val_score(xgb, X_test, y_test, cv=5)

print(scores)

print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))



print(mt.classification_report(y_test, xgb_predictions))

print(mt.confusion_matrix(y_test, xgb_predictions))