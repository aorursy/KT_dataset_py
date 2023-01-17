import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

application_test = pd.read_csv("../input/home-credit-default-risk/application_test.csv")

application_train = pd.read_csv("../input/home-credit-default-risk/application_train.csv")

bureau_balance = pd.read_csv("../input/home-credit-default-risk/bureau_balance.csv")

bureau = pd.read_csv("../input/home-credit-default-risk/bureau.csv")

credit_card_balance = pd.read_csv("../input/home-credit-default-risk/credit_card_balance.csv")

# HomeCredit_columns_description = pd.read_csv("../input/home-credit-default-risk/HomeCredit_columns_description.csv")

installments_payments = pd.read_csv("../input/home-credit-default-risk/installments_payments.csv")

POS_CASH_balance = pd.read_csv("../input/home-credit-default-risk/POS_CASH_balance.csv")

previous_application = pd.read_csv("../input/home-credit-default-risk/previous_application.csv")

sample_submission = pd.read_csv("../input/home-credit-default-risk/sample_submission.csv")
((application_train['TARGET'] == 0) & (application_train['CODE_GENDER'] == 'M')).sum()
height = np.array([((application_train['TARGET'] == 0) & (application_train['CODE_GENDER'] == 'M')).sum(),

                   ((application_train['TARGET'] == 1) & (application_train['CODE_GENDER'] == 'M')).sum(),

                   ((application_train['TARGET'] == 0) & (application_train['CODE_GENDER'] == 'F')).sum(),

                   ((application_train['TARGET'] == 1) & (application_train['CODE_GENDER'] == 'F')).sum()])

left = np.array([1, 2, 3, 4])

label = ['0 & M', '1 & M', '0 & F', '1 & F']



plt.bar(left, height, tick_label=label, align='center')

plt.show()

print(height[1]/(height[0]+height[1]))

print(height[3]/(height[2]+height[3]))
application_train.NAME_EDUCATION_TYPE.unique()
height = np.array([((application_train['TARGET'] == 0) & (application_train['NAME_EDUCATION_TYPE'] == 'Secondary / secondary special')).sum(),

                  ((application_train['TARGET'] == 1) & (application_train['NAME_EDUCATION_TYPE'] == 'Secondary / secondary special')).sum(),

                  ((application_train['TARGET'] == 0) & (application_train['NAME_EDUCATION_TYPE'] == 'Higher education')).sum(),

                  ((application_train['TARGET'] == 1) & (application_train['NAME_EDUCATION_TYPE'] == 'Higher education')).sum(),

                  ((application_train['TARGET'] == 0) & (application_train['NAME_EDUCATION_TYPE'] == 'Incomplete higher')).sum(),

                  ((application_train['TARGET'] == 1) & (application_train['NAME_EDUCATION_TYPE'] == 'Incomplete higher')).sum(),

                  ((application_train['TARGET'] == 0) & (application_train['NAME_EDUCATION_TYPE'] == 'Lower secondary')).sum(),

                  ((application_train['TARGET'] == 1) & (application_train['NAME_EDUCATION_TYPE'] == 'Lower secondary')).sum(),

                  ((application_train['TARGET'] == 0) & (application_train['NAME_EDUCATION_TYPE'] == 'Academic degree')).sum(),

                  ((application_train['TARGET'] == 1) & (application_train['NAME_EDUCATION_TYPE'] == 'Academic degree')).sum()])
left = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

plt.bar(left, height, align='center')

plt.show()

print(height[1]/(height[0]+height[1]))

print(height[3]/(height[2]+height[3]))

print(height[5]/(height[4]+height[5]))

print(height[7]/(height[6]+height[7]))

print(height[9]/(height[8]+height[9]))
application_train.NAME_INCOME_TYPE.unique()
height = np.array([((application_train['TARGET'] == 0) & (application_train['NAME_INCOME_TYPE'] == 'Working')).sum(),

                  ((application_train['TARGET'] == 1) & (application_train['NAME_INCOME_TYPE'] == 'Working')).sum(),

                  ((application_train['TARGET'] == 0) & (application_train['NAME_INCOME_TYPE'] == 'State servant')).sum(),

                  ((application_train['TARGET'] == 1) & (application_train['NAME_INCOME_TYPE'] == 'State servant')).sum(),

                  ((application_train['TARGET'] == 0) & (application_train['NAME_INCOME_TYPE'] == 'Commercial associate')).sum(),

                  ((application_train['TARGET'] == 1) & (application_train['NAME_INCOME_TYPE'] == 'Commercial associate')).sum(),

                  ((application_train['TARGET'] == 0) & (application_train['NAME_INCOME_TYPE'] == 'Pensioner')).sum(),

                  ((application_train['TARGET'] == 1) & (application_train['NAME_INCOME_TYPE'] == 'Pensioner')).sum(),

                  ((application_train['TARGET'] == 0) & (application_train['NAME_INCOME_TYPE'] == 'Unemployed')).sum(),

                  ((application_train['TARGET'] == 1) & (application_train['NAME_INCOME_TYPE'] == 'Unemployed')).sum(),

                  ((application_train['TARGET'] == 0) & (application_train['NAME_INCOME_TYPE'] == 'Student')).sum(),

                  ((application_train['TARGET'] == 1) & (application_train['NAME_INCOME_TYPE'] == 'Student')).sum(),

                  ((application_train['TARGET'] == 0) & (application_train['NAME_INCOME_TYPE'] == 'Businessman')).sum(),

                  ((application_train['TARGET'] == 1) & (application_train['NAME_INCOME_TYPE'] == 'Businessman')).sum(),

                  ((application_train['TARGET'] == 0) & (application_train['NAME_INCOME_TYPE'] == 'Maternity leave')).sum(),

                  ((application_train['TARGET'] == 1) & (application_train['NAME_INCOME_TYPE'] == 'Maternity leave')).sum()])
left = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

plt.bar(left, height, align='center')

plt.show()

print(height[1]/(height[0]+height[1]))

print(height[3]/(height[2]+height[3]))

print(height[5]/(height[4]+height[5]))

print(height[7]/(height[6]+height[7]))

print(height[9]/(height[8]+height[9]))

print(height[11]/(height[10]+height[12]))

print(height[13]/(height[12]+height[13]))

print(height[15]/(height[14]+height[15]))
height = np.array([((application_train['TARGET'] == 0) & (application_train['REGION_RATING_CLIENT_W_CITY'] == 1)).sum(),

                  ((application_train['TARGET'] == 1) & (application_train['REGION_RATING_CLIENT_W_CITY'] == 1)).sum(),

                  ((application_train['TARGET'] == 0) & (application_train['REGION_RATING_CLIENT_W_CITY'] == 2)).sum(),

                  ((application_train['TARGET'] == 1) & (application_train['REGION_RATING_CLIENT_W_CITY'] == 2)).sum(),

                  ((application_train['TARGET'] == 0) & (application_train['REGION_RATING_CLIENT_W_CITY'] == 3)).sum(),

                  ((application_train['TARGET'] == 1) & (application_train['REGION_RATING_CLIENT_W_CITY'] == 3)).sum()])
left = np.array([1, 2, 3, 4, 5, 6])

plt.bar(left, height, align='center')

plt.show()

print(height[1]/(height[0]+height[1]))

print(height[3]/(height[2]+height[3]))

print(height[5]/(height[4]+height[5]))
application_train['AGE'] = -(application_train.DAYS_BIRTH/365)



application_train[application_train["TARGET"] == 1].hist('AGE')
application_train['AMT_INCOME_TOTAL'].hist(by=application_train['TARGET'])
# 外れ値がすごいのでlogとる

# take the logarithm because the graph above is affected by outliers

np.log(application_train['AMT_INCOME_TOTAL']).hist(by=application_train['TARGET'])

plt.show()
# 上のやつでも、外れ値の影響あるので、rangeで表示範囲を狭める

np.log(application_train['AMT_INCOME_TOTAL']).hist(by=application_train['TARGET'], bins=20, range=(10, 14))

plt.show()
application_train['CNT_CHILDREN'].hist(by=application_train['TARGET'], bins=5, range=(-0.5, 4.5))
from sklearn.model_selection import train_test_split

import re

X = application_train.drop(columns=['TARGET', 'SK_ID_CURR'])

y = application_train['TARGET']

X = pd.get_dummies(X, drop_first=True, dummy_na=True)

X = X.fillna(X.mean())

rename_dict = {}

for i in X.columns[:]:

    rename_dict[i] = re.sub('[,:]', '_', i)

print(rename_dict)

X = X.rename(columns=rename_dict)

print(X.head())

X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=0.2, random_state=0)
import lightgbm as lgb

from sklearn.metrics import accuracy_score

gbm = lgb.LGBMClassifier(objective='binary',

                        num_leaves = 23,

                        learning_rate=0.1,

                        n_estimators=100)

gbm.fit(X_train, y_train,

        eval_set=[(X_test, y_test)],

        eval_metric='binary',

        early_stopping_rounds=10)

y_pred = gbm.predict(X_test)   # , num_iteration=gbm.best_iteration

y_pred_proba = gbm.predict_proba(X_test)  # , num_iteration=gbm.best_iteration



accu = accuracy_score(y_test, y_pred)

print('accuracy = {:>.4f}'.format(accu))



# Feature Importance

fti = gbm.feature_importances_



print('Feature Importances:')

for i, feat in enumerate(X_train.columns):

    print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))
columns_fti_list = []

for i, j in zip(X_train.columns, fti):

    columns_fti_list.append((i, j))

columns_fti_list.sort(key=lambda x: x[1], reverse=True)
num = 60

height = []

tick_label = []

for i, j in columns_fti_list[:num]:

    height.append(j)

    tick_label.append(i)

left = [x for x in range(num)]
fig = plt.figure(figsize=(36.0, 6.0))

ax = fig.add_axes([0.5,0.1,0.4,0.8])

plt.bar(left, height, tick_label=tick_label)

plt.xticks(rotation=90)

plt.show()
X = application_train.drop(columns=['TARGET', 'SK_ID_CURR'])

y = application_train['TARGET']

X = pd.get_dummies(X, drop_first=True, dummy_na=True)

# X = X.fillna(X.mean())

rename_dict = {}

for i in X.columns[:]:

    rename_dict[i] = re.sub('[,:]', '_', i)

X = X.rename(columns=rename_dict)

X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=0.2, random_state=0)



gbm = lgb.LGBMClassifier(objective='binary',

                        num_leaves = 23,

                        learning_rate=0.1,

                        n_estimators=100)

gbm.fit(X_train, y_train,

        eval_set=[(X_test, y_test)],

        eval_metric='binary',

        early_stopping_rounds=10)

y_pred = gbm.predict(X_test)   # , num_iteration=gbm.best_iteration

y_pred_proba = gbm.predict_proba(X_test)  # , num_iteration=gbm.best_iteration



accu = accuracy_score(y_test, y_pred)

print('accuracy = {:>.4f}'.format(accu))



# Feature Importance

fti = gbm.feature_importances_

"""

print('Feature Importances:')

for i, feat in enumerate(X_train.columns):

    print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))

"""    

columns_fti_list = []

for i, j in zip(X_train.columns, fti):

    columns_fti_list.append((i, j))

columns_fti_list.sort(key=lambda x: x[1], reverse=True)



num = 60

height = []

tick_label = []

for i, j in columns_fti_list[:num]:

    height.append(j)

    tick_label.append(i)

left = [x for x in range(num)]



fig = plt.figure(figsize=(36.0, 6.0))

ax = fig.add_axes([0.5,0.1,0.4,0.8])

plt.bar(left, height, tick_label=tick_label)

plt.xticks(rotation=90)

plt.show()
X = application_train.drop(columns=['TARGET', 'SK_ID_CURR'])

y = application_train['TARGET']

X = pd.get_dummies(X, drop_first=True, dummy_na=True)

# X = X.fillna(X.mean())

rename_dict = {}

for i in X.columns[:]:

    rename_dict[i] = re.sub('[,:]', '_', i)

X = X.rename(columns=rename_dict)

X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=0.2, random_state=0)



gbm = lgb.LGBMClassifier(objective='binary',

                        num_leaves = 23,

                        learning_rate=0.1,

                        n_estimators=100,

                        metric_types="auc")

gbm.fit(X_train, y_train,

        eval_set=[(X_test, y_test)],

        eval_metric='binary',

        early_stopping_rounds=10)

y_pred = gbm.predict(X_test)   # , num_iteration=gbm.best_iteration

y_pred_proba = gbm.predict_proba(X_test)  # , num_iteration=gbm.best_iteration



accu = accuracy_score(y_test, y_pred)

print('accuracy = {:>.4f}'.format(accu))



# Feature Importance

fti = gbm.feature_importances_

"""

print('Feature Importances:')

for i, feat in enumerate(X_train.columns):

    print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))

"""    

columns_fti_list = []

for i, j in zip(X_train.columns, fti):

    columns_fti_list.append((i, j))

columns_fti_list.sort(key=lambda x: x[1], reverse=True)



num = 60

height = []

tick_label = []

for i, j in columns_fti_list[:num]:

    height.append(j)

    tick_label.append(i)

left = [x for x in range(num)]



fig = plt.figure(figsize=(36.0, 6.0))

ax = fig.add_axes([0.5,0.1,0.4,0.8])

plt.bar(left, height, tick_label=tick_label)

plt.xticks(rotation=90)

plt.show()
from sklearn.metrics import roc_auc_score, roc_curve

print(roc_auc_score(y_test, y_pred_proba[:, 1]))

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

plt.plot(fpr, tpr, marker='o')

plt.xlabel('FPR: False positive rate')

plt.ylabel('TPR: True positive rate')

plt.grid()

plt.show()
X = application_train.drop(columns=['TARGET', 'SK_ID_CURR'])

y = application_train['TARGET']

X_submit = application_test.drop(columns='SK_ID_CURR')

lenX = len(X)

X = pd.concat([X, X_submit])

X = pd.get_dummies(X, drop_first=True, dummy_na=True)

X_submit = X[lenX:]

X = X[:lenX]

# X = X.fillna(X.mean())



rename_dict = {}

for i in X.columns[:]:

    rename_dict[i] = re.sub('[,:]', '_', i)

X = X.rename(columns=rename_dict)

X_submit = X_submit.rename(columns=rename_dict)



X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=0.2, random_state=0)



gbm = lgb.LGBMClassifier(objective='binary',

                        num_leaves = 23,

                        learning_rate=0.1,

                        n_estimators=100,

                        metric_types="auc")

gbm.fit(X_train, y_train,

        eval_set=[(X_test, y_test)],

        eval_metric='binary',

        early_stopping_rounds=10)

y_pred = gbm.predict(X_test)   # , num_iteration=gbm.best_iteration

y_pred_proba = gbm.predict_proba(X_test)  # , num_iteration=gbm.best_iteration



accu = accuracy_score(y_test, y_pred)

print('accuracy = {:>.4f}'.format(accu))



# Feature Importance

fti = gbm.feature_importances_

"""

print('Feature Importances:')

for i, feat in enumerate(X_train.columns):

    print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))

"""    

columns_fti_list = []

for i, j in zip(X_train.columns, fti):

    columns_fti_list.append((i, j))

columns_fti_list.sort(key=lambda x: x[1], reverse=True)



num = 60

height = []

tick_label = []

for i, j in columns_fti_list[:num]:

    height.append(j)

    tick_label.append(i)

left = [x for x in range(num)]



fig = plt.figure(figsize=(36.0, 6.0))

ax = fig.add_axes([0.5,0.1,0.4,0.8])

plt.bar(left, height, tick_label=tick_label)

plt.xticks(rotation=90)

plt.show()
len(gbm.predict_proba(X_submit))
gbm.predict_proba(X_submit)[:, 1]
baseline = gbm.predict_proba(X_submit)[:, 1]
baseline_submit = pd.concat([application_test['SK_ID_CURR'], pd.Series(baseline)], axis=1)
baseline_submit.columns = ['SK_ID_CURR', 'TARGET']
baseline_submit
baseline_submit.to_csv('baseline.csv', index=False)