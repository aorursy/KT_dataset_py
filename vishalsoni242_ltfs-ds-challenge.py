!wget train.zip https://datahack-prod.s3.amazonaws.com/train_file/train_aox2Jxw.zip
!wget test.csv https://datahack-prod.s3.amazonaws.com/test_file/test_bqCt9Pv.csv
!wget sample.csv https://datahack-prod.s3.amazonaws.com/sample_submission/sample_submission_24jSKY6.csv
!unzip train_aox2Jxw.zip
!ls
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train = pd.read_csv('train.csv')

test = pd.read_csv('test_bqCt9Pv.csv')

sample = pd.read_csv('sample_submission_24jSKY6.csv')

data_dict = pd.read_excel('Data Dictionary.xlsx')
data_dict
train.head()
test.head()
train.shape, test.shape
train.isnull().sum()
test.isnull().sum()
train['Employment.Type'].value_counts()
train['Employment.Type'].replace(np.nan, 'Not Provided', inplace=True)

test['Employment.Type'].replace(np.nan, 'Not Provided', inplace=True)
train.head()
%%time

train['Date.of.Birth'] = pd.to_datetime(train['Date.of.Birth'])

train['DisbursalDate'] = pd.to_datetime(train['DisbursalDate'])

train['age'] = (train['DisbursalDate'] - train['Date.of.Birth']).dt.days

train['age'] /= 365



test['Date.of.Birth'] = pd.to_datetime(test['Date.of.Birth'])

test['DisbursalDate'] = pd.to_datetime(test['DisbursalDate'])

test['age'] = (test['DisbursalDate'] - test['Date.of.Birth']).dt.days

test['age'] /= 365
s = '10yrs 10mon'

s[0:s.find('yrs')], s[s.find(' ')+1:s.find('mon')]
%%time

train['AVERAGE.ACCT.AGE'] = train['AVERAGE.ACCT.AGE'].apply(lambda s: int(s[0 : s.find('yrs')]) + int(s[s.find(' ')+1:s.find('mon')])/12)

test['AVERAGE.ACCT.AGE'] = test['AVERAGE.ACCT.AGE'].apply(lambda s: int(s[0 : s.find('yrs')]) + int(s[s.find(' ')+1:s.find('mon')])/12)
test.head()
%%time

train['CREDIT.HISTORY.LENGTH'] = train['CREDIT.HISTORY.LENGTH'].apply(lambda s: int(s[0 : s.find('yrs')]) + int(s[s.find(' ')+1:s.find('mon')])/12)

test['CREDIT.HISTORY.LENGTH'] = test['CREDIT.HISTORY.LENGTH'].apply(lambda s: int(s[0 : s.find('yrs')]) + int(s[s.find(' ')+1:s.find('mon')])/12)
train.head()
uid = test['UniqueID']

cols = ['UniqueID', 'Date.of.Birth', 'DisbursalDate']

train.drop(cols, axis=1, inplace=True)

test.drop(cols, axis=1, inplace=True)
fig, axs = plt.subplots(1,2, figsize = (15,5))

train['Employment.Type'].value_counts().plot.bar(ax = axs[0])

test['Employment.Type'].value_counts().plot.bar(ax = axs[1])
fig, axs = plt.subplots(1,2, figsize = (15,5))

train['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts().plot.bar(ax = axs[0])

test['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts().plot.bar(ax = axs[1])
train['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()
test['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()
train.shape
train = train[(train['PERFORM_CNS.SCORE.DESCRIPTION'] != 'Not Scored: More than 50 active Accounts found')]
train.head()
test.head()
# from sklearn.preprocessing import LabelEncoder

# cat = ['PERFORM_CNS.SCORE.DESCRIPTION', 'Employment.Type']

# for i in cat:

#     lb = LabelEncoder()

#     lb.fit(train[i])

#     train[i] = lb.transform(train[i])

#     test[i] = lb.transform(test[i])    
cat = ['PERFORM_CNS.SCORE.DESCRIPTION', 'Employment.Type']

for i in cat:

    dummy = pd.get_dummies(train[i])

    train = pd.concat([train, dummy], axis = 1)

    train.drop(i, axis = 1, inplace = True)

    

    dummy = pd.get_dummies(test[i])

    test = pd.concat([test, dummy], axis = 1)

    test.drop(i, axis = 1, inplace = True)
train.corr()
corr_matrix = train.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]

print(to_drop)
train.drop(to_drop, axis=1, inplace=True)

test.drop(to_drop, axis=1, inplace=True)
train['loan_default'].value_counts()
train.head()
cols = train.columns
from sklearn.utils import resample

tr1 = train[train['loan_default'] == 0]

tr2 = train[train['loan_default'] != 0]

print(tr1.shape,tr2.shape,train.shape)

tr1 = resample(tr1, replace = False, n_samples = 130000, random_state = 51)

train_downsample = pd.concat([tr1, tr2])
from sklearn.utils import resample

tr1 = train[train['loan_default'] == 1]

tr2 = train[train['loan_default'] != 1]

print(tr1.shape,tr2.shape,train.shape)

tr1 = resample(tr1, replace = True, n_samples = 100000, random_state = 51)

train_upsample = pd.concat([tr1, tr2])
train_downsample['loan_default'].value_counts()
train_upsample['loan_default'].value_counts()
y = train['loan_default']

train.drop(['loan_default'], axis=1, inplace=True)
%%time

cols = train.columns

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=51)

train_smote, y_smote = sm.fit_resample(train, y)
corr_matrix = train.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]

print(to_drop)



train.drop(to_drop, axis=1, inplace=True)

test.drop(to_drop, axis=1, inplace=True)
corr_matrix = train_smote.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]

print(to_drop)



train_smote.drop(to_drop, axis=1, inplace=True)

# test_smote.drop(to_drop, axis=1, inplace=True)
corr_matrix = train_upsample.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]

print(to_drop)



train_upsample.drop(to_drop, axis=1, inplace=True)

# test_upsample.drop(to_drop, axis=1, inplace=True)
corr_matrix = train_downsample.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]

print(to_drop)



train_downsample.drop(to_drop, axis=1, inplace=True)

# test_downsample.drop(to_drop, axis=1, inplace=True)
y_downsample = train_downsample['loan_default']

train_downsample.drop(['loan_default'], axis=1, inplace=True)



y_upsample = train_upsample['loan_default']

train_upsample.drop(['loan_default'], axis=1, inplace=True)
sample.head()
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)

print(class_weights)

class_weight_dict = dict(enumerate(class_weights))

print(class_weight_dict)
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# scaler = MinMaxScaler()

# train = scaler.fit_transform(train)

# test = scaler.transform(test)



# scaler = MinMaxScaler()

# train_smote = scaler.fit_transform(train_smote)

# test_smote = scaler.transform(test_smote)



# scaler = MinMaxScaler()

# train_downsample = scaler.fit_transform(train_downsample)

# test_downsample = scaler.transform(test_downsample)



# scaler = MinMaxScaler()

# train_upsample = scaler.fit_transform(train_upsample)

# test_upsample = scaler.transform(test_upsample)
%%time

from lightgbm import LGBMClassifier

clf1 = LGBMClassifier(random_state=25)

clf1.fit(train, y)



# clf2 = LGBMClassifier(random_state=25, class_weight={0: 0.6386298893393229, 1: 1.5})

# clf2.fit(train, y)



# clf3 = LGBMClassifier(random_state=25)

# clf3.fit(train_smote, y_smote)



# clf4 = LGBMClassifier( random_state=25)

# clf4.fit(train_downsample, y_downsample)



# clf5 = LGBMClassifier( random_state=25)

# clf5.fit(train_upsample, y_upsample)
yp1 = clf1.predict_proba(test)[:, 1]

# yp2 = clf2.predict_proba(test)[:, 1]

# yp3 = clf3.predict_proba(test)[:, 1]

# yp4 = clf4.predict_proba(test)[:, 1]

# yp5 = clf5.predict_proba(test)[:, 1]



# yp = (yp1 + yp2 + yp3 + yp4 + yp5)/5



sub = pd.DataFrame({'UniqueID' : uid, 'loan_default' : yp1})

sub.to_csv('LGB.csv', index = False)

cnt = 0

for i in yp1:

    if(i >= 0.5):

        cnt += 1

print(cnt)
import seaborn as sns

feature_imp = pd.DataFrame(sorted(zip(clf1.feature_importances_, cols)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.show()
# %%time

# from xgboost import XGBClassifier

# clf = XGBClassifier(n_estimators = 500)

# clf.fit(train, y)
# # yp = clf.predict_proba(test)[:, 1]

# # sub = pd.DataFrame({'UniqueID' : uid, 'loan_default' : yp})

# # sub.to_csv('XGB.csv', index = False)

# cnt = 0

# for i in yp:

#     if(i >= 0.5):

#         cnt += 1

# print(cnt)
%%time

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold



params = {

    'n_estimators': [100, 250, 400, 75],

    'max_depth': [7, 9, 11, 13],

    'learning_rate': [0.1, 0.01, 0.05],

#     'num_leaves': [4000, 8000]

#     'min_child_weight': [1, 3, 5],

#     'subsample': [0.6, 0.8, 1.0],

#     'colsample_bytree': [0.6, 0.8, 1.0],

}



clf1 = LGBMClassifier(random_state=51, class_weight={0: 0.6386298893393229, 1: 1.5})

# clf1.fit(train, y)



clf2 = LGBMClassifier(random_state=51)

# clf2.fit(train, y)



clf3 = LGBMClassifier(random_state=51)

# clf3.fit(train_smote, y_smote)



clf4 = LGBMClassifier( random_state=51)

# clf4.fit(train_downsample, y_downsample)



clf5 = LGBMClassifier( random_state=51)

# clf5.fit(train_upsample, y_upsample)



# clf = LGBMClassifier(random_state=25)

skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = 51)

print("--------------------       clf1          ---------------------------")

grid1 = GridSearchCV(estimator=clf1, param_grid=params, scoring='roc_auc', cv=skf.split(train, y), verbose=3 )

grid1.fit(train, y)

print('\n All results:')

print(grid1.cv_results_)

print('\n Best estimator:')

print(grid1.best_estimator_)

print('\n Best score:')

print(grid1.best_score_ * 2 - 1)

print('\n Best parameters:')

print(grid1.best_params_)



print("--------------------       clf2          ---------------------------")

grid2 = GridSearchCV(estimator=clf2, param_grid=params, scoring='roc_auc', cv=skf.split(train, y), verbose=3 )

grid2.fit(train, y)

print('\n All results:')

print(grid2.cv_results_)

print('\n Best estimator:')

print(grid2.best_estimator_)

print('\n Best score:')

print(grid2.best_score_ * 2 - 1)

print('\n Best parameters:')

print(grid2.best_params_)



print("--------------------       clf3          ---------------------------")

grid3 = GridSearchCV(estimator=clf3, param_grid=params, scoring='roc_auc', cv=skf.split(train_smote, y_smote), verbose=3 )

grid3.fit(train_smote, y_smote)

print('\n All results:')

print(grid3.cv_results_)

print('\n Best estimator:')

print(grid3.best_estimator_)

print('\n Best score:')

print(grid3.best_score_ * 2 - 1)

print('\n Best parameters:')

print(grid3.best_params_)



print("--------------------       clf5          ---------------------------")

grid5 = GridSearchCV(estimator=clf5, param_grid=params, scoring='roc_auc', cv=skf.split(train_upsample, y_upsample), verbose=3 )

grid5.fit(train_upsample, y_upsample)

print('\n All results:')

print(grid5.cv_results_)

print('\n Best estimator:')

print(grid5.best_estimator_)

print('\n Best score:')

print(grid5.best_score_ * 2 - 1)

print('\n Best parameters:')

print(grid5.best_params_)





print("--------------------       clf4          ---------------------------")

grid4 = GridSearchCV(estimator=clf4, param_grid=params, scoring='roc_auc', cv=skf.split(train_downsample, y_downsample), verbose=3 )

grid4.fit(train_downsample, y_downsample)

print('\n All results:')

print(grid4.cv_results_)

print('\n Best estimator:')

print(grid4.best_estimator_)

print('\n Best score:')

print(grid4.best_score_ * 2 - 1)

print('\n Best parameters:')

print(grid4.best_params_)
%%time











yp1 = grid1.best_estimator_.predict_proba(test)[:, 1]

yp2 = grid2.best_estimator_.predict_proba(test)[:, 1]

yp3 = grid3.best_estimator_.predict_proba(test)[:, 1]

yp4 = grid4.best_estimator_.predict_proba(test)[:, 1]

yp5 = grid5.best_estimator_.predict_proba(test)[:, 1]





yp = (yp1 + yp2 + yp3 + yp4 + yp5)/5



sub = pd.DataFrame({'UniqueID' : uid, 'loan_default' : yp})

sub.to_csv('LGB_grid1.csv', index = False)

cnt = 0

for i in yp:

    if(i >= 0.5):

        cnt += 1

print(cnt)