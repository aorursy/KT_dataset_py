import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

ownership = pd.read_csv('../input/Building_Ownership_Use.csv')

structure = pd.read_csv('../input/Building_Structure.csv')
train.head()
test.head()
ownership.head()
structure.head()
print(train.shape,test.shape,ownership.shape,structure.shape)
train = pd.merge(train, ownership, on = ['building_id', 'district_id', 'vdcmun_id'])

train = pd.merge(train, structure, on = ['building_id', 'district_id', 'vdcmun_id'])
test = pd.merge(test, ownership, on = ['building_id', 'district_id', 'vdcmun_id'])

test = pd.merge(test, structure, on = ['building_id', 'district_id', 'vdcmun_id'])
train.head()
test.head()
train.dtypes
cat = []

for i in test.columns:

    if(test[i].dtypes == 'O'):

        cat.append(i)

cat.remove('building_id')
cat
fig, axs = plt.subplots(1,2, figsize = (15,5))

train['area_assesed'].value_counts().plot.bar(ax = axs[0])

test['area_assesed'].value_counts().plot.bar(ax = axs[1])
fig, axs = plt.subplots(1,2, figsize = (15,5))

train['legal_ownership_status'].value_counts().plot.bar(ax = axs[0])

test['legal_ownership_status'].value_counts().plot.bar(ax = axs[1])
fig, axs = plt.subplots(1,2, figsize = (15,5))

train['land_surface_condition'].value_counts().plot.bar(ax = axs[0])

test['land_surface_condition'].value_counts().plot.bar(ax = axs[1])
fig, axs = plt.subplots(1,2, figsize = (15,5))

train['foundation_type'].value_counts().plot.bar(ax = axs[0])

test['foundation_type'].value_counts().plot.bar(ax = axs[1])
fig, axs = plt.subplots(1,2, figsize = (15,5))

train['roof_type'].value_counts().plot.bar(ax = axs[0])

test['roof_type'].value_counts().plot.bar(ax = axs[1])
fig, axs = plt.subplots(1,2, figsize = (15,5))

train['ground_floor_type'].value_counts().plot.bar(ax = axs[0])

test['ground_floor_type'].value_counts().plot.bar(ax = axs[1])
fig, axs = plt.subplots(1,2, figsize = (15,5))

train['other_floor_type'].value_counts().plot.bar(ax = axs[0])

test['other_floor_type'].value_counts().plot.bar(ax = axs[1])
fig, axs = plt.subplots(1,2, figsize = (15,5))

train['plan_configuration'].value_counts().plot.bar(ax = axs[0])

test['plan_configuration'].value_counts().plot.bar(ax = axs[1])
fig, axs = plt.subplots(1,2, figsize = (15,5))

train['position'].value_counts().plot.bar(ax = axs[0])

test['position'].value_counts().plot.bar(ax = axs[1])
fig, axs = plt.subplots(1,2, figsize = (15,5))

train['condition_post_eq'].value_counts().plot.bar(ax = axs[0])

test['condition_post_eq'].value_counts().plot.bar(ax = axs[1])
train.isnull().sum()
test.isnull().sum()
train['has_repair_started'].value_counts()
train['has_repair_started'].replace(np.nan, 999, inplace = True)

test['has_repair_started'].replace(np.nan, 999, inplace = True)
# train['count_families'].value_counts()

train['count_families'].fillna(1.0, inplace = True)
remove = ['vdcmun_id', 'building_id', 'district_id', 'count_families']

train.drop(remove, axis = 1, inplace = True)

test.drop(remove, axis = 1, inplace = True)
print(train.shape,test.shape)
for i in cat:

    dummy = pd.get_dummies(train[i])

    train = pd.concat([train, dummy], axis = 1)

    train.drop(i, axis = 1, inplace = True)

    

    dummy = pd.get_dummies(test[i])

    test = pd.concat([test, dummy], axis = 1)

    test.drop(i, axis = 1, inplace = True)
train['damage_grade'].value_counts().plot.bar()
train['damage_grade'] = train['damage_grade'].apply(lambda x: int(x[-1]) - 1)
y = train['damage_grade']

train.drop(['damage_grade'], axis = 1, inplace = True)
print(train.shape,test.shape)

name = []

for i in range(90):

    name.append(str(i))

train.columns = name

test.columns = name
%%time

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

clf.fit(train, y)
pred = clf.predict(test)

pred
yp = []

for i in pred:

    yp.append('Grade ' + str(i+1))
tid = pd.read_csv('../input/test.csv')

my_submission = pd.DataFrame({'building_id': tid['building_id'], 'damage_grade': yp})

my_submission.to_csv('rf.csv', index=False)
%%time

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

clf = LGBMClassifier()

clf.fit(train, y)
pred = clf.predict(test)

yp = []

for i in pred:

    yp.append('Grade ' + str(i+1))

my_submission = pd.DataFrame({'building_id': tid['building_id'], 'damage_grade': yp})

my_submission.to_csv('lgb.csv', index=False)
%%time

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

clf = XGBClassifier()

clf.fit(train, y)
pred = clf.predict(test)

yp = []

for i in pred:

    yp.append('Grade ' + str(i+1))

my_submission = pd.DataFrame({'building_id': tid['building_id'], 'damage_grade': yp})

my_submission.to_csv('xgb.csv', index=False)