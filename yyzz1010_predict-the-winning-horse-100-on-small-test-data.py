import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd
runs = pd.read_csv("../input/hkracing/runs.csv")

runs.head()
races = pd.read_csv('../input/hkracing/races.csv')

races.head()
runs_data = runs[['race_id', 'won', 'horse_age', 'horse_country', 'horse_type', 'horse_rating',

       'horse_gear', 'declared_weight', 'actual_weight', 'draw', 'win_odds',

       'place_odds', 'horse_id']]

runs_data.head()
races_data = races[['race_id', 'venue', 'config', 'surface', 'distance', 'going', 'race_class', 'date']]

races_data.head()
# merge the two datasets based on race_id column

df = pd.merge(runs_data, races_data)

df.head()
df.isnull().any()
df.horse_country.isnull().value_counts(ascending=True)
df.horse_type.isnull().value_counts(ascending=True)
df.place_odds.isnull().value_counts(ascending=True)
df.shape
df = df.dropna()

df.shape
df.date = pd.to_datetime(df.date)

df.date.dtype
min(df.date), max(df.date)

# 8-year duration
start_time = min(df.date).strftime('%d %B %Y')

end_time = max(df.date).strftime('%d %B %Y')

no_of_horses = df.horse_id.nunique()

no_of_races = df.race_id.nunique()



print(f'The dataset was collected from {start_time} to {end_time}, which contains information about {no_of_horses} horses and {no_of_races} races. ')
# drop the unnecessary columns

df = df.drop(columns=['horse_id', 'date'])

df.head()
df.columns
df.horse_gear.value_counts(ascending=False)
df.horse_gear.nunique()
def horse_gear_impute(cols):

    if cols == '--':

        return 0

    else: 

        return 1
df.horse_gear = df.horse_gear.apply(horse_gear_impute)
df.horse_gear.value_counts()
df = pd.get_dummies(df, drop_first=True)

df.head()
df.columns
from time import time

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE

from sklearn.neighbors import KNeighborsClassifier

import lightgbm as lgb

from sklearn.metrics import precision_score, classification_report, confusion_matrix
last_raceid = max(df.race_id)

last_raceid
# split the last race data for deployment & save it in last_race variable

last_race = df[df.race_id == last_raceid]

last_race
new_data = df[:75696]   # drop the last race data for modeling

new_data = new_data.drop(columns='race_id')   # drop the unnecessary race_id column

new_data.tail()
new_data.shape
plt.figure(figsize=(6,4))

sns.countplot(data=new_data, x='won')

plt.title('Number of Labels by Class')
X = new_data.drop(columns='won')

y = new_data['won']
# extermely skewed data

y.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
k_range = range(1,10)

scores = {}

scores_list = []



for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    # precision ratio: tp / (tp + fp), aiming at minimize fp (predict: win, actual: lose)

    scores[k] = precision_score(y_test, y_pred)

    scores_list.append(precision_score(y_test, y_pred))
# find the highest precision score of the positive class (1)

import operator

max(scores.items(), key=operator.itemgetter(1))
plt.plot(k_range, scores_list)

plt.xlabel('Value of K for KNN')

plt.ylabel('Precision Score of the positive class (1)')

plt.title('Original Data')
start = time()



knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)



end = time()

running_time = end - start

print('time cost: %.5f sec' %running_time)
print(classification_report(y_test, y_pred))
labels = ['lose', 'win']

cm = confusion_matrix(y_test, y_pred)

print(cm)

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

plt.title('Confusion matrix')

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')
rus = RandomUnderSampler(random_state=0)

X_rus, y_rus = rus.fit_sample(X_train, y_train)



k_range = range(1,10)

scores = {}

scores_list = []



for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_rus, y_rus)

    y_pred = knn.predict(X_test)

    scores[k] = precision_score(y_test, y_pred)

    scores_list.append(precision_score(y_test, y_pred))
max(scores.items(), key=operator.itemgetter(1))
plt.plot(k_range, scores_list)

plt.xlabel('Value of K for KNN')

plt.ylabel('Precision Score of the positive class (1)')

plt.title('RUS Data')
start = time()



knn_rus = KNeighborsClassifier(n_neighbors=8)

knn_rus.fit(X_rus, y_rus)

y_pred = knn_rus.predict(X_test)



end = time()

running_time = end - start

print('time cost: %.5f sec' %running_time)
print(classification_report(y_test, y_pred))
labels = ['lose', 'win']

cm = confusion_matrix(y_test, y_pred)

print(cm)

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

plt.title('Confusion matrix')

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')
sm = SMOTE(random_state=0)

X_sm, y_sm = sm.fit_sample(X_train, y_train)



k_range = range(1,10)

scores = {}

scores_list = []



for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_sm, y_sm)

    y_pred = knn.predict(X_test)

    scores[k] = precision_score(y_test, y_pred)

    scores_list.append(precision_score(y_test, y_pred))
max(scores.items(), key=operator.itemgetter(1))
# SMOTE data

plt.plot(k_range, scores_list)

plt.xlabel('Value of K for KNN')

plt.ylabel('Precision Score of the positive class (1)')

plt.title('SMOTE Data')
start = time()



knn_sm = KNeighborsClassifier(n_neighbors=2)

knn_sm.fit(X_sm, y_sm)

y_pred = knn_sm.predict(X_test)



end = time()

running_time = end - start

print('time cost: %.5f sec' %running_time)
print(classification_report(y_test, y_pred))
labels = ['lose', 'win']

cm = confusion_matrix(y_test, y_pred)

print(cm)

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

plt.title('Confusion matrix')

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')
start = time()



d_train = lgb.Dataset(X_train, label = y_train)

params = {}

params['learning_rate'] = 0.003

params['boosting_type'] = 'gbdt'

params['objective'] = 'binary'

params['metric'] = 'binary_logloss'

params['sub_feature'] = 0.5

params['num_leaves'] = 100

params['min_data'] = 500

params['max_depth'] = 100

clf = lgb.train(params, d_train, 100)



end = time()

running_time = end - start

print('time cost: %.5f sec' %running_time)
#Prediction

y_pred = clf.predict(X_test)

#convert into binary values

for i in range(15140):

    if y_pred[i] >= 0.0995:       # setting threshold 

        y_pred[i] = 1

    else:  

        y_pred[i] = 0
print(classification_report(y_test, y_pred))
labels = ['lose', 'win']

cm = confusion_matrix(y_test, y_pred)

print(cm)

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

plt.title('Confusion matrix')

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')
# plot model’s feature importances (original data)

lgb.plot_importance(clf, max_num_features=10)
# convert array data into dataframe with column names, and feed into lgb model

X_rus = pd.DataFrame(X_rus, columns=list(X_train))

X_rus.head()
start = time()



d_train = lgb.Dataset(X_rus, label = y_rus)

params = {}

params['learning_rate'] = 0.003

params['boosting_type'] = 'gbdt'

params['objective'] = 'binary'

params['metric'] = 'binary_logloss'

params['sub_feature'] = 0.5

params['num_leaves'] = 100

params['min_data'] = 500

params['max_depth'] = 100

clf_rus = lgb.train(params, d_train, 100)



end = time()

running_time = end - start

print('time cost: %.5f sec' %running_time)
#Prediction

y_pred = clf_rus.predict(X_test)

#convert into binary values

for i in range(15140):

    if y_pred[i] >= 0.55:       # setting threshold 

        y_pred[i] = 1

    else:  

        y_pred[i] = 0
print(classification_report(y_test, y_pred))
labels = ['lose', 'win']

cm = confusion_matrix(y_test, y_pred)

print(cm)

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

plt.title('Confusion matrix')

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')
# plot model’s feature importances (Random Under-sampling)

lgb.plot_importance(clf_rus, max_num_features=10)
# convert array data into dataframe with column names, and feed into lgb model

X_sm = pd.DataFrame(X_sm, columns=list(X_train))

X_sm.head()
start = time()



d_train = lgb.Dataset(X_sm, label = y_sm)

params = {}

params['learning_rate'] = 0.003

params['boosting_type'] = 'gbdt'

params['objective'] = 'binary'

params['metric'] = 'binary_logloss'

params['sub_feature'] = 0.5

params['num_leaves'] = 100

params['min_data'] = 500

params['max_depth'] = 100

clf_sm = lgb.train(params, d_train, 100)



end = time()

running_time = end - start

print('time cost: %.5f sec' %running_time)
#Prediction

y_pred = clf_sm.predict(X_test)

#convert into binary values

for i in range(15140):

    if y_pred[i] >= 0.5:       # setting threshold 

        y_pred[i] = 1

    else:  

        y_pred[i] = 0
print(classification_report(y_test, y_pred))
labels = ['lose', 'win']

cm = confusion_matrix(y_test, y_pred)

print(cm)

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

plt.title('Confusion matrix')

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')
# plot model’s feature importances (SMOTE)

lgb.plot_importance(clf_sm, max_num_features=10)
# data that never been seen by the models

last_race
# drop unnecessary columns & define data and labels

X_deploy = last_race.drop(columns=['race_id', 'won'])

y_deploy = last_race.won
predictions = knn.predict(X_deploy)

print(classification_report(y_deploy, predictions))
predictions = knn_rus.predict(X_deploy)

print(classification_report(y_deploy, predictions))
import numpy as np



data = confusion_matrix(y_deploy, predictions)



fig, ax = plt.subplots()

cax = ax.matshow(data, cmap='RdBu')



for (i, j), z in np.ndenumerate(data):

    ax.text(j, i, '{}'.format(z), ha='center', va='center',

            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    

plt.title('Confusion matrix of kNN_rus', y=1.1)

fig.colorbar(cax)

labels = ['lose', 'win']

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Prediction')

plt.ylabel('Actual')
predictions = knn_sm.predict(X_deploy)

print(classification_report(y_deploy, predictions))
predictions = clf.predict(X_deploy)

#convert into binary values

for i in range(14):

    if predictions[i] >= 0.0995:       # setting threshold 

        predictions[i] = 1

    else:  

        predictions[i] = 0
predictions_rus = clf_rus.predict(X_deploy)

#convert into binary values

for i in range(14):

    if predictions_rus[i] >= 0.55:       # setting threshold 

        predictions_rus[i] = 1

    else:  

        predictions_rus[i] = 0
predictions_sm = clf_sm.predict(X_deploy)

#convert into binary values

for i in range(14):

    if predictions_sm[i] >= 0.5:       # setting threshold 

        predictions_sm[i] = 1

    else:  

        predictions_sm[i] = 0
print(classification_report(y_deploy, predictions))
print(classification_report(y_deploy, predictions_rus))
print(classification_report(y_deploy, predictions_sm))
data = confusion_matrix(y_deploy, predictions)



fig, ax = plt.subplots()

cax = ax.matshow(data, cmap='RdBu')



for (i, j), z in np.ndenumerate(data):

    ax.text(j, i, '{}'.format(z), ha='center', va='center',

            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    

plt.title('Confusion matrix of LightGBM models', y=1.1)

fig.colorbar(cax)

labels = ['lose', 'win']

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Prediction')

plt.ylabel('Actual')