import os

import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

#import itertools
def show_surface(data):

    _,ax = plt.subplots(1,1, figsize=(30,10))

    sns.countplot(data['surface'], order = data['surface'].value_counts().index)

    for p in ax.patches:

        ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 10, '{:.2f}%'.format(100*p.get_height()/len(y_train)))     

    plt.show()    
def show_corr(data):

    plt.subplots(figsize=(10, 8))

    sns.heatmap(data.corr(), annot=True)
def get_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):

    R_earth = 6371

    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon])

    dlat = dropoff_lat - pickup_lat

    dlon = dropoff_lon - pickup_lon

    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2    

    return float("%.5f" % (2 * R_earth * np.arcsin(np.sqrt(a))))
def get_distance_record(x):

    if x.measurement_number == 0:

        return 0

    else:

        return get_distance(x.orientation_X, x.orientation_Y, x.orientation_X_old, x.orientation_Y_old)
def set_distance(data):

    df = data[['row_id', 'series_id', 'measurement_number', 'orientation_X', 'orientation_Y']].copy()

    df['row_id'] = df.apply(lambda x: str(x.series_id) + '_' + str(x.measurement_number + 1), axis = 1)    

    data['distance'] = data.merge(df, on='row_id', suffixes=('','_old')).apply(get_distance_record, axis=1)

    return data
def feature_eng(data):

    ignore_columns = ['row_id','series_id','measurement_number']

    df = pd.DataFrame()



    data['total_ang_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 + data['angular_velocity_Z']**2)** 0.5

    data['total_lin_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 + data['linear_acceleration_Z'])**0.5

    data['total_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z'])**0.5

   

    data['acc_vs_vel'] = data['total_lin_acc'] / data['total_ang_vel']

    

    for col in data.columns:

        if col in ignore_columns:

            continue

        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()

        df[col + '_median'] = data.groupby(['series_id'])[col].median()

        df[col + '_max'] = data.groupby(['series_id'])[col].max()

        df[col + '_min'] = data.groupby(['series_id'])[col].min()

        df[col + '_std'] = data.groupby(['series_id'])[col].std()

        df[col + '_range'] = df[col + '_max'] - df[col + '_min']

        df[col + '_maxToMin'] = df[col + '_max'] / df[col + '_min']

        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))

        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))

        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2

    return df
X_train = pd.read_csv("../input/competicao-dsa-machine-learning-sep-2019/X_treino.csv")

X_test = pd.read_csv("../input/competicao-dsa-machine-learning-sep-2019/X_teste.csv")

y_train = pd.read_csv("../input/competicao-dsa-machine-learning-sep-2019/y_treino.csv")

sub = pd.read_csv("../input/competicao-dsa-machine-learning-sep-2019/sample_submission.csv")
X_train.describe().T
X_train.head()
show_surface(y_train)
show_corr(X_train)
show_corr(X_test)
%%time

X_train = set_distance(X_train)

X_test = set_distance(X_test)
%%time

X_train = feature_eng(X_train)

X_test = feature_eng(X_test)

print(X_train.shape)
X_train.fillna(0, inplace = True)

X_test.fillna(0, inplace = True)

X_train.replace(-np.inf, 0, inplace = True)

X_train.replace(np.inf, 0, inplace = True)

X_test.replace(-np.inf, 0, inplace = True)

X_test.replace(np.inf, 0, inplace = True)
X_train.head()
X_train.describe().T
le = LabelEncoder()

y_train['surface'] = le.fit_transform(y_train['surface'])
def k_folds(X, y, X_test, k, n):

    score = 0

    y_test = np.zeros((X_test.shape[0], 9))

    folds = StratifiedKFold(n_splits = k, shuffle=True, random_state=2020)

    for i, (train_idx, val_idx) in  enumerate(folds.split(X, y)):

        clf =  RandomForestClassifier(n_estimators = n, n_jobs = 1)

        clf.fit(X_train.iloc[train_idx], y[train_idx])

        y_test += clf.predict_proba(X_test) / folds.n_splits

        score += clf.score(X.iloc[val_idx], y[val_idx])

        print(' Fold: {} | Score: {:.4f}%'.format(i+1, clf.score(X.iloc[val_idx], y[val_idx])))

    print('\n Accuracy: {:.4f}%'.format(score / folds.n_splits)) 

    return y_test 
y_test = k_folds(X_train, y_train['surface'], X_test, 200, 1000)
y_test = np.argmax(y_test, axis=1)

sub['surface'] = le.inverse_transform(y_test)

sub.to_csv('submission.csv', index=False)

sub.head(10)
show_surface(sub[['series_id', 'surface']])
y_train['surface'] = le.inverse_transform(y_train['surface'])

show_surface(y_train[['series_id', 'surface']])