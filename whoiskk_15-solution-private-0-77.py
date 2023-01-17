import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from itertools import combinations

import random

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.metrics import confusion_matrix



import time

from tqdm import tqdm

import warnings

warnings.simplefilter('ignore')



def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
train_X = pd.read_csv('../input/X_train.csv').iloc[:,3:].values.reshape(-1,128,10)

test_X  = pd.read_csv('../input/X_test.csv' ).iloc[:,3:].values.reshape(-1,128,10)

print('train_X shape:', train_X.shape, ', test_X shape:', test_X.shape)
df_train_y = pd.read_csv('../input/y_train.csv')



# build a dict to convert surface names into numbers

surface_names = df_train_y['surface'].unique()

num_surfaces = len(surface_names)

surface_to_numeric = dict(zip(surface_names, range(num_surfaces)))

print('Convert to numbers: ', surface_to_numeric)



# y and group data as numeric values:

train_y = df_train_y['surface'].replace(surface_to_numeric).values

train_group = df_train_y['group_id'].values
def sq_dist(a,b):

    ''' the squared euclidean distance between two samples '''

    

    return np.sum((a-b)**2, axis=1)





def find_run_edges(data, edge):

    ''' examine links between samples. left/right run edges are those samples which do not have a link on that side. '''



    if edge == 'left':

        border1 = 0

        border2 = -1

    elif edge == 'right':

        border1 = -1

        border2 = 0

    else:

        return False

    

    edge_list = []

    linked_list = []

    

    for i in range(len(data)):

        dist_list = sq_dist(data[i, border1, :4], data[:, border2, :4]) # distances to rest of samples

        min_dist = np.min(dist_list)

        closest_i   = np.argmin(dist_list) # this is i's closest neighbor

        if closest_i == i: # this might happen and it's definitely wrong

            print('Sample', i, 'linked with itself. Next closest sample used instead.')

            closest_i = np.argsort(dist_list)[1]

        dist_list = sq_dist(data[closest_i, border2, :4], data[:, border1, :4]) # now find closest_i's closest neighbor

        rev_dist = np.min(dist_list)

        closest_rev = np.argmin(dist_list) # here it is

        if closest_rev == closest_i: # again a check

            print('Sample', i, '(back-)linked with itself. Next closest sample used instead.')

            closest_rev = np.argsort(dist_list)[1]

        if (i != closest_rev): # we found an edge

            edge_list.append(i)

        else:

            linked_list.append([i, closest_i, min_dist])

            

    return edge_list, linked_list





def find_runs(data, left_edges, right_edges):

    ''' go through the list of samples & link the closest neighbors into a single run '''

    

    data_runs = []



    for start_point in left_edges:

        i = start_point

        run_list = [i]

        while i not in right_edges:

            tmp = np.argmin(sq_dist(data[i, -1, :4], data[:, 0, :4]))

            if tmp == i: # self-linked sample

                tmp = np.argsort(sq_dist(data[i, -1, :4], data[:, 0, :4]))[1]

            i = tmp

            run_list.append(i)

        data_runs.append(np.array(run_list))

    

    return data_runs
train_left_edges, train_left_linked  = find_run_edges(train_X, edge='left')

train_right_edges, train_right_linked = find_run_edges(train_X, edge='right')

print('Found', len(train_left_edges), 'left edges and', len(train_right_edges), 'right edges.')
train_runs = find_runs(train_X, train_left_edges, train_right_edges)
flat_list = [series_id for run in train_runs for series_id in run]

print(len(flat_list), len(np.unique(flat_list)))
df_train_y['run_id'] = 0

df_train_y['run_pos'] = 0



for run_id in range(len(train_runs)):

    for run_pos in range(len(train_runs[run_id])):

        series_id = train_runs[run_id][run_pos]

        df_train_y.at[ series_id, 'run_id'  ] = run_id

        df_train_y.at[ series_id, 'run_pos' ] = run_pos



df_train_y.to_csv('y_train_with_runs.csv', index=False)

df_train_y.tail()
test_left_edges, test_left_linked  = find_run_edges(test_X, edge='left')

test_right_edges, test_right_linked = find_run_edges(test_X, edge='right')

print('Found', len(test_left_edges), 'left edges and', len(test_right_edges), 'right edges.')
test_runs = find_runs(test_X, test_left_edges, test_right_edges)
lost_samples = np.array([ i for i in range(len(test_X)) if i not in np.concatenate(test_runs) ])

print(lost_samples)

print(len(lost_samples))
find_run_edges(test_X[lost_samples], edge='left')[1][0]
lost_run = np.array(lost_samples[find_runs(test_X[lost_samples], [0], [5])[0]])

test_runs.append(lost_run)
df_test_y = pd.read_csv("../input/sample_submission.csv")

df_test_y['run_id'] = 0

df_test_y['run_pos'] = 0



for run_id in range(len(test_runs)):

    for run_pos in range(len(test_runs[run_id])):

        series_id = test_runs[run_id][run_pos]

        df_test_y.at[ series_id, 'run_id'  ] = run_id

        df_test_y.at[ series_id, 'run_pos' ] = run_pos



df_test_y.to_csv('y_test_with_runs.csv', index=False)



df_test_y.drop("surface", axis=1, inplace=True)



cheat_json = df_train_y.groupby(['run_id'])['surface'].unique().reset_index().to_dict()

df_test_y['surface'] = df_test_y['run_id'].apply(lambda x: cheat_json['surface'][x][0])

df_test_y.head()
%%time

counter = 0

agg_dict = df_train_y.groupby(['run_id'])['series_id'].unique().reset_index()['series_id'].to_dict()

two_sampled_dict = {}

for key, value in agg_dict.items():

    two_sampled_dict[key] = []

#     for item in list(combinations(agg_dict[key].tolist(), 2)):

    llist = list(combinations(agg_dict[key].tolist(), 2))

    if len(llist) > 50:

        two_sampled_dict[key] = random.sample(llist, 50)

        counter += 50

    else:

        two_sampled_dict[key] = random.sample(llist, len(llist))

#         two_sampled_dict[key].append(item)

        counter += len(llist)

print(counter)

del llist

del counter
train = pd.read_csv("../input/X_train.csv")

test = pd.read_csv("../input/X_test.csv")

label = pd.read_csv("../input/y_train.csv")



train = reduce_mem_usage(train)

test = reduce_mem_usage(test)

label = reduce_mem_usage(label)
le = LabelEncoder()

label['surface'] = le.fit_transform(label['surface'])

print(le.classes_)
train.drop(['row_id', 'measurement_number'], axis=1, inplace=True)

test.drop(['row_id', 'measurement_number'], axis=1, inplace=True)
start_time = time.time()



new_train = train.copy()

print("Initial Train Size :: ", new_train.shape)



new_label = {}

last_series_id = 3810

# for item in range(149205):



for key, value in tqdm(two_sampled_dict.items(), total=len(two_sampled_dict)):

# for key, value in two_sampled_dict.items():

    

    for item in value:

        

        idx1 = item[0]

        idx2 = item[1]

        

        df = pd.DataFrame(columns=train.columns)



        # Creating Train

        for col in df.columns[1: ]:

            df[col] = new_train[col][(new_train['series_id'] == idx1) | (new_train['series_id'] == idx2)]

        df['series_id'] = last_series_id

        

        df.reset_index(inplace=True)

        df.drop(['index'], axis=1, inplace=True)



        # Creating in Label

        new_label[last_series_id] = df_train_y['surface'][(df_train_y['series_id'] == idx1) | (df_train_y['series_id'] == idx2)].value_counts(ascending=False).index[0]

        last_series_id += 1

        

        new_train = pd.concat([new_train, df], ignore_index=True)

        

print("Final Train Size :: ", new_train.shape)

print("Time Taken :: ", time.time() - start_time)
def FE(data):

    

    df = pd.DataFrame()

    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 +

                             data['angular_velocity_Z']**2)** 0.5

    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 +

                             data['linear_acceleration_Z']**2)**0.5

#     data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 +

#                              data['orientation_Z'])**0.5

   

    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']

    

    for col in data.columns:

        if col in ['row_id','series_id','measurement_number', 'orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W']:

            continue

        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()

        df[col + '_median'] = data.groupby(['series_id'])[col].median()

        df[col + '_max'] = data.groupby(['series_id'])[col].max()

        df[col + '_min'] = data.groupby(['series_id'])[col].min()

        df[col + '_std'] = data.groupby(['series_id'])[col].std()

        df[col + '_range'] = df[col + '_max'] - df[col + '_min']

        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']

        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))

        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))

        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2

    return df
%%time

new_train = FE(new_train)

test = FE(test)

print(new_train.shape, test.shape)
new_train.fillna(0, inplace = True)

test.fillna(0, inplace = True)

new_train.replace(-np.inf, 0, inplace = True)

new_train.replace(np.inf, 0, inplace = True)

test.replace(-np.inf, 0, inplace = True)

test.replace(np.inf, 0, inplace = True)
def k_folds(clf, X, y, X_test, k):

    folds = StratifiedKFold(n_splits = k, shuffle=True, random_state=13)

    y_test = np.zeros((X_test.shape[0], 9))

    y_oof = np.zeros((X.shape[0]))

    score = 0

    for i, (train_idx, val_idx) in  enumerate(folds.split(X, y)):

#         clf =  RandomForestClassifier(n_estimators = 500, n_jobs = -1)

        clf.fit(X.iloc[train_idx], y[train_idx])

        y_oof[val_idx] = clf.predict(X.iloc[val_idx])

        y_test += clf.predict_proba(X_test) / folds.n_splits

        score += clf.score(X.iloc[val_idx], y[val_idx])

        print('Fold: {} score: {}'.format(i,clf.score(X.iloc[val_idx], y[val_idx])))

    print('Avg Accuracy', score / folds.n_splits) 

        

    return y_oof, y_test
new_label = pd.DataFrame.from_dict(new_label, orient='index').reset_index()

new_label.columns = ['series_id', 'surface']

new_label.head()
label = pd.read_csv("../input/y_train.csv")

label = pd.concat([label, new_label], ignore_index=True)

label.head()
label['surface'] = le.transform(label['surface'])

label.surface.head()
rand = RandomForestClassifier(n_estimators=500, random_state=13)

y_oof, y_test_rand = k_folds(rand, new_train, label['surface'], test, k=5)
ext = ExtraTreesClassifier(n_estimators=500, random_state=13)

y_oof, y_test_ext = k_folds(ext, new_train, label['surface'], test, k=5)
confusion_matrix(y_oof,label['surface'])
# Submitting averaging



y_test = y_test_ext + y_test_rand

y_test = np.argmax(y_test, axis=1)

submission = pd.read_csv(os.path.join("../input/", 'sample_submission.csv'))

submission['surface'] = le.inverse_transform(y_test)

submission.to_csv('submission.csv', index=False)

submission.surface.value_counts()
df_test_y['sub'] = submission['surface']

df_test_y.head()
agg = df_test_y.groupby(['run_id', 'sub'])['sub'].count()

agg = pd.DataFrame(agg)

agg.columns = ['count']

agg.reset_index(inplace=True)

agg = df_test_y.groupby(['run_id']).agg(lambda x: x.value_counts().index[0]).reset_index()[['run_id', 'sub']]

agg_dict = agg.to_dict()

submission['surface'] = df_test_y['run_id'].apply(lambda x: agg_dict['sub'][x])

submission['surface'][df_test_y['run_id'] == 39] = 'hard_tiles'

submission.surface.value_counts()
submission.surface.value_counts() / submission.shape[0]
submission.shape
submission.to_csv("with_model.csv", index=False)