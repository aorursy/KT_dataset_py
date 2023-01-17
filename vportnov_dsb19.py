import pandas as pd

import random

import matplotlib.pyplot as plt

import numpy as np

import json



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report



from sklearn import datasets

from sklearn.utils import shuffle
# random sample

#filename = "../input/data-science-bowl-2019/train.csv"

#n = sum(1 for line in open(filename)) - 1

#s = 1000000 #desired sample size



#skip = sorted(random.sample(range(1,n+1),n-s))



#train_data = pd.read_csv(filename)

train_data = pd.read_csv("../input/data-science-bowl-2019/train.csv")

specs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")

test_data = pd.read_csv("../input/data-science-bowl-2019/test.csv")

train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")
train_data.shape
test_data.shape
tmp_labels = train_labels.drop_duplicates(subset="installation_id", keep="last")



tmp_labels.head(20)
acc_groups = tmp_labels["accuracy_group"]

acc_groups.head()
specs.shape
users = train_data['installation_id'].drop_duplicates()

print('unique users: {}'.format(users.size))

attempted_users = train_data[train_data['type']=='Assessment'][['installation_id']].drop_duplicates() 

print('users, who attempted assessments: {}'.format(attempted_users.size))

train_data = pd.merge(train_data, attempted_users, on="installation_id", how="inner")
names = []

values = []

type_count = train_data.groupby('type').count()

for t in train_data['type'].drop_duplicates():

    names.append(t)

    values.append(len(train_data[train_data.type == t]))



fig = plt.figure(figsize=(8, 5))

plt.bar(names, values)

plt.title('Number of events by type')

plt.show()
names = []

values = []

for t in train_data['title'].drop_duplicates():

    names.append(t)

    values.append(len(train_data[train_data.title == t]))



fig = plt.figure(figsize=(13, 15))

plt.barh(names, values)

plt.title('Number of events by title')

plt.show()
train_data.world.drop_duplicates()
print('MAGMAPEAK - {}\n'.format(pd.unique(train_data[(train_data.world == 'MAGMAPEAK') & (train_data.type == 'Assessment')].title)))

print('CRYSTALCAVES - {}\n'.format(pd.unique(train_data[(train_data.world == 'CRYSTALCAVES') & (train_data.type == 'Assessment')].title)))

print('TREETOPCITY - {}\n'.format(pd.unique(train_data[(train_data.world == 'TREETOPCITY') & (train_data.type == 'Assessment')].title)))
names = []

values = []

type_count = train_data.groupby('world').count()

for t in train_data['world'].drop_duplicates():

    names.append(t)

    values.append(len(train_data[train_data.world == t]))



fig = plt.figure(figsize=(8, 5))

plt.bar(names, values)

plt.title('Number of events by world')

plt.show()
train_data[train_data.event_code == 4100].title.drop_duplicates()
train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])

train_data['weekday'] = train_data['timestamp'].dt.dayofweek

train_data['hour'] = train_data['timestamp'].dt.hour
fig = plt.figure(figsize=(12, 8))

names = ['Mon', 'Tue', 'Wd', 'Thu', 'Fri', 'Sat', 'Sun']

values = []

for d in range(7):

    values.append(len(train_data[train_data.weekday == d]))

plt.bar(names, values)

plt.title('Event count by weekday')

plt.show()
fig = plt.figure(figsize=(14, 9))

names = range(24)

values = []

for h in range(24):

    values.append(len(train_data[train_data.hour == h]))

plt.bar(names, values, width=0.5)

plt.title('Event count by hour')

plt.xticks(range(24))

plt.show()
# fig = plt.figure(figsize=())

time_by_session = train_data[['game_session', 'world', 'game_time']].groupby(['game_session', 'world']).max()
def calc_playtime():

    result = list()

    for u in attempted_users['installation_id']:

        time_by_world = {'MAGMAPEAK':0,'TREETOPCITY':0,'CRYSTALCAVES':0,'NONE':0}

        sessions_by_user = train_data[train_data.installation_id == u]['game_session'].drop_duplicates()

        for s in sessions_by_user:

            tmp = time_by_session.loc[s]['game_time'].iloc[0]

            time_by_world[time_by_session.loc[s].index.tolist()[0]] += tmp

        result.append(time_by_world)

    return result



playtime = calc_playtime()
fig = plt.figure(figsize=(15,8))

plt.plot([u['MAGMAPEAK'] + u['TREETOPCITY'] + u['CRYSTALCAVES'] for u in playtime])

plt.title('Users playtime')

plt.show()
fig = plt.figure(figsize=(8, 5))

val = [0, 0, 0]

for u in playtime:

    val[0] += u['MAGMAPEAK']

    val[1] += u['TREETOPCITY']

    val[2] += u['CRYSTALCAVES']

plt.bar(['MAGMAPEAK', 'TREETOPCITY', 'CRYSTALCAVES'], val)

plt.show()
train_labels.head(9)
train_labels[['installation_id', 'accuracy_group']].groupby(['accuracy_group']).count().plot.bar(figsize=(10, 6))

plt.show()
tasks = pd.unique(train_labels.title)

mean_wrong = []

for t in tasks:

    mean_wrong.append(train_labels[train_labels.title == t].num_incorrect.mean())

fig = plt.figure(figsize=(7, 7))

plt.pie(mean_wrong, labels=tasks)

plt.show()
def calc_labels(data):

    temp_data = data.query('event_code == [4100,4110] and type == "Assessment"').copy()

    sessions = temp_data[temp_data.type == 'Assessment'][['game_session']].drop_duplicates()

    labels = ['game_session', 'installation_id', 'title', 'num_correct', 'num_incorrect', 'accuracy', 'accuracy_group']

    result = pd.DataFrame(columns=labels)    

    for s in sessions['game_session']:

        tmp = pd.DataFrame(columns=labels)

        events_by_session = temp_data[temp_data.game_session == s]['event_data']

        num_correct = 0

        num_incorrect = 0

        for e in events_by_session:

            if json.loads(e)['correct']:

                num_correct += 1

            else:

                num_incorrect += 1

        if num_correct < 1:

            accuracy = 0.0

        else:

            accuracy = num_correct / (num_correct + num_incorrect)

        if num_incorrect == 0 and num_correct > 0:

            accuracy_group = 3

        elif num_incorrect == 1 and num_correct > 0:

            accuracy_group = 2

        elif num_incorrect >= 2 and num_correct > 0:

            accuracy_group = 1

        else:

            accuracy_group = 0            

        tmp['game_session'] = pd.Series(s)

        tmp['installation_id'] = temp_data.loc[temp_data.game_session == s].iloc[0]['installation_id']

        tmp['title'] = temp_data.loc[temp_data.game_session == s].iloc[0]['title']

        tmp['num_correct'] = num_correct

        tmp['num_incorrect'] = num_incorrect

        tmp['accuracy'] = accuracy

        tmp['accuracy_group'] = accuracy_group        

        result = result.append(tmp, ignore_index=True)

    return result
train_labels = calc_labels(train_data)
test_labels = calc_labels(test_data)
def create_features(data, data_labels):

    global attempted_users, playtime

    labels = ['id', 'activities', 'games', 'clips', 'assessments', 'mean_activity_daytime', 'mean_game_daytime', 'mean_clip_daytime', 'mean_assessment_daytime', 'MAGMAPEAK', 'TREETOPCITY', 'CRYSTALCAVES', 'accuracy']

    result = pd.DataFrame(columns=labels)

    

    for i, u in enumerate(attempted_users.installation_id):

        tmp = pd.DataFrame(columns=labels)

        cur_user = data[data.installation_id == u]

        

        tmp['id'] = pd.Series(u)

        sub = cur_user[cur_user.type == 'Activity']

        tmp['activities'] = pd.Series(len(sub))

        m = sub[['installation_id', 'hour']].groupby(['installation_id']).mean().values

        if len(m) > 0:

            tmp['mean_activity_daytime'] = pd.Series(m[0][0])

        else:

            tmp['mean_activity_daytime'] = pd.Series(0.)

        

        sub = cur_user[cur_user.type == 'Game']

        m = sub[['installation_id', 'hour']].groupby(['installation_id']).mean().values

        tmp['games'] = pd.Series(len(sub))

        if len(m) > 0:

            tmp['mean_game_daytime'] = pd.Series(m[0][0])

        else:

            tmp['mean_game_daytime'] = pd.Series(0.)

            

        sub = cur_user[cur_user.type == 'Clip']

        m = sub[['installation_id', 'hour']].groupby(['installation_id']).mean().values

        tmp['clips'] = pd.Series(len(sub))

        if len(m) > 0:

            tmp['mean_clip_daytime'] = pd.Series(m[0][0])

        else:

            tmp['mean_clip_daytime'] = pd.Series(0.)

            

        sub = cur_user[cur_user.type == 'Assessment']

        m = sub[['installation_id', 'hour']].groupby(['installation_id']).mean().values

        tmp['assessments'] = pd.Series(len(sub))

        if len(m) > 0:

            tmp['mean_assessment_daytime'] = pd.Series(m[0][0]) 

        else:

            tmp['mean_assessment_daytime'] = pd.Series(0.)

        

        tmp['MAGMAPEAK'] = pd.Series(playtime[i]['MAGMAPEAK'])

        tmp['TREETOPCITY'] = pd.Series(playtime[i]['TREETOPCITY'])

        tmp['CRYSTALCAVES'] = pd.Series(playtime[i]['CRYSTALCAVES'])

        

        acc_mode = data_labels[data_labels.installation_id == u]['accuracy_group'].mode()

        if acc_mode.dropna().empty:

            tmp['accuracy'] = pd.Series(0)

        else:

            tmp['accuracy'] = pd.Series(acc_mode.max())

            

        result = result.append(tmp, ignore_index=True)        

    return result
features = create_features(train_data, train_labels)
features.head(10)
test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])

test_data['weekday'] = test_data['timestamp'].dt.dayofweek

test_data['hour'] = test_data['timestamp'].dt.hour
def extract_features(data, data_labels):

    all_users = data[data['type']=='Assessment'][['installation_id']].drop_duplicates()

    time_by_session = data[['game_session', 'world', 'game_time']].groupby(['game_session', 'world']).max()

    ext_playtime = []

    for u in all_users['installation_id']:

        time_by_world = {'MAGMAPEAK':0,'TREETOPCITY':0,'CRYSTALCAVES':0,'NONE':0}

        sessions_by_user = data[data.installation_id == u]['game_session'].drop_duplicates()

        for s in sessions_by_user:

            tmp = time_by_session.loc[s]['game_time'].iloc[0]

            time_by_world[time_by_session.loc[s].index.tolist()[0]] += tmp

        ext_playtime.append(time_by_world)

    labels = ['id', 'activities', 'games', 'clips', 'assessments', 'mean_activity_daytime', 'mean_game_daytime', 'mean_clip_daytime', 'mean_assessment_daytime', 'MAGMAPEAK', 'TREETOPCITY', 'CRYSTALCAVES', 'accuracy']

    result = pd.DataFrame(columns=labels)

    

    for i, u in enumerate(all_users.installation_id):

        tmp = pd.DataFrame(columns=labels)

        cur_user = data[data.installation_id == u]

        

        tmp['id'] = pd.Series(u)

        sub = cur_user[cur_user.type == 'Activity']

        tmp['activities'] = pd.Series(len(sub))

        m = sub[['installation_id', 'hour']].groupby(['installation_id']).mean().values

        if len(m) > 0:

            tmp['mean_activity_daytime'] = pd.Series(m[0][0])

        else:

            tmp['mean_activity_daytime'] = pd.Series(0.)

        

        sub = cur_user[cur_user.type == 'Game']

        m = sub[['installation_id', 'hour']].groupby(['installation_id']).mean().values

        tmp['games'] = pd.Series(len(sub))

        if len(m) > 0:

            tmp['mean_game_daytime'] = pd.Series(m[0][0])

        else:

            tmp['mean_game_daytime'] = pd.Series(0.)

            

        sub = cur_user[cur_user.type == 'Clip']

        m = sub[['installation_id', 'hour']].groupby(['installation_id']).mean().values

        tmp['clips'] = pd.Series(len(sub))

        if len(m) > 0:

            tmp['mean_clip_daytime'] = pd.Series(m[0][0])

        else:

            tmp['mean_clip_daytime'] = pd.Series(0.)

            

        sub = cur_user[cur_user.type == 'Assessment']

        m = sub[['installation_id', 'hour']].groupby(['installation_id']).mean().values

        tmp['assessments'] = pd.Series(len(sub))

        if len(m) > 0:

            tmp['mean_assessment_daytime'] = pd.Series(m[0][0]) 

        else:

            tmp['mean_assessment_daytime'] = pd.Series(0.)

        

        tmp['MAGMAPEAK'] = pd.Series(ext_playtime[i]['MAGMAPEAK'])

        tmp['TREETOPCITY'] = pd.Series(ext_playtime[i]['TREETOPCITY'])

        tmp['CRYSTALCAVES'] = pd.Series(ext_playtime[i]['CRYSTALCAVES'])

        

        acc_mode = data_labels[data_labels.installation_id == u]['accuracy_group'].mode()

        if acc_mode.dropna().empty:

            tmp['accuracy'] = pd.Series(0)

        else:

            tmp['accuracy'] = pd.Series(acc_mode.max())

            

        result = result.append(tmp, ignore_index=True)

    return result
test_features = extract_features(test_data, test_labels)
x_train = features.loc[:, 'activities':'CRYSTALCAVES'].copy()

y_train = features.loc[:, 'accuracy'].astype(int).copy()
x_test = test_features.loc[:, 'activities':'CRYSTALCAVES'].copy()

y_test = test_features.loc[:, 'accuracy'].astype(int).copy()
clf = GradientBoostingClassifier()

clf.fit(x_train, y_train)
res = clf.predict(x_test.values)
print(classification_report(y_test, res))
def gen_submission(data, result):

    labels = ['installation_id', 'accuracy_group']

    submission = pd.DataFrame(columns=labels)

    submission.installation_id = data['id']

    submission.accuracy_group = pd.Series(result)

    submission.to_csv('submission.csv', index=False)

gen_submission(test_features, res)