import pandas as pd

pd.options.display.max_columns = None



X_train_init = pd.read_csv('../input/X_train.csv')

y_train = pd.read_csv('../input/y_train.csv')

X_test_init = pd.read_csv('../input/X_test.csv')
X_train_grouped = X_train_init.groupby(['series_id']).mean().reset_index()

X_train_grouped['mean_orientation_X'] = X_train_grouped['orientation_X']

X_train_grouped['mean_orientation_Y'] = X_train_grouped['orientation_Y']

X_train_grouped['mean_orientation_Z'] = X_train_grouped['orientation_Z']

X_train_grouped['mean_orientation_W'] = X_train_grouped['orientation_W']

del X_train_grouped['orientation_X']

del X_train_grouped['orientation_Y']

del X_train_grouped['orientation_Z']

del X_train_grouped['orientation_W']

del X_train_grouped['measurement_number']

del X_train_grouped['angular_velocity_X']

del X_train_grouped['angular_velocity_Y']

del X_train_grouped['angular_velocity_Z']

del X_train_grouped['linear_acceleration_X']

del X_train_grouped['linear_acceleration_Y']

del X_train_grouped['linear_acceleration_Z']



X_train = pd.merge(X_train_init, X_train_grouped, on = ['series_id'])

del X_train_init

del X_train_grouped
X_test_grouped = X_test_init.groupby(['series_id']).mean().reset_index()

X_test_grouped['mean_orientation_X'] = X_test_grouped['orientation_X']

X_test_grouped['mean_orientation_Y'] = X_test_grouped['orientation_Y']

X_test_grouped['mean_orientation_Z'] = X_test_grouped['orientation_Z']

X_test_grouped['mean_orientation_W'] = X_test_grouped['orientation_W']

del X_test_grouped['orientation_X']

del X_test_grouped['orientation_Y']

del X_test_grouped['orientation_Z']

del X_test_grouped['orientation_W']

del X_test_grouped['measurement_number']

del X_test_grouped['angular_velocity_X']

del X_test_grouped['angular_velocity_Y']

del X_test_grouped['angular_velocity_Z']

del X_test_grouped['linear_acceleration_X']

del X_test_grouped['linear_acceleration_Y']

del X_test_grouped['linear_acceleration_Z']



X_test = pd.merge(X_test_init, X_test_grouped, on = ['series_id'])

del X_test_init

del X_test_grouped
del X_train['orientation_X']

del X_train['orientation_Y']

del X_train['orientation_Z']

del X_train['orientation_W']



del X_test['orientation_X']

del X_test['orientation_Y']

del X_test['orientation_Z']

del X_test['orientation_W']
X_train.describe()
X_train.info()
for measurement_number in range(0,128):

    X_train_mes = X_train[X_train['measurement_number'] == measurement_number]

    #X_train['orientation_X' + str(measurement_number)] = X_train_mes['orientation_X']

    #X_train['orientation_Y' + str(measurement_number)] = X_train_mes['orientation_Y']

    #X_train['orientation_Z' + str(measurement_number)] = X_train_mes['orientation_Z']

    #X_train['orientation_W' + str(measurement_number)] = X_train_mes['orientation_W']

    

    X_train['angular_velocity_X' + str(measurement_number)] = X_train_mes['angular_velocity_X']

    X_train['angular_velocity_Y' + str(measurement_number)] = X_train_mes['angular_velocity_Y']

    X_train['angular_velocity_Z' + str(measurement_number)] = X_train_mes['angular_velocity_Z']



    X_train['linear_acceleration_X' + str(measurement_number)] = X_train_mes['linear_acceleration_X']

    X_train['linear_acceleration_Y' + str(measurement_number)] = X_train_mes['linear_acceleration_Y']

    X_train['linear_acceleration_Z' + str(measurement_number)] = X_train_mes['linear_acceleration_Z']



    X_test_mes = X_test[X_test['measurement_number'] == measurement_number]

    #X_test['orientation_X' + str(measurement_number)] = X_test_mes['orientation_X']

    #X_test['orientation_Y' + str(measurement_number)] = X_test_mes['orientation_Y']

    #X_test['orientation_Z' + str(measurement_number)] = X_test_mes['orientation_Z']

    #X_test['orientation_W' + str(measurement_number)] = X_test_mes['orientation_W']

    

    X_test['angular_velocity_X' + str(measurement_number)] = X_test_mes['angular_velocity_X']

    X_test['angular_velocity_Y' + str(measurement_number)] = X_test_mes['angular_velocity_Y']

    X_test['angular_velocity_Z' + str(measurement_number)] = X_test_mes['angular_velocity_Z']



    X_test['linear_acceleration_X' + str(measurement_number)] = X_test_mes['linear_acceleration_X']

    X_test['linear_acceleration_Y' + str(measurement_number)] = X_test_mes['linear_acceleration_Y']

    X_test['linear_acceleration_Z' + str(measurement_number)] = X_test_mes['linear_acceleration_Z']

    

#del X_train['orientation_X']

#del X_train['orientation_Y']

#del X_train['orientation_Z']

#del X_train['orientation_W']



del X_train['angular_velocity_X']

del X_train['angular_velocity_Y']

del X_train['angular_velocity_Z']



del X_train['linear_acceleration_X']

del X_train['linear_acceleration_Y']

del X_train['linear_acceleration_Z']



del X_train['measurement_number']



#del X_test['orientation_X']

#del X_test['orientation_Y']

#del X_test['orientation_Z']

#del X_test['orientation_W']



del X_test['angular_velocity_X']

del X_test['angular_velocity_Y']

del X_test['angular_velocity_Z']



del X_test['linear_acceleration_X']

del X_test['linear_acceleration_Y']

del X_test['linear_acceleration_Z']



del X_test['measurement_number']
X_train=X_train.fillna(0)

X_test = X_test.fillna(0)

X_gtrain=X_train.groupby(['series_id']).sum().reset_index()

X_gtest=X_test.groupby(['series_id']).sum().reset_index()

train_labels = y_train['surface']



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_gtrain_scaled = scaler.fit_transform(X_gtrain)

X_gtest_scaled = scaler.transform(X_gtest)



# подготовим df для сабмита

sub = pd.DataFrame(X_gtest['series_id'])

sub['surface'] = 0



del X_gtrain

del X_gtest
# деревья далее не рассматриваем, слишком слабые предсказания



#from sklearn.tree import DecisionTreeClassifier



#cl_tree = DecisionTreeClassifier(random_state = 61)

#cl_tree.fit(X_gtrain, train_labels)

#predicted = cl_tree.predict(X_gtest)

#print(predicted)

#print(len(predicted))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=50)

knn.fit(X_gtrain_scaled, train_labels)

predicted = knn.predict(X_gtest_scaled)

print(predicted)

sub['surface'] = predicted

sub.to_csv('submission_knn.csv', index=False)
from sklearn.linear_model import LogisticRegression

C=1e-1

logit = LogisticRegression(C=C, random_state=61)

logit.fit(X_gtrain_scaled, train_labels)

predicted = logit.predict(X_gtest_scaled)

print(predicted)

sub['surface'] = predicted

sub.to_csv('submission_logreg.csv', index=False)
import lightgbm as lgb



param = {

        'objective':'multiclass',

        'num_class':10,

        'metric': 'multi_logloss',

        'learning_rate': 0.016,

        'device': 'gpu',

        'gpu_platform_id': 0,

        'gpu_device_id': 0

    }



gbm = lgb.LGBMClassifier(n_estimators=1100, silent=True)

gbm.fit(X_gtrain_scaled, train_labels, verbose=False)

predicted = gbm.predict(X_gtest_scaled)

print(predicted)

sub['surface'] = predicted

sub.to_csv('submission_lgbm.csv', index=False)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



train_acc = []

val_acc = []

temp_train_acc = []

temp_val_acc = []

#trees_grid = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600]

trees_grid = [1100]

"""

# Train on the training set

for ntrees in trees_grid:

    print('running with ntrees = ', ntrees)

    rfc = RandomForestClassifier(n_estimators=ntrees, random_state=42, n_jobs=-1, oob_score=True)

    temp_train_acc = []

    temp_val_acc = []

    for train_index, val_index in skf.split(X_gtrain_scaled, train_labels):

        X_train, X_val = X_gtrain_scaled[train_index], X_gtrain_scaled[val_index]

        y_train, y_val = train_labels[train_index], train_labels[val_index]

        rfc.fit(X_train, y_train)

        temp_train_acc.append(rfc.score(X_train, y_train))

        temp_val_acc.append(rfc.score(X_val, y_val))

    train_acc.append(temp_train_acc)

    val_acc.append(temp_val_acc)

    print("train/val acc = ", temp_train_acc, temp_val_acc)

"""
import numpy as np    

"""

train_acc, val_acc = np.asarray(train_acc), np.asarray(val_acc)

#np.mean(val_acc)

print("Best accuracy on CV is {:.2f}% with {} trees".format(max(val_acc.mean(axis=1))*100, trees_grid[np.argmax(val_acc.mean(axis=1))]))

"""
import matplotlib.pyplot as plt

plt.style.use('ggplot')

"""

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(trees_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')

ax.plot(trees_grid, val_acc.mean(axis=1), alpha=0.5, color='red', label='cv')

ax.fill_between(trees_grid, val_acc.mean(axis=1) - val_acc.std(axis=1), val_acc.mean(axis=1) + val_acc.std(axis=1), color='#888888', alpha=0.4)

ax.fill_between(trees_grid, val_acc.mean(axis=1) - 2*val_acc.std(axis=1), val_acc.mean(axis=1) + 2*val_acc.std(axis=1), color='#888888', alpha=0.2)

ax.legend(loc='best')

ax.set_ylim([0.55,0.65])

ax.set_ylabel("Accuracy")

ax.set_xlabel("N_estimators");

"""
"""

train_acc = []

val_acc = []

#min_samples_leaf_grid = list(range(1,15))

min_samples_leaf_grid = [3]



for min_samples_leaf in min_samples_leaf_grid:

    print('running with min_samples_leaf = ', min_samples_leaf)

    rfc = RandomForestClassifier(n_estimators=1100, min_samples_leaf=min_samples_leaf, random_state=42, n_jobs=-1, oob_score=True)

    temp_train_acc = []

    temp_val_acc = []

    for train_index, val_index in skf.split(X_gtrain_scaled, train_labels):

        X_train, X_val = X_gtrain_scaled[train_index], X_gtrain_scaled[val_index]

        y_train, y_val = train_labels[train_index], train_labels[val_index]

        rfc.fit(X_train, y_train)

        temp_train_acc.append(rfc.score(X_train, y_train))

        temp_val_acc.append(rfc.score(X_val, y_val))

    train_acc.append(temp_train_acc)

    val_acc.append(temp_val_acc)

    print("train/val acc = ", temp_train_acc, temp_val_acc)

train_acc, val_acc = np.asarray(train_acc), np.asarray(val_acc)

print("Best accuracy on CV is {:.2f}% with min_samples_leaf = {}".format(max(val_acc.mean(axis=1))*100, min_samples_leaf_grid[np.argmax(val_acc.mean(axis=1))]))

"""
#val_acc.shape
"""fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(min_samples_leaf_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')

ax.plot(min_samples_leaf_grid, val_acc.mean(axis=1), alpha=0.5, color='red', label='cv')

ax.fill_between(min_samples_leaf_grid, val_acc.mean(axis=1) - val_acc.std(axis=1), val_acc.mean(axis=1) + val_acc.std(axis=1), color='#888888', alpha=0.4)

ax.fill_between(min_samples_leaf_grid, val_acc.mean(axis=1) - 2*val_acc.std(axis=1), val_acc.mean(axis=1) + 2*val_acc.std(axis=1), color='#888888', alpha=0.2)

ax.legend(loc='best')

ax.set_ylim([0.55,1.05])

ax.set_ylabel("Accuracy")

ax.set_xlabel("min_samples_leaf");"""
"""train_acc = []

val_acc = []

max_depth_grid = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]



for max_depth in max_depth_grid:

    print('running with max_depth = ', max_depth)

    rfc = RandomForestClassifier(n_estimators=1100, min_samples_leaf=3, max_depth=max_depth, random_state=42, n_jobs=-1, oob_score=True)

    temp_train_acc = []

    temp_val_acc = []

    for train_index, val_index in skf.split(X_gtrain_scaled, train_labels):

        X_train, X_val = X_gtrain_scaled[train_index], X_gtrain_scaled[val_index]

        y_train, y_val = train_labels[train_index], train_labels[val_index]

        rfc.fit(X_train, y_train)

        temp_train_acc.append(rfc.score(X_train, y_train))

        temp_val_acc.append(rfc.score(X_val, y_val))

    train_acc.append(temp_train_acc)

    val_acc.append(temp_val_acc)

    print("train/val acc = ", temp_train_acc, temp_val_acc)

train_acc, val_acc = np.asarray(train_acc), np.asarray(val_acc)

print("Best accuracy on CV is {:.2f}% with max_depth = {}".format(max(val_acc.mean(axis=1))*100, max_depth_grid[np.argmax(val_acc.mean(axis=1))]))

"""
"""fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(max_depth_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')

ax.plot(max_depth_grid, val_acc.mean(axis=1), alpha=0.5, color='red', label='cv')

ax.fill_between(max_depth_grid, val_acc.mean(axis=1) - val_acc.std(axis=1), val_acc.mean(axis=1) + val_acc.std(axis=1), color='#888888', alpha=0.4)

ax.fill_between(max_depth_grid, val_acc.mean(axis=1) - 2*val_acc.std(axis=1), val_acc.mean(axis=1) + 2*val_acc.std(axis=1), color='#888888', alpha=0.2)

ax.legend(loc='best')

ax.set_ylim([0.55,0.65])

ax.set_ylabel("Accuracy")

ax.set_xlabel("max_depth");"""
"""max_depth=max_depth_grid[np.argmax(val_acc.mean(axis=1))]

train_acc = []

val_acc = []

max_features_grid = list(range(16,43,1))



for max_features in max_features_grid:

    print('running with max_features = ', max_features)

    rfc = RandomForestClassifier(n_estimators=1100, min_samples_leaf=3, max_depth=max_depth, max_features=max_features, 

                                 random_state=42, n_jobs=-1, oob_score=True)

    temp_train_acc = []

    temp_val_acc = []

    for train_index, val_index in skf.split(X_gtrain_scaled, train_labels):

        X_train, X_val = X_gtrain_scaled[train_index], X_gtrain_scaled[val_index]

        y_train, y_val = train_labels[train_index], train_labels[val_index]

        rfc.fit(X_train, y_train)

        temp_train_acc.append(rfc.score(X_train, y_train))

        temp_val_acc.append(rfc.score(X_val, y_val))

    train_acc.append(temp_train_acc)

    val_acc.append(temp_val_acc)

    print("train/val acc = ", temp_train_acc, temp_val_acc)

train_acc, val_acc = np.asarray(train_acc), np.asarray(val_acc)

print("Best accuracy on CV is {:.2f}% with max_features = {}".format(max(val_acc.mean(axis=1))*100, max_features_grid[np.argmax(val_acc.mean(axis=1))]))

"""
"""fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(max_features_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')

ax.plot(max_features_grid, val_acc.mean(axis=1), alpha=0.5, color='red', label='cv')

ax.fill_between(max_features_grid, val_acc.mean(axis=1) - val_acc.std(axis=1), val_acc.mean(axis=1) + val_acc.std(axis=1), color='#888888', alpha=0.4)

ax.fill_between(max_features_grid, val_acc.mean(axis=1) - 2*val_acc.std(axis=1), val_acc.mean(axis=1) + 2*val_acc.std(axis=1), color='#888888', alpha=0.2)

ax.legend(loc='best')

ax.set_ylim([0.55,0.65])

ax.set_ylabel("Accuracy")

ax.set_xlabel("max_features");"""
train_acc = []

val_acc = []

rfc = RandomForestClassifier(n_estimators=1100, min_samples_leaf=3, max_depth=23, max_features=42,random_state=42, n_jobs=-1, oob_score=True)

temp_train_acc = []

temp_val_acc = []

for train_index, val_index in skf.split(X_gtrain_scaled, train_labels):

    X_train, X_val = X_gtrain_scaled[train_index], X_gtrain_scaled[val_index]

    y_train, y_val = train_labels[train_index], train_labels[val_index]

    rfc.fit(X_train, y_train)

    temp_train_acc.append(rfc.score(X_train, y_train))

    temp_val_acc.append(rfc.score(X_val, y_val))

train_acc.append(temp_train_acc)

val_acc.append(temp_val_acc)

print("train/val acc = ", temp_train_acc, temp_val_acc)

train_acc, val_acc = np.asarray(train_acc), np.asarray(val_acc)

print("Best accuracy on CV is {:.2f}%".format(max(val_acc.mean(axis=1))*100))
rfc = RandomForestClassifier(n_estimators=1100, min_samples_leaf=3, max_depth=23, max_features=42,random_state=42, n_jobs=-1, oob_score=True)

rfc.fit(X_gtrain_scaled, train_labels)

predicted = rfc.predict(X_gtest_scaled)

print(predicted)

sub['surface'] = predicted

sub.to_csv('submission_rforest_1.csv', index=False)
rfc = RandomForestClassifier(n_estimators=1100, min_samples_leaf=3, max_depth=24, max_features=43,random_state=42, n_jobs=-1, oob_score=True)

rfc.fit(X_gtrain_scaled, train_labels)

predicted = rfc.predict(X_gtest_scaled)

print(predicted)

sub['surface'] = predicted

sub.to_csv('submission_rforest_2.csv', index=False)
rfc = RandomForestClassifier(n_estimators=1100, min_samples_leaf=3, max_depth=25, max_features=45,random_state=42, n_jobs=-1, oob_score=True)

rfc.fit(X_gtrain_scaled, train_labels)

predicted = rfc.predict(X_gtest_scaled)

print(predicted)

sub['surface'] = predicted

sub.to_csv('submission_rforest_3.csv', index=False)
rfc = RandomForestClassifier(n_estimators=1100, min_samples_leaf=3, max_depth=27, max_features=47,random_state=42, n_jobs=-1, oob_score=True)

rfc.fit(X_gtrain_scaled, train_labels)

predicted = rfc.predict(X_gtest_scaled)

print(predicted)

sub['surface'] = predicted

sub.to_csv('submission_rforest_4.csv', index=False)
from catboost import CatBoostClassifier, Pool



train_dataset = Pool(X_gtrain_scaled, label=train_labels)

test_dataset = Pool(X_gtest_scaled)



cc = CatBoostClassifier(learning_rate=0.016, random_seed=61, verbose=False, n_estimators=1100, task_type='GPU',loss_function='MultiClass')

cc.fit(train_dataset)

predicted = cc.predict(test_dataset)

sub['surface'] = predicted

sub.to_csv('submission_catboost.csv', index=False)