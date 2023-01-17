from __future__ import division, print_function

# отключим всякие предупреждения Anaconda

import warnings

import numpy as np

import pandas as pd

warnings.filterwarnings('ignore')

%matplotlib inline

from matplotlib import pyplot as plt

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = 10, 6

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
df_train = pd.read_csv('../input/Kannada-MNIST/train.csv')

images = df_train.drop("label", axis=1).values.astype('float32')

labels = df_train['label'].values.astype('float32')

del df_train
df_test = pd.read_csv('../input/Kannada-MNIST/test.csv')

df_test = df_test.drop("id", axis=1).values.astype('float32')
# Initialize the stratified breakdown of our dataset for validation

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Initialize our classifier with default parameters

rfc = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)
# teach on a training dataset

results = cross_val_score(rfc, images, labels, cv=skf)
# evaluate the accuracy of the train dataset

print("CV accuracy score: {:.2f}%".format(results.mean()*100))
# Initialize validation

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Create lists to maintain accuracy on the training and test dataset

train_acc = []

test_acc = []

temp_train_acc = []

temp_test_acc = []

trees_grid = [5, 10, 15, 20, 30, 50, 75, 100]



# Teach on a training dataset

for ntrees in trees_grid:

    rfc = RandomForestClassifier(n_estimators=ntrees, random_state=42, n_jobs=-1, oob_score=True)

    temp_train_acc = []

    temp_test_acc = []

    for train_index, test_index in skf.split(images, labels):

        X_train, X_test = images[train_index], images[test_index]

        y_train, y_test = labels[train_index], labels[test_index]

        rfc.fit(X_train, y_train)

        temp_train_acc.append(rfc.score(X_train, y_train))

        temp_test_acc.append(rfc.score(X_test, y_test))

    train_acc.append(temp_train_acc)

    test_acc.append(temp_test_acc)

    

train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)

print("Best accuracy on CV is {:.2f}% with {} trees".format(max(test_acc.mean(axis=1))*100, 

                                                        trees_grid[np.argmax(test_acc.mean(axis=1))]))
plt.style.use('ggplot')



fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(trees_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')

ax.plot(trees_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')

ax.fill_between(trees_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)

ax.fill_between(trees_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)

ax.legend(loc='best')

ax.set_ylim([0.88,1.02])

ax.set_ylabel("Accuracy")

ax.set_xlabel("N_estimators");


# Create lists to maintain accuracy on the training and test dataset

train_acc = []

test_acc = []

temp_train_acc = []

temp_test_acc = []

max_depth_grid = [3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]



# Teach on a training dataset

for max_depth in max_depth_grid:

    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True, max_depth=max_depth)

    temp_train_acc = []

    temp_test_acc = []

    for train_index, test_index in skf.split(images, labels):

        X_train, X_test = images[train_index], images[test_index]

        y_train, y_test = labels[train_index], labels[test_index]

        rfc.fit(X_train, y_train)

        temp_train_acc.append(rfc.score(X_train, y_train))

        temp_test_acc.append(rfc.score(X_test, y_test))

    train_acc.append(temp_train_acc)

    test_acc.append(temp_test_acc)

    

train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)

print("Best accuracy on CV is {:.2f}% with {} max_depth".format(max(test_acc.mean(axis=1))*100, 

                                                        max_depth_grid[np.argmax(test_acc.mean(axis=1))]))



fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(max_depth_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')

ax.plot(max_depth_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')

ax.fill_between(max_depth_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)

ax.fill_between(max_depth_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)

ax.legend(loc='best')

ax.set_ylim([0.88,1.02])

ax.set_ylabel("Accuracy")

ax.set_xlabel("Max_depth");
# Create lists to maintain accuracy on the training and test dataset

train_acc = []

test_acc = []

temp_train_acc = []

temp_test_acc = []

min_samples_leaf_grid = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]



# Teach on a training dataset

for min_samples_leaf in min_samples_leaf_grid:

    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, 

                                 oob_score=True, min_samples_leaf=min_samples_leaf)

    temp_train_acc = []

    temp_test_acc = []

    for train_index, test_index in skf.split(images, labels):

        X_train, X_test = images[train_index], images[test_index]

        y_train, y_test = labels[train_index], labels[test_index]

        rfc.fit(X_train, y_train)

        temp_train_acc.append(rfc.score(X_train, y_train))

        temp_test_acc.append(rfc.score(X_test, y_test))

    train_acc.append(temp_train_acc)

    test_acc.append(temp_test_acc)

    

train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)

print("Best accuracy on CV is {:.2f}% with {} min_samples_leaf".format(max(test_acc.mean(axis=1))*100, 

                                                        min_samples_leaf_grid[np.argmax(test_acc.mean(axis=1))]))
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(min_samples_leaf_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')

ax.plot(min_samples_leaf_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')

ax.fill_between(min_samples_leaf_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)

ax.fill_between(min_samples_leaf_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)

ax.legend(loc='best')

ax.set_ylim([0.88,1.02])

ax.set_ylabel("Accuracy")

ax.set_xlabel("Min_samples_leaf");
# Create lists to maintain accuracy on the training and test dataset

train_acc = []

test_acc = []

temp_train_acc = []

temp_test_acc = []

max_features_grid = [2, 4, 6, 8, 10, 12, 14, 16]



# Teach on a training dataset

for max_features in max_features_grid:

    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, 

                                 oob_score=True, max_features=max_features)

    temp_train_acc = []

    temp_test_acc = []

    for train_index, test_index in skf.split(images, labels):

        X_train, X_test = images[train_index], images[test_index]

        y_train, y_test = labels[train_index], labels[test_index]

        rfc.fit(X_train, y_train)

        temp_train_acc.append(rfc.score(X_train, y_train))

        temp_test_acc.append(rfc.score(X_test, y_test))

    train_acc.append(temp_train_acc)

    test_acc.append(temp_test_acc)

    

train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)

print("Best accuracy on CV is {:.2f}% with {} max_features".format(max(test_acc.mean(axis=1))*100, 

                                                        max_features_grid[np.argmax(test_acc.mean(axis=1))]))



fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(max_features_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')

ax.plot(max_features_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')

ax.fill_between(max_features_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)

ax.fill_between(max_features_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)

ax.legend(loc='best')

ax.set_ylim([0.88,1.02])

ax.set_ylabel("Accuracy")

ax.set_xlabel("Max_features");
# Сделаем инициализацию параметров, по которым хотим сделать полный перебор

parameters = {'max_features': [4, 7, 10, 13], 'min_samples_leaf': [1, 3, 5, 7], 'max_depth': [5,10,15,20]}

rfc = RandomForestClassifier(n_estimators=100, random_state=42, 

                             n_jobs=-1, oob_score=True)

gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)

gcv.fit(images, labels)
gcv.best_estimator_, gcv.best_score_
rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                        max_depth=20, max_features=13, max_leaf_nodes=None,

                        min_impurity_decrease=0.0, min_impurity_split=None,

                        min_samples_leaf=1, min_samples_split=2,

                        min_weight_fraction_leaf=0.0, n_estimators=100,

                        n_jobs=-1, oob_score=True, random_state=42, verbose=0,

                        warm_start=False)
rfc.fit(images, labels)
pred = rfc.predict(df_test).astype(int)
def write_preds(preds, fname):

    pd.DataFrame({"id": list(range(0,len(preds))), "label": preds}).to_csv(fname, index=False, header=True)
write_preds(pred, "samplesubmission.csv")