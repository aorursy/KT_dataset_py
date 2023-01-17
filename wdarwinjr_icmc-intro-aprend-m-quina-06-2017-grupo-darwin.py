import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# load data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



# remove "ID" column

train_df.drop("ID", axis=1, inplace=True)

test_df.drop("ID", axis=1, inplace=True)



# remove constant columns

for col in train_df.columns:

    if (train_df[col].values.std()==0):

        train_df.drop(col, axis=1, inplace=True)

        test_df.drop(col, axis=1, inplace=True)



# remove duplicate columns

cols_to_drop = []

for i in range(len(train_df.columns)):

    col1 = train_df.columns[i]

    for j in range(i+1, len(train_df.columns)):

        col2 = train_df.columns[j]

        if (np.array_equal(train_df[col2].values, train_df[col1].values)):

            cols_to_drop.append(col2)

for col in cols_to_drop:

    train_df.drop(col, axis=1, inplace=True)

    test_df.drop(col, axis=1, inplace=True)



# split original train data frame into 80:20 train and test subsets for internal tests

from sklearn.model_selection import train_test_split

train, test = train_test_split(train_df, test_size = 0.2)



# split features and target variables

train_features_df = train_df.drop('TARGET', axis=1)

train_target_df = train_df['TARGET']

test_features_df = test_df
[len(train_df), len(train_df[0]), len(test_df)]
categoric_ref = 20

feature_types = []

for i in range(len(train_features_df.columns)):

    col = train_features_df.columns[i]

    var_set = set(train_features_df[col].values)

    var_nr = len(var_set)

    var_sum = sum(var_set)

    if (var_sum-int(var_sum)>0.01):

        var_type = "Continuous"

    else:

        if (var_nr>categoric_ref):

            var_type = "Discrete"

        else:

            var_type = "Categoric"

    feature_types.append(var_type)

feature_types[0:20]
# replace extreme values by mean values of each feature

import statistics as stc

vars_max = max([max(train_features_df[train_features_df.columns[i]].values) for i in range(len(train_features_df.columns))])

vars_min = min([min(train_features_df[train_features_df.columns[i]].values) for i in range(len(train_features_df.columns))])

print([vars_max, vars_min])

for i in range(len(train_features_df.columns)):

    col = train_features_df.columns[i]

    if (feature_types[i] != "Categoric"):

        var_replace = train_features_df[col].values.mean()

    else:

        feat_vals = set(train_features_df[col].values)

        try:

            feat_vals = feat_vals.discard(vars_min)

            feat_vals = feat_vals.discard(vars_max)

            var_replace=stc.mode(feat_vals)

        except:

            var_replace=0

    train_features_df[col].replace(to_replace=vars_max, value=var_replace, inplace=True)

    train_features_df[col].replace(to_replace=vars_min, value=var_replace, inplace=True)

vars_max = max([max(train_features_df[train_features_df.columns[i]].values) for i in range(len(train_features_df.columns))])

vars_min = min([min(train_features_df[train_features_df.columns[i]].values) for i in range(len(train_features_df.columns))])

print([vars_max, vars_min])
train_features_df.head()
train_target_df.head()
test_features_df.head()
# show 20 feature distributions randomly chosen

from random import randint

cols_nr = 5

rows_nr = 20 #int(vars_nr/5) +1

plt.figure(figsize=(15, 50))

for i in range(20):

    ind = randint(0,len(train_features_df.columns))

    col = train_features_df.columns[ind]

    plt.subplot(rows_nr, cols_nr, i + 1)

    plt.hist(train_features_df[col])

    if (feature_types[ind]=="Continuous"):

        plt.gca().set_xscale("log")

    plt.gca().set_yscale("log")

    plt.xlabel("${}$".format(col), fontsize=14)

    if i == 0:

        plt.ylabel("$TARGET$", fontsize=14)

    plt.title("{}".format(feature_types[ind]+" - id="+str(ind)), fontsize=16)

    plt.tight_layout()

plt.show()
set(train_features_df[train_features_df.columns[176]].values)
# ordering all variables according to its MI regarding to TARGET

from sklearn.metrics.cluster import normalized_mutual_info_score

import bisect

mi = []

var_ids_mitop = []

for i in range(len(train_features_df.columns)):

    scoring_var = train_features_df[train_features_df.columns[i]].values

    mi.append(round(normalized_mutual_info_score(train_target_df, scoring_var), 4))

    inserted=False

    for j in range(len(var_ids_mitop)):

        if (mi[i]>mi[var_ids_mitop[j]]):

            var_ids_mitop.insert(j, i)

            inserted=True

            break

    if (not inserted):

        var_ids_mitop.append(i)



# selecting the best variables with low MI (<0.5) between them

mi_ref = 0.5

var_ids_mitop_relev = []

for i in range(len(var_ids_mitop)):

    var_id = var_ids_mitop[i]

    var_ref = train_features_df[train_features_df.columns[var_id]].values

    low_mi = True

    for j in range(len(var_ids_mitop_relev)):

        var = train_features_df[train_features_df.columns[var_ids_mitop_relev[j]]].values

        if (normalized_mutual_info_score(var, var_ref)>mi_ref):

            low_mi = False

            break

    if (low_mi):

        var_ids_mitop_relev.append(var_id)



mis_top20 = [[train_features_df.columns[i], feature_types[i], mi[i]] for i in var_ids_mitop_relev[0:20]]

mis_top20
len(var_ids_mitop_relev)
cols_nr = 5

rows_nr = 4 #int(vars_nr/5) +1

plt.figure(figsize=(15, 10))

for i in range(20):

    ind = randint(0, len(var_ids_mitop_relev))

    col = train_features.columns[var_ids_mitop_relev[ind]]

    plt.subplot(rows_nr, cols_nr, i + 1)

    plt.scatter(train_features[col], train_target.values)

    plt.xlabel("$var_{}$".format(str(var_ids_mitop_relev[ind] + 1)), fontsize=14)

    if i == 0:

        plt.ylabel("$TARGET$", fontsize=14)

    plt.title("MI={:.2f}".format(mi[var_ids_mitop_relev[ind]]), fontsize=16)

    plt.tight_layout()

plt.show()
cols_nr = 5

rows_nr = 4 #int(vars_nr/5) +1

plt.figure(figsize=(15, 10))

for i in range(20):

    var_id = var_ids_mitop_relev[i]

    col = train_features.columns[var_id]

    plt.subplot(rows_nr, cols_nr, i + 1)

    plt.hist(train_features[col])

    if (feature_types[var_id]=="Continuous"):

        plt.gca().set_xscale("log")

    plt.gca().set_yscale("log")

    plt.xlabel("${}$".format(col), fontsize=14)

    if i == 0:

        plt.ylabel("$TARGET$", fontsize=14)

    plt.title("MI={:.2f}".format(mi[var_id]), fontsize=16)

    plt.tight_layout()

plt.show()
# split data into train and test

from sklearn.cross_validation import train_test_split

test = test_features

X = train_features

y = train_target.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1729)



## # Feature selection

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(random_state=1729)

selector = clf.fit(train_features, train_target)



# plot most important features

feat_imp = pd.Series(clf.feature_importances_, index = train_features.columns).sort_values(ascending=False)

feat_imp[:40].plot(kind='bar', title='Feature Importances according to ExtraTreesClassifier', figsize=(12, 8))

plt.ylabel('Feature Importance Score')

plt.subplots_adjust(bottom=0.3)

plt.savefig('1.png')

plt.show()



# clf.feature_importances_ 

fs = SelectFromModel(selector, prefit=True)



X_train = fs.transform(train_features)

X_test = fs.transform(test_features)



print(X_train.shape, X_test.shape)



# calculate the auc score

#print("Roc AUC: ", roc_auc_score(y_test, m2_xgb.predict_proba(X_test)[:,1], average='macro'))

              

## # Submission

#probs = m2_xgb.predict_proba(test)



#submission = pd.DataFrame({"ID":test_id, "TARGET": probs[:,1]})

#submission.to_csv("submission.csv", index=False)
from numpy import unique

from numpy import random 



def balanced_sample_maker(X, y, random_seed=None):

    """ return a balanced data set by oversampling minority class 

        current version is developed on assumption that the positive

        class is the minority.

    Parameters:

    ===========

    X: {numpy.ndarrray}

    y: {numpy.ndarray}

    """

    uniq_levels = unique(y)

    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:

        random.seed(random_seed)



    # find observation index of each class levels

    groupby_levels = {}

    for ii, level in enumerate(uniq_levels):

        obs_idx = [idx for idx, val in enumerate(y) if val == level]

        groupby_levels[level] = obs_idx



    # oversampling on observations of positive label

    sample_size = uniq_counts[0]

    over_sample_idx = random.choice(groupby_levels[1], size=sample_size, replace=True).tolist()

    balanced_copy_idx = groupby_levels[0] + over_sample_idx

    random.shuffle(balanced_copy_idx)



    return X[balanced_copy_idx, :], y[balanced_copy_idx]



train_features_balanced, train_target_balanced = balanced_sample_maker(train_features, train_target.ravel())

[len(train_features), len(train_features_balanced)]
import sklearn.naive_bayes as nb

feature_idxs = var_ids_mitop_relev[0:1]

train_features_array = np.array(train_features[feature_idxs])

train_target_array = np.array(train_target.ravel())

test_features_array = np.array(test_features[feature_idxs])

#gnb = nb.GaussianNB().fit(train_features_array, train_target_array)

train_features_array = np.array([[0,1] for i in range(50)]+[[1,0] for i in range(50)])

train_target_array = np.array([0 for i in range(50)]+[1 for i in range(50)])

mnb = nb.MultinomialNB().fit(train_features_array, train_target_array)

train_prediction_array = mnb.predict(train_features_array)

#test_prediction_array = mnb.predict(test_features_array)

# calculate the auc score

print("Roc AUC: ", roc_auc_score(train_target_array, train_prediction_array, average='macro'))

print("Roc AUC: ", roc_auc_score(train_target_array, [randint(0,1) for i in range(len(train_target_array))], average='macro'))



#out = [sum(train_prediction_array), sum(train_target_array), sum(test_prediction_array), len(test_prediction_array)]

#out
selected_features = feat_imp.keys().values[0:20]

train_features_array = np.array(train_features[selected_features])

train_target_array = np.array(train_target).ravel()

test_features_array = np.array(test_features[selected_features])

gnb = nb.GaussianNB().fit(train_features_array, train_target_array)

#mnb = nb.MultinomialNB().fit(train_features, train_target)

train_prediction_array = gnb.predict(train_features_array)

test_prediction_array = gnb.predict(test_features_array)

out = [sum(train_prediction_array), sum(train_target_array), sum(test_prediction_array), len(test_prediction_array)]

out
print(gnb.theta_)

print(gnb.sigma_)
df = train_features

var_maxs = [max(df[df.columns[i]].values) for i in range(len(df.columns))]

var_maxs