# standard imports

import numpy as np

import pandas as pd

import os

from tqdm.notebook import tqdm

# modeling

from sklearn.ensemble import RandomForestClassifier

import scipy.stats.distributions as dists

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split

from sklearn import metrics

#viz

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/lish-moa/train_features.csv")

df.head()
df.columns[(df.dtypes.values != np.dtype('float64'))]
df_cp_type = df[['cp_type','sig_id']].groupby('cp_type').count().reset_index()

df_cp_type.head()
df_cp_dose = df[['cp_dose','sig_id']].groupby('cp_dose').count().reset_index()

df_cp_dose.head()
print("training dataset size: ", df.values.shape)
targets = pd.read_csv("../input/lish-moa/train_targets_scored.csv")

targets.head()
print("training dataset target size: ", targets.values.shape)
not_same = 0

for i,j in zip(df['sig_id'].values,targets['sig_id'].values):

    if i!=j:

        not_same += 1

print("if ids in training set and target data are ordered the same, the following value should be 0:")

print(not_same)
df['ctl_vehicle'] = df['cp_type'].apply(lambda x: 1 if x=='ctl_vehicle' else 0)

df.drop(['cp_type'], axis=1, inplace=True)

df['D1'] = df['cp_dose'].apply(lambda x: 1 if x=='D1' else 0)

df.drop(['cp_dose'], axis=1, inplace=True)

df.head()
features_list = df.columns.tolist()

features_list.remove('sig_id')

print("total features: ",len(features_list))

target_list = targets.columns.tolist()

target_list.remove('sig_id')

print("total target categories: ",len(target_list))
# create a pandas dataframe with labels and a column called 'total' that we'll use to count on later

labels = np.argmax(targets[target_list].values, axis=1)

ldf = pd.DataFrame()

ldf['label'] = labels

ldf['total'] = 1

ldf.head()
# group by label and count up our ones to get a total number of oberservations per label

ldfgrp = ldf.groupby('label').agg({'total':'count'}).reset_index()

ldfgrp.head()
# plot distribution of labels

plt.bar(ldfgrp['label'],ldfgrp['total'])

plt.show()
# remove 0 group

ldfgrpf = ldfgrp[ldfgrp.label!=0]

plt.bar(ldfgrpf['label'],ldfgrpf['total'])

plt.show()
ldfgrp = ldfgrp.sort_values(by=['total'])

ldfgrp.head()
ldfgrp = ldfgrp.sort_values(by=['total'], ascending=False)

ldfgrp.head(25)
mean_predictions = targets.mean()

mean_predictions[:5]
targets[target_list].head()
ldfgrp.label.values[:25]
new_target_idxs = ldfgrp.label.values[:25].tolist()

new_targets = [target_list[i] for i in new_target_idxs]

new_targets[:5]
targets[new_targets].head()
X_train, X_holdout, y_train, y_holdout = train_test_split(df[features_list], np.argmax(targets[new_targets].values, axis=1), test_size=0.5, random_state=42)

print("training data size: ", X_train.values.shape)

print("training data target size: ", y_train.shape)

print("holdout data size: ", X_holdout.values.shape)

print("holdout data target size: ", y_holdout.shape)
# downsampling

X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=0)
clf = RandomForestClassifier(n_jobs=-1)



distributions = dict(n_estimators = dists.randint(4,1000),

                     max_depth = dists.randint(1,30),

                     max_features=dists.uniform(loc=0.05,scale=0.95))



search = RandomizedSearchCV(estimator=clf,

                           param_distributions=distributions,

                           verbose=1,

                           cv=2,

                           n_iter=4,

                           n_jobs=-1,

                           scoring='roc_auc_ovr_weighted',

                           random_state=0)
%%time



search_results = search.fit(X_train,y_train)
print("Best: %f using %s" % (search_results.best_score_,search_results.best_params_))
model = search_results.best_estimator_
pred = np.argmax(model.predict_proba(X_holdout), axis=1)

print("holdout score: ", metrics.cohen_kappa_score(pred,y_holdout))
test = pd.read_csv("../input/lish-moa/test_features.csv")

test.head()
test['ctl_vehicle'] = test['cp_type'].apply(lambda x: 1 if x=='ctl_vehicle' else 0)

test.drop(['cp_type'], axis=1, inplace=True)

test['D1'] = test['cp_dose'].apply(lambda x: 1 if x=='D1' else 0)

test.drop(['cp_dose'], axis=1, inplace=True)

test.head()
sub = pd.read_csv("../input/lish-moa/sample_submission.csv")

sub.head()
not_same = 0

for i,j in zip(test['sig_id'].values,sub['sig_id'].values):

    if i!=j:

        not_same += 1

print("if ids in training set and target data are ordered the same, the following value should be 0:")

print(not_same)
pred = model.predict_proba(test[features_list])

print("prediction output shape: ", pred.shape)
# make a copy of submission with all targets

new_sub = sub[target_list].copy()

new_sub.head()
# change prediction to average value from training data for each row

for i in tqdm(range(len(sub))):

    new_sub.loc[i,:] = mean_predictions

new_sub.head()
new_sub.iloc[:,new_target_idxs] = pred
# reload submissions file and add predictions (this way we have predictions and sig_id field in final submission file)

sub = pd.read_csv("../input/lish-moa/sample_submission.csv")

sub.iloc[:,1:] = new_sub.values
# submit

sub.to_csv("submission.csv", index=False)

sub.head()