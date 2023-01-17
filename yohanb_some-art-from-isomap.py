import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.manifold import Isomap

%matplotlib inline
### Read datasets ######################################

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



### Some sets of columns ###############################

cat_cols = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']

dub_cols = ['X262', 'X266', 'X119', 'X296', 'X199', 'X302', 'X113', 'X134',

            'X147', 'X222', 'X360', 'X299', 'X254', 'X245', 'X122', 'X243',

            'X320', 'X76', 'X102', 'X214', 'X239', 'X324', 'X93', 'X107',

            'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330',

            'X347', 'X146', 'X382', 'X232', 'X279', 'X248', 'X253', 'X385',

            'X172', 'X216', 'X213', 'X227', 'X39', 'X35', 'X37', 'X364', 'X365',

            'X226', 'X326', 'X84', 'X244', 'X94', 'X242', 'X247']

const_cols = ['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290',

              'X293', 'X297', 'X330', 'X347']



### Glue train + test ###################################

train['eval_set'] = 0; test['eval_set'] = 1

df = pd.concat([train, test], axis=0, copy=True)

df.reset_index(drop=True, inplace=True) # reset index



### Categorical feature encoding ########################

def add_new_col(x):

    if x not in new_col.keys(): 

        # set n/2 x if is contained in test, but not in train 

        # (n is the number of unique labels in train)

        # or an alternative could be -100 (something out of range [0; n-1]

        return int(len(new_col.keys())/2)

    return new_col[x] # rank of the label



for c in cat_cols:

    # get labels and corresponding means

    new_col = train.groupby(c).y.mean().sort_values().reset_index()

    # make a dictionary, where key is a label and value is the rank of that label

    new_col = new_col.reset_index().set_index(c).drop('y', axis=1)['index'].to_dict()

    # add new column to the dataframe

    df[c + '_new'] = df[c].apply(add_new_col)



### Dummies #############################################

list_of_dummies = []

for col in cat_cols:

    dum_col = pd.get_dummies(df[col], prefix=col, drop_first=True)

    list_of_dummies.append(dum_col)

X_dum = pd.concat(list_of_dummies, axis=1, copy=True)



### Train-test split ####################################

X = pd.concat([df, X_dum], axis=1, copy=True)

X = X.drop(list((set(const_cols) | set(dub_cols) | set(cat_cols))), axis=1)

# Train

X_train = X[X.eval_set == 0]

y_train = X_train.pop('y'); 

X_train = X_train.drop(['eval_set', 'ID'], axis=1)

# Test

X_test = X[X.eval_set == 1]

X_test = X_test.drop(['y', 'eval_set', 'ID'], axis=1)
iso = Isomap(n_neighbors=2, n_components=2)

iso_res = iso.fit_transform(X_train)

plt.figure(figsize=(16,6))

plt.subplot(1,2,1)

cmap = plt.cm.get_cmap('Set2')

sc = plt.scatter(iso_res[:,0], iso_res[:,1], alpha=0.6, c=df.y[:4209], cmap=cmap, s=100)

plt.colorbar(sc)

plt.subplot(1,2,2)

cmap = plt.cm.get_cmap('inferno')

sc = plt.scatter(iso_res[:,0], iso_res[:,1], alpha=0.4, c=df.y[:4209], cmap=cmap, s=100)

plt.colorbar(sc)