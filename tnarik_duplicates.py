# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
feature_columns = train_data.columns[2:]

feature_columns
label_columns = []

for dtype, column in zip(train_data.dtypes, train_data.columns):

    if dtype == object:

        label_columns.append(column)

label_columns
print("{} duplicate entries in training, out of {}, a {:.2f} %".format(

    len(train_data[train_data.duplicated(subset=feature_columns, keep=False)]),

    len(train_data),

    100 * len(train_data[train_data.duplicated(subset=feature_columns, keep=False)]) / len(train_data)

    ))

train_data[train_data.duplicated(subset=feature_columns, keep=False)].sort_values(by=label_columns)
duplicate_std = train_data[train_data.duplicated(subset=feature_columns,

                             keep=False)].groupby(list(feature_columns.values))['y'].aggregate(['std', 'size']).reset_index(drop=True)



duplicate_std.sort_values(by='std', ascending=False)
print("{} duplicate groups in training".format(

    len(train_data[train_data.duplicated(subset=feature_columns,

                             keep=False)].groupby(list(feature_columns.values)).size().reset_index())))



    

train_data[train_data.duplicated(subset=feature_columns,

                             keep=False)].groupby(list(feature_columns.values)).size().reset_index()
print("{} duplicate entries in test, out of {}, a {:.2f} %".format(

    len(test_data[test_data.duplicated(subset=feature_columns, keep=False)]),

    len(test_data),

    100 * len(test_data[test_data.duplicated(subset=feature_columns, keep=False)]) / len(test_data) 

    ))

test_data[test_data.duplicated(subset=feature_columns,

                               keep=False)].groupby(label_columns, axis=0).count()[['ID']]
print("{} duplicate groups in test".format(

    len(test_data[test_data.duplicated(subset=feature_columns,

                             keep=False)].groupby(list(feature_columns.values)).size().reset_index())))



test_data[test_data.duplicated(subset=feature_columns,

                             keep=False)].groupby(list(feature_columns.values)).size().reset_index()
all_data = pd.concat((train_data.drop('y', axis=1), test_data))

print("{} duplicate entries in total, out of {}, a {:.2f} %".format(

    len(all_data[all_data.duplicated(subset=feature_columns, keep=False)]),

    len(all_data),

    100 * len(all_data[all_data.duplicated(subset=feature_columns, keep=False)])/ len(all_data)

    ))



print("{} duplicate groups in total".format(

    len(all_data[all_data.duplicated(subset=feature_columns,

                             keep=False)].groupby(list(feature_columns.values)).size().reset_index())))



all_data[all_data.duplicated(subset=feature_columns,

                             keep=False)].groupby(list(feature_columns.values)).size().reset_index()