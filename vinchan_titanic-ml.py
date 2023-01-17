# import packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv as csv

from sklearn import linear_model

from ggplot import *



# seed the rng

np.random.seed=314159



# read in test and training files

train_df_ = pd.read_csv('../input/train.csv', header=0)

test_df_ = pd.read_csv('../input/test.csv',header=0)
# add a column in train for subsetting

train_df_['subset'] = np.random.uniform(0,1, train_df_.shape[0])



# subset the data into learning and validation sets

train_learn_df_ = train_df_[train_df_.subset < .8]

train_validate_df_ = train_df_[train_df_.subset >= .8]



# check the shapes of the subset

print(train_learn_df_.shape)

print(train_validate_df_.shape)

print(train_df_.shape)
print(pd.crosstab(train_df_.Pclass, train_df_.Survived, margins=True))

print()

print(pd.crosstab(train_df_.Pclass, train_df_.Survived, margins=True).apply(lambda r: 2*r/r.sum(), axis=1))

print(pd.crosstab(train_df_.Embarked, train_df_.Survived, margins=True).apply(lambda r: 2*r/r.sum(), axis=1))
train_df_.Pclass = pd.Categorical(train_df_.Pclass)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.