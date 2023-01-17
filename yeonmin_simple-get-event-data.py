# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import json

# Any results you write to the current directory are saved as output.
def get_event_data(specs, train, test, col):

    specs[col] = specs['args'].apply(lambda x: 1 if x.find(f'"{col}"')!=-1 else 0)

    idx = train['event_id'].isin(specs.loc[specs[col]==1,'event_id'].values)

    train.loc[idx,col] = train.loc[idx,'event_data'].apply(lambda x: json.loads(x)[col])



    idx = test['event_id'].isin(specs.loc[specs[col]==1,'event_id'].values)

    test.loc[idx,col] = test.loc[idx,'event_data'].apply(lambda x: json.loads(x)[col])

    return specs, train, test
train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
specs, train, test = get_event_data(specs, train, test, 'level')
train.loc[train['level'].notnull()].head()
train_unique_event_id = set(train.loc[train['level'].notnull(),'event_id'].unique())

specs_unique_event_id = set(specs.loc[specs['level']==1,'event_id'].values)

print(train_unique_event_id.difference(specs_unique_event_id), specs_unique_event_id.difference(train_unique_event_id))