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



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import datetime

import pdb

import re

import gc

from matplotlib import pyplot as plt

import seaborn as sns

import json
train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
# encode title

list_of_user_activities = list(set(train['title'].value_counts().index).union(set(test['title'].value_counts().index)))

activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

train['title'] = train['title'].map(activities_map)

test['title'] = test['title'].map(activities_map)

train_labels['title'] = train_labels['title'].map(activities_map)

win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))

win_code[activities_map['Bird Measurer (Assessment)']] = 4110

train['timestamp'] = pd.to_datetime(train['timestamp'])

test['timestamp'] = pd.to_datetime(test['timestamp'])

#[ 7,  9, 31, 22,  1] assessments
titles = train.title.unique()

activities = train.type.unique()

worlds = train.world.unique()

event_codes = train.event_code.unique()

types = train.type.unique()
import json

for e in types:

    if e == 'Clip':

        continue

    titles_ = train.loc[ train.type == e, :].title.unique()

    print(f'{e} has event4070 ', len(train.loc[(train['type'] == e)&(train['event_code'] == 4070), 'event_data'].map(lambda x: json.loads(x))))

    for i in titles_:

        df_tmp  =train.loc[(train['type'] == e)&(train['event_code'] == 4070)&(train['title']==i), 'event_data']

        print(f'{e} title{i} {4070} has', len(df_tmp))

        coordinates =df_tmp.map(lambda x: json.loads(x)['coordinates'])



        x = (coordinates.map(lambda d:d['x']) / coordinates.map(lambda d:d['stage_width']) * 500).map(np.floor).map(int)

        y = (coordinates.map(lambda d:d['y']) / coordinates.map(lambda d:d['stage_height'])* 400).map(np.floor).map(int)

        x = x.clip(0,499)

        y = y.clip(0,399)

        heatmap_ = np.zeros((400, 500))

        for y0, x0 in zip(x,y):

            heatmap_[x0,y0] += 1

        heatmap_ = np.clip(heatmap_,0,10)

        f , ax = plt.subplots(figsize = (14,12))

        plt.title(f'{e} title{i} {4070} has {len(df_tmp)}')

        sns.heatmap(heatmap_)

    