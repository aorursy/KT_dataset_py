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
input_path = '/kaggle/input/data-science-bowl-2019/'
labels = pd.read_csv(input_path + 'train_labels.csv')
groups = labels[['title', 'accuracy_group']].groupby(['title'])
for group in groups:

    print(group[0])

    print(group[1]['accuracy_group'].value_counts())
test = pd.read_csv(input_path + 'test.csv')

test
test_assessments = test.query('type == "Assessment"')
test_assessments
# for submission we need just the latest assesments for each installation id

test_latest = test_assessments.sort_values('timestamp')[['installation_id', 'title']].groupby('installation_id').tail(1)

test_latest
test_latest['accuracy_group'] = test_latest['title'].apply(lambda x: 

                                                                     0 if 'Bird' in x or 'Chest' in x else 3)

test_latest
test_latest[['installation_id', 'accuracy_group']].to_csv('submission.csv', index=False)