# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#### get the data
train_users   = pd.read_csv('../input/train_users_2.csv')
test_users    = pd.read_csv('../input/test_users.csv')
gender = pd.read_csv('../input/age_gender_bkts.csv')
# Any results you write to the current directory are saved as output.
#### let's build a random forest to tackle this challenge


