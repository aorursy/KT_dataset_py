# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 500)

pd.set_option('display.max_rows', 1000)

import time

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

severity=pd.read_csv('../input/severity_type.csv')

event=pd.read_csv('../input/event_type.csv')

log_feature=pd.read_csv('../input/log_feature.csv')

resource=pd.read_csv('../input/resource_type.csv')
# first look at train & test data

train.head(3)
train.fault_severity.value_counts()
train.shape
test.shape
event.shape
event.head(5)
test.head(3)
print('train', train.shape)

print('test', test.shape)
#have a look a severity

severity.head(3)
print('severity', severity.shape)

severity.shape[0]==train.shape[0]+test.shape[0]
severity['id'].nunique()
train.append(test)['id'].nunique()
print('severity', severity['id'].describe())

print('train + test', train.append(test)['id'].describe())
event.head(3)
event.shape
event['id'].value_counts()
# have a look at the frequence of each event

event.event_type.value_counts()
resource.head(3)
resource.shape
resource['id'].value_counts()
# have a look at the frequence of each resource

resource.resource_type.value_counts()
log_feature.head(3)
# there are larger number of rows in log_feature than number of rows for train + test

log_feature.shape
log_feature['id'].value_counts()

# so same number of unique ids with train+test, but some ids are repeated 
# have a look at the frequence of each log_feature

log_feature.log_feature.value_counts()
log_feature.loc[log_feature.log_feature=='feature 312']['volume'].describe()
log_feature.loc[log_feature.log_feature=='feature 232']['volume'].describe()
log_feature.loc[log_feature.log_feature=='feature 82']['volume'].describe()