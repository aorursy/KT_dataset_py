# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import sys

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 500)

pd.set_option('display.max_rows', 1000)

import time
def str_to_num(string):

    return int(string.split(" ")[1])



train=pd.read_csv('../input/train.csv', converters={'location':str_to_num})

test=pd.read_csv('../input/test.csv', converters={'location':str_to_num})

event=pd.read_csv('../input/event_type.csv', converters={'event_type':str_to_num})

log_feature=pd.read_csv('../input/log_feature.csv', converters={'log_feature':str_to_num})

severity=pd.read_csv('../input/severity_type.csv', converters={'severity_type':str_to_num})

resource=pd.read_csv('../input/resource_type.csv', converters={'resource_type':str_to_num})
event.head(3)
traintest=train.append(test)

traintest=traintest.merge(right=severity, on='id')

resource_by_id=pd.get_dummies(resource,columns=['resource_type'])

resource_by_id=resource_by_id.groupby(['id']).sum().reset_index(drop=False)



event_by_id=pd.get_dummies(event,columns=['event_type'])

event_by_id=event_by_id.groupby(['id']).sum().reset_index(drop=False)
resource_by_id.head(5)
log_feature_dict={}



for row in log_feature.itertuples():

    if row.id not in log_feature_dict:

        log_feature_dict[row.id]={}

    if row.log_feature not in log_feature_dict[row.id]:

        log_feature_dict[row.id][row.log_feature]=row.volume



colnames=['id']

for i in range(1,387):

    colnames.append('log_feature_'+str(i))



log_feature_by_id_np=np.zeros((18552,387))

count=0

for key, feature_dict in log_feature_dict.items():

    log_feature_by_id_np[count, 0]=np.int(key)

    for feature, volume in feature_dict.items():

        log_feature_by_id_np[count, feature]=np.int(volume)

    count+=1

log_feature_by_id=pd.DataFrame(data=log_feature_by_id_np, columns=colnames, dtype=np.int)
log_feature_by_id.head(3)
print(traintest.shape)

print(resource_by_id.shape)

print(event_by_id.shape)

print(log_feature_by_id.shape)
traintest=traintest.merge(right=severity, on='id')

print(traintest.shape)



traintest=traintest.merge(right=resource_by_id, on='id')

print(traintest.shape)



traintest=traintest.merge(right=event_by_id, on='id')

print(traintest.shape)



traintest=traintest.merge(right=log_feature_by_id, on='id')

print(traintest.shape)
train_input=traintest.loc[0:train.shape[0]-1]

print("train_input shape is", train_input.shape)



test_input=traintest.loc[train.shape[0]::]

print("test_inpue shape is", test_input.shape)