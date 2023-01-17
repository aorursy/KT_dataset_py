import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings("ignore")
train=pd.read_csv('../input/data-science-bowl-2019/train.csv')

train.timestamp=pd.to_datetime(train['timestamp'])

train=train.sort_values(by=['installation_id','timestamp'])
test=pd.read_csv('../input/data-science-bowl-2019/test.csv')

test.timestamp=pd.to_datetime(test['timestamp'])

test=test.sort_values(by=['installation_id','timestamp'])
train_labels=pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")

specs=pd.read_csv("../input/data-science-bowl-2019/specs.csv")
len(train_labels)
#Extract installation_id without assessment record

no_record=train[~train.installation_id.isin(train_labels.installation_id)]



#Id of players who have no record but accually took the assessment

id_no_record_with_assessment=no_record[no_record.type=='Assessment'].installation_id.unique()



#The unique event code of these players

event_code_no_record=no_record[no_record.type=='Assessment'].event_code.unique()
print("Number of players with no record:", no_record.installation_id.nunique())

print("Number of players 'took' the assessment but has no record:", len(id_no_record_with_assessment))

print("Event code of the players above:\n", event_code_no_record)
no_record_with_assessment=no_record[no_record.installation_id.isin(id_no_record_with_assessment)]

no_record_with_assessment=no_record_with_assessment[no_record_with_assessment.type=='Assessment']

#Note that I have sort the timestamp and game_time is accumulated, thus 

#the following code tell me the max game time for each sesstion

by_session_no_record=no_record_with_assessment.groupby("game_session").last()
print("Number of assessment session", len(by_session_no_record))

print("Number of assessment that has time less than 5s:",(by_session_no_record.game_time.values <5000).sum())
record=train[(train.installation_id.isin(train_labels.installation_id)) & (train.type=='Assessment')]

by_session_record=record.groupby("game_session").last()

print("Number of assessment session", len(by_session_record))

print("Number of assessment that has time longer than 5s:",(by_session_record.game_time.values<5000).sum())
index_lessthan_5s=by_session_record[by_session_record.game_time.values<5000].index
train_labels[train_labels.game_session.isin(index_lessthan_5s)].accuracy_group.value_counts()
test_assessment_count=test.groupby('installation_id').apply(lambda x: x[x.type=='Assessment'].game_session.nunique())
test_assessment_count.value_counts()