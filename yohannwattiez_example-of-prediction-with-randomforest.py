import riiideducation



import numpy as np

import os

import pandas as pd

from sklearn.ensemble import RandomForestClassifier



from sklearn import metrics
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', nrows=10**5, 

                    dtype={'row_id': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',

                              'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 'prior_question_elapsed_time': 'float32', 

                             'prior_question_had_explanation': 'boolean',

                             })

train = train.drop(train[train['answered_correctly']==-1].index)
RFC = RandomForestClassifier(max_depth=10, random_state=0)

RFC.fit(train[['timestamp', 'content_id', 'content_type_id', 'task_container_id', 'prior_question_elapsed_time', 'prior_question_had_explanation']].fillna(0), train['answered_correctly'])
#Create the env

env = riiideducation.make_env()



#Create the iterator

iter_test = env.iter_test()



#Iter and predict

for (test_df, sample_prediction_df) in iter_test:

    test_df['answered_correctly'] = RFC.predict(test_df[['timestamp', 'content_id', 'content_type_id', 'task_container_id', 'prior_question_elapsed_time', 'prior_question_had_explanation']].fillna(0))

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])