# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', 

                       dtype={'row_id': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',

                              'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 'prior_question_elapsed_time': 'float32', 

                             'prior_question_had_explanation': 'boolean',

                             }

                      )

lectures_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/lectures.csv')

questions_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')
train_df[40:43]
train_df['user_answer'].unique()
questions_df['correct_answer'].unique()
lectures_df['tag'].unique()
lectures_df['part'].unique()
questions_df.loc[questions_df.duplicated('bundle_id', keep=False),:][:3]
train_df.iloc[1023702:1023702+5,:]
train_df.iloc[94636590:94636590+5,:]