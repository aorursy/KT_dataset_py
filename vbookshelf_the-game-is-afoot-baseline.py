import pandas as pd

import numpy as np

import os
os.listdir('../input/contradictory-my-dear-watson')
# Load the train info



path = '../input/contradictory-my-dear-watson/train.csv'



df_train = pd.read_csv(path)



df_train.head()
# Check which is the majority class.



df_train['label'].value_counts()
# Load the sample submission.



path = '../input/contradictory-my-dear-watson/sample_submission.csv'



df_sample = pd.read_csv(path)



print(df_sample.shape)



df_sample.head()
# Assign the majority class as the target value for all rows.



df_sample['prediction'] = 0



df_sample.head()
# Create a submission csv file

df_sample.to_csv('submission.csv', index=False)
!ls