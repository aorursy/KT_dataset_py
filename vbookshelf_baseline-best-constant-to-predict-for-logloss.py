import pandas as pd
import os
os.listdir('../input')
# read the train set into a dataframe
df_train = pd.read_csv('../input/train.csv')

df_train.head()
df_train['event'].value_counts()
total = len(df_train)

freq_A = 2848809/total
freq_B = 130597/total
freq_C = 1652686/total
freq_D = 235329/total
# delete df_train to free up memory space
del df_train

# read the test set into a dataframe
df_test = pd.read_csv('../input/test.csv')

# create new columns in the test dataframe
df_test['A'] = freq_A
df_test['B'] = freq_B
df_test['C'] = freq_C
df_test['D'] = freq_D

df_test.head()
# select the columns that will be part of the submission 
submission = df_test[['id', 'A', 'B', 'C', 'D']]

# save the submission dataframe as a csv file
submission.to_csv('submission.csv', index=False, 
                  columns=['id', 'A', 'B', 'C', 'D'])
submission.head()
