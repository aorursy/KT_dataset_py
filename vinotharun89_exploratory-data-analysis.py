import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



p = sns.color_palette()



print('# File sizes')

for f in os.listdir('../input'):

    if 'zip' not in f:

        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
df_train = pd.read_csv('../input/train.csv')

df_train[1:10]
print('Total number of question pairs for training: {}'.format(len(df_train)))

print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean()*100, 2)))

qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())

print('Total number of questions in the training data: {}'.format(len(np.unique(qids))))

print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))



plt.hist(qids.value_counts(), bins=50)

plt.yscale('log', nonposy='clip')

plt.title('Log-Histogram of question appearance counts')

plt.xlabel('Number of occurences of question')

plt.ylabel('Number of questions')

print()
df_train[df_train["is_duplicate"] == 1].dtypes
from sklearn.metrics import log_loss



p = df_train['is_duplicate'].mean() # Our predicted probability

print('Predicted score:', log_loss(df_train['is_duplicate'], np.zeros_like(df_train['is_duplicate']) + p))



df_test = pd.read_csv('../input/test.csv')

sub = pd.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': p})

sub.to_csv('naive_submission.csv', index=False)

sub.head()