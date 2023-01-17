# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv').fillna(' ')
test_data = pd.read_csv('../input/test.csv').fillna(' ')
full_data = [train_data, test_data]

train_data.head(10)
test_data.head(10)
import re
words = set()
word_re = re.compile('\w+\'*\w*')
classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
word_infos = {}
for row in train_data.as_matrix():
    cid = row[0]
    comment = row[1]
    cwords = re.findall(word_re, comment)
    if not cwords is None:
        char_set = set()
        for l in range(2, 7):
            for i in range(len(comment)-l+1):
                word = comment[i:i+l-1]
                char_set.add(word)
            for word in cwords:
                char_set.add(word)
            cwords = list(char_set)
        for word in cwords:
            word = word.lower()
            if not word in word_infos:
                word_infos[word] = [0, 0, 0, 0, 0, 0, 0]
            word_infos[word][0] += 1
            for i in range(6):
                word_infos[word][i+1] += row[i+2]
print(len(word_infos))
cclass_infos = {}
cclass_infos['word'] = []
cclass_infos['count'] = []
for cclass in classes:
    cclass_infos[cclass] = []
for word in word_infos:
    cclass_infos['word'].append(word)
    cclass_infos['count'].append(word_infos[word][0])
    i = 1
    for cclass in classes:
        cclass_infos[cclass].append(word_infos[word][i])
        i += 1
word_info_df = pd.DataFrame(cclass_infos)
word_info_df.head(10)
for cclass in classes:
    word_info_df[cclass+'_rate'] = word_info_df[cclass] / word_info_df['count']
word_info_df.head(20)
word_rate_info = word_info_df.set_index('word').T.to_dict('dict')
for cclass in classes:
    test_data[cclass] = 0
test_data.head(10)
uncommon_rate = {}
uncommon_number = 0
for cclass in classes:
    uncommon_rate[cclass] = 0
for word in word_rate_info:
    if word_rate_info[word]['count'] == 1:
        uncommon_number += 1
        for cclass in classes:
            uncommon_rate[cclass] += word_rate_info[word][cclass+'_rate']
for cclass in classes:
    uncommon_rate[cclass] /= uncommon_number
print(uncommon_rate)
limited_words_info = {}
for word in word_rate_info:
    if word_rate_info[word]['count'] < 3 :
        continue
    max_rate = 0
    for cclass in classes:
        max_rate = max(max_rate, word_rate_info[word][cclass+'_rate'])
    if max_rate < 0.1:
        continue
    limited_words_info[word] = word_rate_info[word]
word_rate_info = limited_words_info
print(len(word_rate_info))
submission = {}
submission['id'] = []
for cclass in classes:
    submission[cclass] = []
for i in range(test_data.shape[0]):
    id_text = test_data['id'][i]
    comment = test_data['comment_text'][i]
    cwords = re.findall(word_re, comment)
    max_rate = {}
    for cclass in classes:
        max_rate[cclass] = 0
    all_uncommon = True
    if not cwords is None:
        char_set = set()
        for l in range(2, 7):
            for i in range(len(comment)-l+1):
                word = comment[i:i+l-1]
                char_set.add(word)
            for word in cwords:
                char_set.add(word)
            cwords = list(char_set)
        for word in cwords:
            all_uncommon = False
            if word in word_rate_info:
                for cclass in classes:
                    max_rate[cclass] = max(max_rate[cclass], word_rate_info[word][cclass+'_rate'])
    if all_uncommon:
        for cclass in classes:
            max_rate[cclass] = uncommon_rate[cclass]
    submission['id'].append(id_text)
    for cclass in classes:
        submission[cclass].append(max_rate[cclass])
import math
for i in range(len(submission['id'])):
    for cclass in classes:
        val = submission[cclass][i];
        if val >= 0.5:
            submission[cclass][i] = math.sqrt(val)
submission_df = pd.DataFrame(submission)
submission_df.to_csv('submission.csv', index=False)
submission_df.head()