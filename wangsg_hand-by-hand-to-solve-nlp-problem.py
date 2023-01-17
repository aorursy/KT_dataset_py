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
import re, os, time
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
def clean_text(comment_text):
    comment_list = []
    for text in comment_text:
        # 将单词转换为小写
        text = text.lower()
        # 删除非字母、数字字符
        text = re.sub(r"[^A-Za-z0-9(),!?@&$\'\`\"\_\n]", " ", text)
        text = re.sub(r"\n", " ", text)
        
        # 恢复常见的简写
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        
        # 恢复特殊符号的英文单词
        text = text.replace('&', ' and')
        text = text.replace('@', ' at')
        text = text.replace('$', ' dollar')
        
        comment_list.append(text)
    return comment_list

train["clean_comment_text"] = clean_text(train['comment_text'])
test['clean_comment_text'] = clean_text(test['comment_text'])
train[['comment_text','clean_comment_text']].head(5)
all_comment_list = list(train['clean_comment_text']) + list(test['clean_comment_text'])
text_vector = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode',token_pattern=r'\w{1,}',
                         max_features=5000, ngram_range=(1, 1), analyzer='word')
text_vector.fit(all_comment_list)
train_vec = text_vector.transform(train['clean_comment_text'])
test_vec = text_vector.transform(test['clean_comment_text'])
train_vec
x_train, x_valid, y_train, y_valid = train_test_split(train_vec, train[labels], test_size=0.1, random_state=2018)
x_test = test_vec
accuracy = []
for label in labels:
    clf = LogisticRegression(C=6)
    clf.fit(x_train, y_train[label])
    y_pre = clf.predict(x_valid)
    train_scores = clf.score(x_train, y_train[label])
    valid_scores = accuracy_score(y_pre, y_valid[label])
    print("{} train score is {}, valid score is {}".format(label, train_scores, valid_scores))
    accuracy.append(valid_scores)
    pred_proba = clf.predict_proba(x_test)[:, 1]
    sample_submission[label] = pred_proba
print("Total cv accuracy is {}".format(np.mean(accuracy)))
from datetime import datetime

def submission(submission):
    file_name = '{}.csv'.format(datetime.now().strftime("%Y-%m-%d_%H-%M"))
    submission.to_csv(file_name, index=False)
    print("write to {}".format(file_name))
submission(sample_submission)
