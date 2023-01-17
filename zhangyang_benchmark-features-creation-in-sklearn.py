# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn.feature_extraction.text

import sklearn.pipeline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv').fillna('')

test = pd.read_csv('../input/test.csv').fillna('')
questions = np.concatenate([

    train.question1, train.question2, test.question1, test.question2,

])
binary_count_vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=True).fit(questions)
tfidf_transformer = sklearn.pipeline.make_pipeline(

    sklearn.feature_extraction.text.CountVectorizer(binary=True),

    sklearn.feature_extraction.text.TfidfTransformer()

).fit(questions)
def compute_overlap_rate(df, transformer):

    q1_one_hot = transformer.transform(df['question1'])

    q2_one_hot = transformer.transform(df['question2'])



    q1_weights = q1_one_hot.sum(axis=1)

    q2_weights = q2_one_hot.sum(axis=1)

    q1_q2_overlap_weights = q1_one_hot.multiply(q2_one_hot).sum(axis=1)

    overlap_rates = q1_q2_overlap_weights / (q1_weights + q2_weights)

    return overlap_rates
def build_overlap_feature(df):

    df['overlap_rates_one_hot'] = compute_overlap_rate(df, binary_count_vectorizer)

    df['overlap_rates_tfidf'] = compute_overlap_rate(df, tfidf_transformer)

    return df
train = build_overlap_feature(train)

test = build_overlap_feature(test)
test = test.fillna(0)
train[['is_duplicate', 'overlap_rates_one_hot', 'overlap_rates_tfidf'

       ]].to_csv('train_features.csv', index=False)

test[['test_id', 'overlap_rates_one_hot', 'overlap_rates_tfidf']].to_csv(

    'test_features.csv', index=False)