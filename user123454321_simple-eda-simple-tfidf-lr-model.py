import numpy as np 

import pandas as pd 

import seaborn as sns

import os

import matplotlib.pyplot as plt

import random

from sklearn import preprocessing

from sklearn.feature_extraction import text

from sklearn import linear_model

from sklearn import metrics
input_path = '/kaggle/input/nlp-getting-started/'
train_df = pd.read_csv(os.path.join(input_path, 'train.csv'))

test_df = pd.read_csv(os.path.join(input_path, 'test.csv'))

submission_df = pd.read_csv(os.path.join(input_path, 'sample_submission.csv'))
train_df.head()
train_df.target.value_counts()
most_frequent_keywords = train_df.keyword.value_counts().head(10).keys()
train_df_selected = train_df[train_df.keyword.isin(list(most_frequent_keywords))]

sns.violinplot(data=train_df_selected, x='keyword', y='target', )

plt.xticks(rotation=45)
most_frequent_locations = train_df.location.value_counts().head(10).keys()
train_df_selected = train_df[train_df.location.isin(list(most_frequent_locations))]

sns.violinplot(data=train_df_selected, x='location', y='target', )

plt.xticks(rotation=45)
new_train_df = train_df.copy()

new_train_df['location_null'] = train_df.location.isnull()

sns.violinplot(data=new_train_df, x='location_null', y='target', )

plt.xticks(rotation=45)
new_train_df = train_df.copy()

new_train_df['keyword_null'] = train_df.keyword.isnull()

sns.violinplot(data=new_train_df, x='keyword_null', y='target', )

plt.xticks(rotation=45)
len(new_train_df[(new_train_df.keyword_null == True) & (new_train_df.target == 1)])
len(new_train_df[(new_train_df.keyword_null == False) & (new_train_df.target == 0)])
len(train_df)
new_train_df = train_df.iloc[:6000]

new_val_df = train_df.iloc[6000:]
tfidf_extractor = text.TfidfVectorizer()
tfidf_extractor.fit(new_train_df.text)
train_tfidf_features = tfidf_extractor.transform(new_train_df.text).todense()

val_tfidf_features = tfidf_extractor.transform(new_val_df.text).todense()

test_tfidf_features = tfidf_extractor.transform(test_df.text).todense()
train_other_features = new_train_df.drop(columns=['id', 'text', 'target']).fillna('')

val_other_features = new_val_df.drop(columns=['id', 'text', 'target']).fillna('')

test_other_features = test_df.drop(columns=['id', 'text']).fillna('')
dummy_train = pd.get_dummies(train_other_features)

dummy_val = pd.get_dummies(val_other_features)

dummy_test = pd.get_dummies(test_other_features)
dummy_val = dummy_val.reindex(columns = dummy_train.columns, fill_value=0)

dummy_test = dummy_test.reindex(columns = dummy_train.columns, fill_value=0)
train_features = np.concatenate([train_tfidf_features, dummy_train.values], 1)

val_features = np.concatenate([val_tfidf_features, dummy_val.values], 1)

test_features = np.concatenate([test_tfidf_features, dummy_test.values], 1)
model = linear_model.LogisticRegression()
model.fit(train_features, new_train_df.target.values)
val_preds = model.predict(val_features)
metrics.f1_score(val_preds, new_val_df.target.values)
test_preds = model.predict(test_features)
submission_df['target'] = test_preds
submission_df.to_csv('submission.csv', index=False)
submission_df