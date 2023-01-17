import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
train_df = pd.read_csv("../input/nlp-getting-started/train.csv")
test_df = pd.read_csv("../input/nlp-getting-started/test.csv")
#データをのぞいてみよう
train_df.shape
train_df.isnull().sum()
#text, targetに欠損はなかった
train_df.head()
train_df["target"].value_counts()
#単語を抽出してみよう
count_vectorizer = CountVectorizer()
example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])
print(count_vectorizer.get_feature_names())
len(count_vectorizer.get_feature_names())
print(example_train_vectors.todense())
train_vectors = count_vectorizer.fit_transform(train_df['text'])
test_vectors = count_vectorizer.transform(test_df['text'])
rr = RidgeClassifier(alpha=1)
scores_rr = model_selection.cross_val_score(rr, train_vectors, train_df["target"], cv=5, scoring="f1")
scores_rr
np.average(scores_rr)
lr = LogisticRegression(C=1, max_iter=10000)
scores_lr = model_selection.cross_val_score(lr, train_vectors, train_df["target"], cv=5, scoring = "f1")
scores_lr
np.average(scores_lr)
lr.fit(train_vectors, train_df['target'])
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = lr.predict(test_vectors)
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)
NB = MultinomialNB()
NB.fit(train_vectors, train_df['target'])
scores_NB
np.average(scores_NB)
submission_NB = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
submission_NB["target"] = NB.predict(test_vectors)
submission_NB.head()
submission_NB.to_csv("submission_NB.csv", index=False)
