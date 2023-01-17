import numpy as np

import pandas as pd

from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df[train_df["target"] == 0]["text"].values[1]

train_df[train_df["target"] == 1]["text"].values[1]
count_v = feature_extraction.text.CountVectorizer()

example_train_vectors = count_v.fit_transform(train_df["text"][0:5])

train_v = count_v.fit_transform(train_df["text"])

test_v = count_v.transform(test_df["text"])
clf = linear_model.RidgeClassifier()

scores = model_selection.cross_val_score(clf, train_v, train_df["target"], cv=3, scoring="f1")

scores
clf.fit(train_v, train_df["target"])

sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission["target"] = clf.predict(test_v)

sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)