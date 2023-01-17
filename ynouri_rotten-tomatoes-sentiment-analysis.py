import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/train.tsv", sep="\t")
df_test = pd.read_csv("../input/test.tsv", sep="\t")
df.shape
def print_sentence(df, sentence_id=None):
    if not sentence_id:
        sentence_ids = df_test.SentenceId.unique()
        sentence_id = np.random.choice(sentence_ids)
    print("Sentence ID = {}".format(sentence_id))
    return df[df.SentenceId == sentence_id].iloc[:].Phrase

print_sentence(df_test)
dist = df.groupby(["Sentiment"]).size()
dist = dist / dist.sum()
fig, ax = plt.subplots(figsize=(12,8))
sns.barplot(dist.keys(), dist.values);
def generate_dummy_submission():
    df_submission = df_test.copy()
    n = df_submission.shape[0]
    df_submission["Sentiment"] = [2] * n
    df_submission = df_submission.loc[:, ["PhraseId", "Sentiment"]]
    df_submission.to_csv("submission.csv", index=False)
    
# generate_dummy_submission() 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
svc = LinearSVC(
    C=1.0,
    class_weight='balanced',
    dual=True,
    fit_intercept=True,
    intercept_scaling=1,
    loss='squared_hinge',
    max_iter=1000,
    multi_class='ovr',
    penalty='l2',
    random_state=0,
    tol=1e-05, 
    verbose=0
)

tfidf = CountVectorizer(
    input='content',
    encoding='utf-8',
    decode_error='strict',
    strip_accents=None,
    lowercase=True,
    preprocessor=None,
    tokenizer=None,
    stop_words=None,
    token_pattern=r"(?u)\b\w\w+\b",
    ngram_range=(1, 1),
    analyzer='word',
    max_df=1.0,
    min_df=1,
    max_features=None,
    vocabulary=None,
    binary=False,
    dtype=np.int64
)

pipeline = Pipeline([
    ('tfidf', tfidf),
    ('svc', svc),
])
skf = StratifiedKFold(n_splits=3)

X = df.Phrase
y = df.Sentiment

for train, test in skf.split(X, y):
    pipeline.fit(X[train], y[train])
    train_score = pipeline.score(X[train], y[train])
    test_score = pipeline.score(X[test], y[test])
    print("Train = {}, Test = {}".format(train_score, test_score))