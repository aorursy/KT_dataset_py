import re



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')
train.head()
test.head()
# remove url

# keep "http" part, might be useful info



#train["text"] = train["text"].apply(lambda x: re.sub(r"http.*", "http", x))

#test["text"] = test["text"].apply(lambda x: re.sub(r"http.*", "http", x))
X_train, y_train = train["text"], train["target"]

X_test = test["text"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



gnb_f1s = []

gnb_f1s_norm = []

bnb_f1s = []

bnb_f1s_norm = []

mnb_f1s = []

mnb_f1s_norm = []



for train_idx, val_idx in skf.split(X_train, y_train):

    X_train_cv, X_val = X_train[train_idx], X_train[val_idx]

    y_train_cv, y_val = y_train[train_idx], y_train[val_idx]

    

    vectorizer = CountVectorizer(stop_words='english', min_df=1)

    normalizer = Normalizer(norm="l2")

    

    X_train_cv = vectorizer.fit_transform(X_train_cv).toarray()

    X_val = vectorizer.transform(X_val).toarray()

    

    X_train_cv_norm = normalizer.transform(X_train_cv)

    X_val_norm = normalizer.transform(X_val)

    

    gnb = GaussianNB()

    bnb = BernoulliNB(binarize=0.0) # binalize inputs

    mnb = MultinomialNB()

    

    gnb.fit(X_train_cv, y_train_cv)

    gnb_f1s.append(f1_score(y_val, gnb.predict(X_val)))

    gnb.fit(X_train_cv_norm, y_train_cv)

    gnb_f1s_norm.append(f1_score(y_val, gnb.predict(X_val_norm)))

    

    bnb.fit(X_train_cv, y_train_cv)

    bnb_f1s.append(f1_score(y_val, bnb.predict(X_val)))

    bnb.fit(X_train_cv_norm, y_train_cv)

    bnb_f1s_norm.append(f1_score(y_val, bnb.predict(X_val_norm)))

    

    mnb.fit(X_train_cv, y_train_cv)

    mnb_f1s.append(f1_score(y_val, mnb.predict(X_val)))

    mnb.fit(X_train_cv_norm, y_train_cv)

    mnb_f1s_norm.append(f1_score(y_val, mnb.predict(X_val_norm)))
print(f"5-fold CV f1 score")

print(f"Gaussian NB: {np.mean(gnb_f1s)}")

print(f"Gaussian NB with l2 normalization: {np.mean(gnb_f1s_norm)}")

print(f"Bernoulli NB: {np.mean(bnb_f1s)}")

print(f"Bernoulli NB with l2 normalization: {np.mean(bnb_f1s_norm)}")

print(f"Multinomial NB: {np.mean(mnb_f1s)}")

print(f"Multinomial NB with l2 normalization: {np.mean(mnb_f1s_norm)}")
# submit with the best model: Multinomial NB

spsbm = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")



X_train = vectorizer.fit_transform(X_train).toarray()

X_test = vectorizer.transform(X_test).toarray()



mnb = MultinomialNB()

mnb.fit(X_train, y_train)



spsbm["target"] = mnb.predict(X_test)

spsbm.to_csv("submission.csv",index=False)