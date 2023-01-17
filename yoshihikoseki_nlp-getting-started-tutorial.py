import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing, kernel_ridge
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_df[train_df["target"] == 0]["text"].values[1]
train_df[train_df["target"] == 1]["text"].values[1]
count_vectorizer = feature_extraction.text.CountVectorizer()



## let's get counts for the first 5 tweets in the data

example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])
## we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)

print(example_train_vectors[0].todense().shape)

print(example_train_vectors[0].todense())
train_vectors = count_vectorizer.fit_transform(train_df["text"])



## note that we're NOT using .fit_transform() here. Using just .transform() makes sure

# that the tokens in the train vectors are the only ones mapped to the test vectors - 

# i.e. that the train and test vectors use the same set of tokens.

test_vectors = count_vectorizer.transform(test_df["text"])
## Our vectors are really big, so we want to push our model's weights

## toward 0 without completely discounting different words - ridge regression 

## is a good way to do this.

#clf = linear_model.RidgeClassifier()

#clf = kernel_ridge.KernelRidge()

#clf = linear_model.PassiveAggressiveClassifier()

#clf = linear_model.SGDClassifier()

#clf = linear_model.RidgeClassifierCV()

#clf = linear_model.locally_linear_embedding()

#clf = linear_model.LinearRegression()

clf = linear_model.LinearRegression()
#scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

#scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

#print(scores)

#scores
clf.fit(train_vectors, train_df["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
#sample_submission["target"] = clf.predict(test_vectors)



L = clf.predict(test_vectors)

ary = [0]*len(L)



for index, item in enumerate(L):

    if item >= 1.0:

        ary[index] = int(L[index])

    else:

        ary[index] = int(L[index])



sample_submission["target"] = ary

#print(sample_submission["target"])

#L = sample_submission["target"]



#先頭行 なし＝初期値５行、引数必要な分だけ行数をだす。

sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)
