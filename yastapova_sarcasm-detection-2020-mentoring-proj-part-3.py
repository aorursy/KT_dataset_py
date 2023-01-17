# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, cross_validate

from sklearn.metrics import accuracy_score, confusion_matrix



from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.naive_bayes import MultinomialNB



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/sarcasm-detection-2020-mentoring-proj-part-2/sarcasm_train_split.csv")

test_data = pd.read_csv("/kaggle/input/sarcasm-detection-2020-mentoring-proj-part-2/sarcasm_test_split.csv")

train_data.head()
train_comments = train_data["comment"]

train_comments.head()
vect = CountVectorizer(max_features=20000)

train_bow = vect.fit_transform(train_comments)

train_bow.shape
log_reg_model = LogisticRegression(random_state=42)

cross_validate(log_reg_model, train_bow, train_data["label"], cv=3, scoring="accuracy", n_jobs=-1)
# make sure stopwords are all lowercase because

# the vectorizer makes everything lowercase by default

stopwords = ["the", "a", "an", "she", "he", "i", "you", "me", "they",

             "her", "his", "your", "their", "my", "we", "our", "ours",

             "hers", "yours", "it", "its", "him", "them", "theirs",

             "this", "that", "is", "was", "are", "were", "am", "or",

             "as", "of", "at", "by", "for", "with", "to", "from", "in",

             "m", "s", "ve", "d", "ll", "o", "re"]



vect = CountVectorizer(strip_accents='unicode', stop_words=stopwords, min_df=0.0001, max_df=0.70)

train_bow = vect.fit_transform(train_comments)

train_bow.shape
tf_trans = TfidfTransformer()

train_tf = tf_trans.fit_transform(train_bow)

train_tf.shape
# recreate the vectorizer and transformer so they are not fit yet

vect = CountVectorizer(strip_accents='unicode', stop_words=stopwords, min_df=0.0001, max_df=0.70)

tf_trans = TfidfTransformer()



# create the model

log_reg_model = LogisticRegression(random_state=42, penalty="elasticnet", solver="saga")



pipeline = Pipeline([

    ('vect', vect),

    ('tftrans', tf_trans),

    ('model', log_reg_model)

])
param_grid = {

    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],

    'vect__max_features': (5000, 15000, 30000),

    'model__l1_ratio': (0.0, 0.25, 0.50, 0.75, 1.0)

}



grid_logreg = GridSearchCV(pipeline, param_grid, scoring="accuracy", cv=3, n_jobs=-1)

grid_logreg.fit(train_comments, train_data["label"])
print(grid_logreg.best_score_)

for param_name in sorted(param_grid.keys()):

    print("%s: %r" % (param_name, grid_logreg.best_params_[param_name]))
# recreate the vectorizer and transformer so they are not fit yet

vect = CountVectorizer(strip_accents='unicode', stop_words=stopwords, min_df=0.0001, max_df=0.70)

tf_trans = TfidfTransformer()



# create the NB model

nb_model = MultinomialNB()



pipeline = Pipeline([

    ('vect', vect),

    ('tftrans', tf_trans),

    ('model', nb_model)

])
param_grid = {

    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],

    'vect__max_features': (5000, 15000, 30000)

}



grid_nb = GridSearchCV(pipeline, param_grid, scoring="accuracy", cv=3, n_jobs=-1)

grid_nb.fit(train_comments, train_data["label"])
print(grid_nb.best_score_)

for param_name in sorted(param_grid.keys()):

    print("%s: %r" % (param_name, grid_nb.best_params_[param_name]))
# recreate the vectorizer and transformer so they are not fit yet

vect = CountVectorizer(strip_accents='unicode', stop_words=stopwords, min_df=0.0001, max_df=0.70)

tf_trans = TfidfTransformer()



# create the SVM model

svm_model = SGDClassifier(penalty="elasticnet", random_state=42, n_jobs=-1)



pipeline = Pipeline([

    ('vect', vect),

    ('tftrans', tf_trans),

    ('model', svm_model)

])
param_grid = {

    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],

    'vect__max_features': (5000, 15000, 30000),

    'model__l1_ratio': (0.0, 0.15, 0.40, 0.60, 0.85, 1.0)

}



grid_svm = GridSearchCV(pipeline, param_grid, scoring="accuracy", cv=3, n_jobs=-1)

grid_svm.fit(train_comments, train_data["label"])
print(grid_svm.best_score_)

for param_name in sorted(param_grid.keys()):

    print("%s: %r" % (param_name, grid_svm.best_params_[param_name]))
# recreate the vectorizer and transformer so they are not fit yet

vect = CountVectorizer(strip_accents='unicode', stop_words=stopwords, min_df=0.0001, max_df=0.70, max_features=15000, ngram_range=(1, 3))

tf_trans = TfidfTransformer()



# create the NB model

nb_model = MultinomialNB()



pipeline = Pipeline([

    ('vect', vect),

    ('tftrans', tf_trans),

    ('model', nb_model)

])



pipeline.fit(train_comments, train_data["label"])
test_comments = test_data["comment"]

preds = pipeline.predict(test_comments)

preds.shape
acc = accuracy_score(test_data["label"], preds)

acc
confusion_matrix(test_data["label"], preds) / test_data.shape[0]