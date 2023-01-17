from sklearn.datasets import load_files

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression



import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
reviews_train = load_files("data/train/")

text_train, y_train = reviews_train.data, reviews_train.target



reviews_test = load_files("data/test/")

text_test, y_test = reviews_test.data, reviews_test.target
print("Number of documents in train data: {}".format(len(text_train)))

print("Class values: {}".format(np.unique(y_train)))

print("Samples per class (train): {}".format(np.bincount(y_train)))
# Positive example

i=600

print(y_train[i]," : ",text_train[i])
# Negative example

i=1800

print(y_train[i]," : ",text_train[i])
vect = TfidfVectorizer(min_df=5, max_df=0.6, ngram_range=(1, 2))

X_train = vect.fit(text_train).transform(text_train)

X_test = vect.transform(text_test)

feature_names = vect.get_feature_names()
print("X_train:\n{}".format(X_train.shape))

print("X_test: \n{}".format(X_test.shape))

print("Number of features: {}".format(len(feature_names)))
feature_names = vect.get_feature_names()

feature_names=np.array(feature_names)

feature_names[1100:1110]
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}



# Cross validation : CV = 5 folds

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)

grid.fit(X_train, y_train)



print("Best cross-validation score: {:.2f}".format(grid.best_score_))

print("Best parameters: ", grid.best_params_)

print("Best estimator: ", grid.best_estimator_)
sortindx = np.argsort(grid.best_estimator_.coef_[0])
# Top 10 Words with the most positive coefficients

plt.figure(figsize=(10,10))

plt.bar(feature_names[sortindx[-10:]], grid.best_estimator_.coef_[0][sortindx[-10:]])
# Top 10 Words with the most negative coefficients

plt.figure(figsize=(10,10))

plt.bar(feature_names[sortindx[:10]], grid.best_estimator_.coef_[0][sortindx[:10]])