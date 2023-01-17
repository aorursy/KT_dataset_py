import numpy as np
import pandas as pd

# Preprocessing
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Feature extraction, model evaluation and hyperparemter optimization
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("../input/spam.csv", encoding = "latin-1")
df.head()
print(sum(df.iloc[:, 2].notna()))
df.iloc[:, 2].unique()
print(sum(df.iloc[:, 3].notna()))
df.iloc[:, 3].unique()
print(sum(df.iloc[:, 4].notna()))
df.iloc[:, 4].unique()
df = df[["v1", "v2"]]
df.head()
df.columns = ["class", "message"]
df.head()
# Download the last available version of the stopwords
nltk.download("stopwords")
def clean_message(message):
    """
    Receives a raw message and clean it using the following steps:
    1. Remove all non-words in the message
    2. Transform the message in lower case
    3. Remove all stop words
    4. Perform stemming

    Args:
        message: the raw message
    Returns:
        a clean message using the mentioned steps above.
    """
    
    message = re.sub("[^A-Za-z]", " ", message)
    message = message.lower()
    message = message.split()
    stemmer = PorterStemmer()
    message = [stemmer.stem(word) for word in message if word not in set(stopwords.words("english"))]
    message = " ".join(message)
    return message
# Testing how our function works
message = df.message[0]
print(message)

message = clean_message(message)
print(message)
corpus = []
for i in range(0, len(df)):
    message = clean_message(df.message[i])
    corpus.append(message)
corpus[:5]
print(round(sum(df["class"] == "ham") / len(df) * 100, 2))
print(round(sum(df["class"] == "spam") / len(df) * 100, 2))
count_vectorizer = CountVectorizer()
features = count_vectorizer.fit_transform(corpus).toarray()
features.shape
labels = df["class"].values
labels[:5]
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
    test_size = 0.20, stratify = labels, random_state = 42)
print(count_vectorizer.get_feature_names()[:10])
print(count_vectorizer.get_feature_names()[-10:])
nb_classifier = MultinomialNB()
k_fold = StratifiedKFold(n_splits = 10)
scores = cross_val_score(nb_classifier, features_train, labels_train, cv = k_fold)
print("mean:" , scores.mean(), "std:", scores.std())
nb_classifier.fit(features_train, labels_train)
labels_predicted = nb_classifier.predict(features_test)
accuracy_score(labels_test, labels_predicted)
confusion_matrix(labels_test, labels_predicted, labels = ["ham", "spam"])
kfold = StratifiedKFold(n_splits = 10)
parameters = {"alpha": np.arange(0, 1, 0.1)}
searcher = GridSearchCV(MultinomialNB(), param_grid = parameters, cv = kfold)
searcher.fit(features_train, labels_train)
best_multinomial_nb = searcher.best_estimator_

print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)
print("Test accuracy of best grid search hypers:", searcher.score(features_test, labels_test))
labels_predicted = best_multinomial_nb.predict(features_test)
print("Accuracy Score:", accuracy_score(labels_test, labels_predicted))
print(classification_report(labels_test, labels_predicted))
%%time

models = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]
for model in models:
    model.fit(features_train, labels_train)

    scores = cross_val_score(model, features_train, labels_train, cv = kfold)
    print(type(model))
    print("Mean score:" , scores.mean(), "Std:", scores.std())
    print()

    predictions = model.predict(features_test)
    accuracy_score(labels_test, predictions)

    labels_predicted = model.predict(features_test)
    print("Test Accuracy Score:", accuracy_score(labels_test, labels_predicted))
    print(classification_report(labels_test, labels_predicted))