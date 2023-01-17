import pandas as pd
df = pd.read_csv("../input/amazonearphonesreviews/AllProductReviews.csv")
print(df.shape)
df.head()
y = df["ReviewStar"]
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_short
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_numeric, remove_stopwords
#import spacy

#nlp = spacy.load("en_core_web_sm")
reviews = df["ReviewBody"].values.tolist()
CUSTOM_FILTERS = [lambda x: x.lower(), strip_punctuation,
                  strip_numeric]

texts = [preprocess_string(review, CUSTOM_FILTERS) for review in reviews]
import gensim.downloader as api
glove = api.load('glove-wiki-gigaword-300')
import numpy as np
X = []

for text in texts:
    vecs = []
    for word in text:
        try:
            vecs.append(glove[word])
        except:
            continue
    X.append(np.mean(vecs, axis = 0))
len(X)
X[0].shape[0]
missing_indices = []

for i in range(len(X)):
    if X[i].shape != (300,):
        missing_indices.append(i)
print(len(missing_indices))
for idx in missing_indices:
    X[idx] = np.zeros(shape = (300,))
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
ss = sia.polarity_scores(reviews[0])
print(ss)
for i in range(len(X)):
    ss = sia.polarity_scores(reviews[i])
    ss = list(ss.values())
    X[i] = np.concatenate((X[i], np.array(ss)))
y.value_counts()
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, random_state = 34, stratify = y)
np.array(X).shape
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipelines = []
pipelines.append(('Logistic Regression', Pipeline([('imp', SimpleImputer()),
                                                   ('sc', StandardScaler()),
                                                   ('LR', SGDClassifier(loss = "log"))] )))
pipelines.append(('Support Vector Machine', Pipeline([('imp', SimpleImputer()), 
                                                      ('sc', StandardScaler()),
                                                      ('SV', SGDClassifier())] )))
pipelines.append(('Naive Bayes', Pipeline([('imp', SimpleImputer()), 
                                           ('sc', StandardScaler()),
                                           ('NB', GaussianNB())] )))
pipelines.append(('Decision Tree', Pipeline([('imp', SimpleImputer()), 
                                             ('sc', StandardScaler()),
                                             ('Tree', DecisionTreeClassifier())] )))

results = []
names = []

for name, model in pipelines:
    kfold = RepeatedStratifiedKFold(n_splits = 3, random_state = 34)
    cvresults = cross_val_score(model, X_train, y_train, cv = kfold, scoring = "accuracy")
    results.append(cvresults)
    names.append(name)
    msg ="%s: %f (%f)" % (name, cvresults.mean(), cvresults.std())
    print(msg)
%matplotlib inline
import matplotlib.pyplot as plt

plt.boxplot(results, labels = names)
plt.xticks(rotation = 'vertical')
plt.tight_layout()
plt.show()
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

pipelines = []
pipelines.append(('Support Vector Machine', Pipeline([('imp', SimpleImputer()),
                                                   ('sc', StandardScaler()),
                                                   ('SVM', SVC())] )))
pipelines.append(('Random Forest', Pipeline([('imp', SimpleImputer()), 
                                                      ('sc', StandardScaler()),
                                                      ('RF', RandomForestClassifier())] )))
pipelines.append(('KNN', Pipeline([('imp', SimpleImputer()), 
                                           ('sc', StandardScaler()),
                                           ('KNN', KNeighborsClassifier())] )))

results = []
names = []

for name, model in pipelines:
    kfold = RepeatedStratifiedKFold(n_splits = 3, random_state = 34)
    cvresults = cross_val_score(model, X_train, y_train, cv = kfold, scoring = "accuracy")
    results.append(cvresults)
    names.append(name)
    msg ="%s: %f (%f)" % (name, cvresults.mean(), cvresults.std())
    print(msg)
%matplotlib inline
import matplotlib.pyplot as plt

plt.boxplot(results, labels = names)
plt.xticks(rotation = 'vertical')
plt.tight_layout()
plt.show()
model = Pipeline([('imp', SimpleImputer()),
                                                   ('sc', StandardScaler()),
                                                   ('SVM', SVC())] )
model.get_params()
from scipy.stats import uniform

params = {'SVM__C' : uniform(1e-3, 2),
          'SVM__gamma' : uniform(1e-6, 1)}
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(model, params, n_iter = 70, cv = 5, random_state = 6)
random_search.fit(X_train, y_train)

print("Best Score: ", random_search.best_score_)
print("Best Parameters: ", random_search.best_params_)
model.fit(X_train, y_train)
from sklearn.metrics import classification_report

y_pred = model.predict(X_validation)

print(classification_report(y_validation, y_pred))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_validation, y_pred))
