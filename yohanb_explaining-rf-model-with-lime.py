import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



from xgboost import XGBClassifier

%matplotlib inline



df = pd.read_csv('../input/YoutubeSpamMergeddata.csv', encoding = "ISO-8859-1")

data = df[['CONTENT', 'CLASS']]

class_names = ['not_spam', 'spam']



X_train, X_test, y_train, y_test = train_test_split(data.CONTENT, data.CLASS, 

                                                    stratify=data.CLASS, 

                                                    test_size=0.25, 

                                                    random_state=42)



vectorizer = TfidfVectorizer(lowercase=False)

train_vectors = vectorizer.fit_transform(X_train)

test_vectors = vectorizer.transform(X_test)



rf = RandomForestClassifier(n_estimators=500)

rf.fit(train_vectors, y_train)



pred = rf.predict(test_vectors)

accuracy_score(y_test, pred)
from lime import lime_text

from sklearn.pipeline import make_pipeline

c = make_pipeline(vectorizer, rf)
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=class_names)
idx = 2

exp = explainer.explain_instance(X_test.iloc[idx], c.predict_proba, num_features=10)

print('Document id: %d' % idx)

#print('Text: %s' % X_test.iloc[idx])

print('Probability(spam) =', c.predict_proba([X_test.iloc[idx]])[0,1])

print('True class: %s' % class_names[y_test.iloc[idx]])



fig = exp.as_pyplot_figure()

exp.show_in_notebook(text=False)
idx = 3

exp = explainer.explain_instance(X_test.iloc[idx], c.predict_proba, num_features=10)

print('Document id: %d' % idx)

#print('Text: %s' % X_test.iloc[idx])

print('Probability(spam) =', c.predict_proba([X_test.iloc[idx]])[0,1])

print('True class: %s' % class_names[y_test.iloc[idx]])



fig = exp.as_pyplot_figure()

exp.show_in_notebook(text=False)
idx = 110

exp = explainer.explain_instance(X_test.iloc[idx], c.predict_proba, num_features=10)

print('Document id: %d' % idx)

#print('Text: %s' % X_test.iloc[idx])

print('Probability(spam) =', c.predict_proba([X_test.iloc[idx]])[0,1])

print('True class: %s' % class_names[y_test.iloc[idx]])



fig = exp.as_pyplot_figure()

exp.show_in_notebook(text=False)
idx = 18

exp = explainer.explain_instance(X_test.iloc[idx], c.predict_proba, num_features=10)

print('Document id: %d' % idx)

#print('Text: %s' % X_test.iloc[idx])

print('Probability(spam) =', c.predict_proba([X_test.iloc[idx]])[0,1])

print('True class: %s' % class_names[y_test.iloc[idx]])



fig = exp.as_pyplot_figure()

exp.show_in_notebook(text=False)