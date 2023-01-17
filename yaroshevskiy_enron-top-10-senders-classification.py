import numpy as np

import pandas as pd

import re

import random

import email

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn import metrics 

from sklearn.feature_selection import SelectFromModel

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.cross_validation import cross_val_score

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



from sklearn.cross_validation import train_test_split

from sklearn.decomposition import TruncatedSVD



from scipy.sparse import coo_matrix, hstack
enron_data = pd.read_csv("../input/emails.csv", header=0, quoting=2)
enron_data.head()
# filtering only those that contain 'sent' in file name (f.e _sent_mail, sent_mail, sent etc) 



enron_sent = enron_data[enron_data["file"].str.contains('sent').tolist()]
# extracting senders (there might me cases like "orgname/sender" but so far as we need only top 10 senders we are ok)



enron_sent = enron_sent.assign(sender=enron_sent["file"].map(lambda x: re.search("(.*)/.*sent", x).group(1)).values)

enron_sent.drop("file", axis=1, inplace=True)

enron_sent["sender"].value_counts().head(10)
# mapping top senders' names to use later as label series

# we work only with top 10 senders



top_senders = enron_sent["sender"].value_counts().head(10).index.values

mapping = dict(zip(top_senders, range(10)))

print(mapping)
# info



print(enron_sent.shape)

print(enron_sent[enron_sent.sender.isin(top_senders)].shape)



enron_sent = enron_sent[enron_sent.sender.isin(top_senders)]
# now let's take a look at random email



print(enron_sent.iloc[random.randint(0, enron_sent.shape[0]), 0])
# I use default email library just for simplicity. For real product I would use more complicated parsing tools or write my own

# We extract email artificials and content from raw text



def email_from_string(raw_email):

    msg = email.message_from_string(raw_email)

    

    content = []

    for part in msg.walk():

        if part.get_content_type() == 'text/plain':

            content.append(part.get_payload())

            

    result = {}

    for key in msg.keys(): 

        result[key] = msg[key]

    result["content"] = ''.join(content)

    

    return result
enron_parsed = pd.DataFrame(list(map(email_from_string, enron_sent.message)))

enron_parsed.head(1)
# cc and bcc stand for carbon copy and blind carbon copy and that may be useful for classification

# Also we might use "To" or any other metadata but I believe the idea of this work is to use simply "content" + "subject" 



enron_parsed.info()
#here we do simply two things: 1 remove numbers and 2 remove stowords using nltk stopwords corpus



def content_to_wordlist( content, remove_stopwords=False ):

    content = re.sub("[^a-zA-Z]"," ", content)

    words = content.lower().split()

    

    if remove_stopwords:

        stops = set(stopwords.words("english"))

        words = [w for w in words if not w in stops]



    return ' '.join(words)
# enron_parsed['To'] = enron_parsed['To'].astype(str) # in case we want to use 'To' as information

data = pd.DataFrame(list(map(content_to_wordlist, 

                          enron_parsed[['Subject', 'content']].apply(lambda x: ' '.join(x), axis=1))), 

                          columns = ["content"])
data = data.assign(sender=enron_sent["sender"].values)

data = data.replace({'sender': mapping})

data.head()
# now we split data for training and test sets



data_train, data_test, y_train, y_test = train_test_split(data.content.values, data.sender.values, test_size=0.25)
# 72k features!

X_train.shape
# lets vectorize our content using default params



vect = TfidfVectorizer(sublinear_tf=True, use_idf=True)

X_train = vect.fit_transform(data_train)

X_test = vect.transform(data_test)
# let's try to use simple feature selection using l1 regularization and default threshold



clf = LogisticRegression(penalty='l1')

clf.fit(X_train, y_train)



model = SelectFromModel(clf, prefit=True)

X_train_new = model.transform(X_train)



X_train_new.shape
# non linear methods? I skip GradientBoostingClassifier for now because of my laptop low productivity :(

# we will use cross validation with 3 folds for estimation



for classifier in [LinearSVC, LogisticRegression, SGDClassifier, RandomForestClassifier]:

    print(cross_val_score(classifier(), X_train_new, y_train, cv=3).mean())
clf = LogisticRegression(C=0.15, penalty='l1')

clf.fit(X_train, y_train)



n_comp = np.sum(np.abs(clf.coef_) > 1e-4)

print(n_comp)



tsvd = TruncatedSVD(n_components = n_comp)

X_train_pca = tsvd.fit_transform(X_train)



for classifier in [LinearSVC, LogisticRegression, SGDClassifier, RandomForestClassifier]:

    print(cross_val_score(classifier(), X_train_pca, y_train, cv=3).mean())
# let's fit min number of components for previous model



scores = []

for n_components in range(10, 100, 10):

    tsvd = TruncatedSVD(n_components = n_components)

    X_train_pca = tsvd.fit_transform(X_train)



    score = cross_val_score(LinearSVC(), X_train_pca, y_train, cv=3).mean()

    scores.append(score)

    print(score)
import matplotlib.pyplot as plt

plt.plot(range(10, 100, 10), scores)

plt.show()
#let's fit parameters for linear svm



tsvd = TruncatedSVD(n_components = 120)

X_train_pca = tsvd.fit_transform(X_train)



parameters = {'C':[0.1, 0.3, 0.5, 1, 3, 5, 10, 30]}

clf = GridSearchCV(LinearSVC(), parameters)

clf.fit(X_train_pca, y_train)



clf.grid_scores_
# finally let's train the model on the test data and do some model evaluation



tsvd = TruncatedSVD(n_components = 120)

X_train_pca = tsvd.fit_transform(X_train)

X_test_pca = tsvd.transform(X_test)



clf = LinearSVC()

clf.fit(X_train_pca, y_train)



print(metrics.accuracy_score(y_test, clf.predict(X_test_pca)))
confusion_matrix(y_test, clf.predict(X_test_pca))
print(classification_report(y_test, clf.predict(X_test_pca), target_names=top_senders))
# we add To and Bcc/cc features from mail metadata to see if it will increase our accuracy:



enron_parsed['To'] = enron_parsed['To'].astype(str) 

data = pd.DataFrame(list(map(content_to_wordlist, enron_parsed[['Subject', 'content', 'To']]

                             .apply(lambda x: ' '.join(x), axis=1))), 

                          columns = ["content"])
data['bcc'] = enron_parsed['Bcc'].isnull().astype(int)



data = data.assign(sender=enron_sent["sender"].values)

data = data.replace({'sender': mapping})

data.head()
data_train, data_test, y_train, y_test = train_test_split(data[['content', 'bcc']], data['sender'], test_size=0.25)



vect = TfidfVectorizer(sublinear_tf=True, use_idf=True)

X_train = vect.fit_transform(data_train.content)

X_test = vect.transform(data_test.content)



X_train = hstack([coo_matrix(X_train), coo_matrix(data_train.bcc).transpose()])

X_test = hstack([coo_matrix(X_test), coo_matrix(data_test.bcc).transpose()])
print(X_train.shape )

print(X_test.shape )
tsvd = TruncatedSVD(n_components = 120)

X_train_pca = tsvd.fit_transform(X_train)

X_test_pca = tsvd.transform(X_test)



clf = LinearSVC()

clf.fit(X_train_pca, y_train)



print(metrics.accuracy_score(y_test, clf.predict(X_test_pca)))

print(metrics.f1_score(y_test, clf.predict(X_test_pca), average="weighted"))
confusion_matrix(y_test, clf.predict(X_test_pca))
print(classification_report(y_test, clf.predict(X_test_pca), target_names=top_senders))