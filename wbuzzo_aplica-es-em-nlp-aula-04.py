import nltk



nltk.download('nps_chat')
from nltk.corpus import nps_chat



print(nps_chat.fileids())
import nltk



x_data_nps = []



for fileid in nltk.corpus.nps_chat.fileids():

    x_data_nps.extend([post.text for post in nps_chat.xml_posts(fileid)])



y_data_nps = [0] * len(x_data_nps)



x_data_gut = []

for fileid in nltk.corpus.gutenberg.fileids():

    x_data_gut.extend([' '.join(sent) for sent in nltk.corpus.gutenberg.sents(fileid)])

    

y_data_gut = [1] * len(x_data_gut)



x_data_full = x_data_nps[:500] + x_data_gut[:500]

print(len(x_data_full))

y_data_full = y_data_nps[:500] + y_data_gut[:500]

print(len(y_data_full))
import numpy as np



x_data = np.array(x_data_full, dtype=object)

#x_data = np.array(x_data_full)

print(x_data.shape)

y_data = np.array(y_data_full)

print(y_data.shape)
train_indexes = np.random.rand(len(x_data)) < 0.80



print(len(train_indexes))

print(train_indexes[:10])
x_data_train = x_data[train_indexes]

y_data_train = y_data[train_indexes]



print(len(x_data_train))

print(len(y_data_train))
x_data_test = x_data[~train_indexes]

y_data_test = y_data[~train_indexes]



print(len(x_data_test))

print(len(y_data_test))
from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

import string

from nltk.corpus import wordnet



stopwords_list = stopwords.words('english')



lemmatizer = WordNetLemmatizer()



def my_tokenizer(doc):

    words = word_tokenize(doc)

    

    pos_tags = pos_tag(words)

    

    non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list]

    

    non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation]

    

    lemmas = []

    for w in non_punctuation:

        if w[1].startswith('J'):

            pos = wordnet.ADJ

        elif w[1].startswith('V'):

            pos = wordnet.VERB

        elif w[1].startswith('N'):

            pos = wordnet.NOUN

        elif w[1].startswith('R'):

            pos = wordnet.ADV

        else:

            pos = wordnet.NOUN

        

        lemmas.append(lemmatizer.lemmatize(w[0], pos))



    return lemmas

    

    
from sklearn.decomposition import TruncatedSVD



class SVDDimSelect(object):

    def fit(self, X, y=None):        

        try:

            self.svd_transformer = TruncatedSVD(n_components=round(X.shape[1]/2))

            self.svd_transformer.fit(X)

        

            cummulative_variance = 0.0

            k = 0

            for var in sorted(self.svd_transformer.explained_variance_ratio_)[::-1]:

                cummulative_variance += var

                if cummulative_variance >= 0.5:

                    break

                else:

                    k += 1

                

            self.svd_transformer = TruncatedSVD(n_components=k)

        except Exception as ex:

            print(ex)

            

        return self.svd_transformer.fit(X)

    

    def transform(self, X, Y=None):

        return self.svd_transformer.transform(X)

        

    def get_params(self, deep=True):

        return {}
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn import neighbors



clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')



my_pipeline = Pipeline([('tfidf', TfidfVectorizer(tokenizer=my_tokenizer)),\

                       ('svd', SVDDimSelect()), \

                       ('clf', clf)])
from sklearn.model_selection import RandomizedSearchCV

import scipy



par = {'clf__n_neighbors': range(1, 60), 'clf__weights': ['uniform', 'distance']}





hyperpar_selector = RandomizedSearchCV(my_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20)

print(x_data_train)
hyperpar_selector.fit(X=x_data_train, y=y_data_train)
print("Best score: %0.3f" % hyperpar_selector.best_score_)

print("Best parameters set:")

best_parameters = hyperpar_selector.best_estimator_.get_params()

for param_name in sorted(par.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))
from sklearn.metrics import *



y_pred = hyperpar_selector.predict(x_data_test)



print(accuracy_score(y_data_test, y_pred))
import pickle



string_obj = pickle.dumps(hyperpar_selector)
model_file = open('model.pkl', 'wb')



model_file.write(string_obj)



model_file.close()


model_file = open('model.pkl', 'rb')

model_content = model_file.read()



obj_classifier = pickle.loads(model_content)



model_file.close()



res = obj_classifier.predict(["what's up bro?"])



print(res)
res = obj_classifier.predict(x_data_test)

print(accuracy_score(y_data_test, res))
res = obj_classifier.predict(x_data_test)



print(res)
formal = [x_data_test[i] for i in range(len(res)) if res[i] == 1]



for txt in formal:

    print("%s\n" % txt)

informal = [x_data_test[i] for i in range(len(res)) if res[i] == 0]



for txt in informal:

    print("%s\n" % txt)
res2 = obj_classifier.predict(["Emma spared no exertions to maintain this happier flow of ideas , and hoped , by the help of backgammon , to get her father tolerably through the evening , and be attacked by no regrets but her own"])



print(res2)
import nltk

nltk.download('sentence_polarity')

from nltk.corpus import sentence_polarity



sentence_polarity.sents(categories=['pos'])
nltk.corpus.sentence_polarity.fileids()

['rt-polarity.neg', 'rt-polarity.pos']
import nltk



x_data_pos = []



for sent in nltk.corpus.sentence_polarity.fileids():

    x_data_pos.extend([' '.join(sent) for sent in sentence_polarity.sents(categories=['pos'])])



y_data_pos = [0] * len(x_data_pos)



x_data_neg = []



for fileid in nltk.corpus.sentence_polarity.fileids():

    x_data_pos.extend([' '.join(sent) for sent in sentence_polarity.sents(categories=['neg'])])

    

y_data_neg = [1] * len(x_data_neg)



x_data_pos_neg = x_data_pos[:500] + x_data_neg[:500]

print(len(x_data_pos_neg))

y_data_pos_neg = y_data_pos[:500] + y_data_neg[:500]

print(len(y_data_pos_neg))
import numpy as np



x_data = np.array(x_data_pos_neg, dtype=object)

print(x_data.shape)

y_data = np.array(y_data_pos_neg)

print(y_data.shape)
train_indexes = np.random.rand(len(x_data)) < 0.80



print(len(train_indexes))

print(train_indexes[:10])
x_data_train = x_data[train_indexes]

y_data_train = y_data[train_indexes]



print(len(x_data_train))

print(len(y_data_train))
x_data_test = x_data[~train_indexes]

y_data_test = y_data[~train_indexes]



print(len(x_data_test))

print(len(y_data_test))
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn import neighbors



clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')



my_pipeline = Pipeline([('tfidf', TfidfVectorizer(tokenizer=my_tokenizer)),\

                       ('svd', SVDDimSelect()), \

                       ('clf', clf)])
from sklearn.model_selection import RandomizedSearchCV

import scipy



par = {'clf__n_neighbors': range(1, 60), 'clf__weights': ['uniform', 'distance']}

hyperpar_selector = RandomizedSearchCV(my_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20)
hyperpar_selector.fit(X=x_data_train, y=y_data_train)
print("Best score: %0.3f" % hyperpar_selector.best_score_)

print("Best parameters set:")

best_parameters = hyperpar_selector.best_estimator_.get_params()

for param_name in sorted(par.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))