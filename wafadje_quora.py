import seaborn as sns

import pandas as pd

import numpy as np

import string

import matplotlib.pyplot as plt

from tqdm import tqdm

tqdm.pandas()



from sklearn.metrics import f1_score

from sklearn.model_selection import KFold

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB,BernoulliNB
glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

paragram =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

wiki_news = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
df = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print("Train shape : ",df.shape)

print("Test shape : ",test.shape)
df.info()

test.info()
df.head(n=2)
df.shape
df.describe()
df.isnull().sum()

test.isnull().sum()
df.where(df['target']==1).count()
df.where(df['target']==0).count()
sincere_questions = df[df['target'] == 0]

insincere_questions = df[df['target'] == 1]

insincere_questions.tail(5)
question = df['question_text']

i=0

for q in question[:5]:

    i=i+1

    print('sample '+str(i)+':' ,q)
df["num_words"] = df["question_text"].apply(lambda x: len(str(x).split()))



print('maximum of num_words in train',df["num_words"].max())

print('min of num_words in train',df["num_words"].min())

df["num_unique_words"] = df["question_text"].apply(lambda x: len(set(str(x).split())))

#test["num_unique_words"] = test["question_text"].apply(lambda x: len(set(str(x).split())))

print('maximum of num_unique_words in train',df["num_unique_words"].max())

print('mean of num_unique_words in train',df["num_unique_words"].mean())

#print("maximum of num_unique_words in test",test["num_unique_words"].max())

#print('mean of num_unique_words in train',test["num_unique_words"].mean())
df["num_stopwords"] = df["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

#test["num_stopwords"] = test["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

print('maximum of num_stopwords in train',df["num_stopwords"].max())

#print("maximum of num_stopwords in test",test["num_stopwords"].max())
df["num_punctuations"] =df['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

#test["num_punctuations"] =test['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

print('maximum of num_punctuations in train',df["num_punctuations"].max())

#print("maximum of num_punctuations in test",test["num_punctuations"].max()
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

import re

from nltk.stem import WordNetLemmatizer



lemm_ = WordNetLemmatizer()

st = PorterStemmer()

stops = set(stopwords.words("english"))

def cleanData(text, lowercase = True, remove_stops = True, stemming = False, lemma = True):



    txt = str(text)

    txt = re.sub(r'[^a-zA-Z. ]+|(?<=\\d)\\s*(?=\\d)|(?<=\\D)\\s*(?=\\d)|(?<=\\d)\\s*(?=\\D)',r'',txt)

    txt = re.sub(r'\n',r' ',txt)

    

    #converting to lower case

    if lowercase:

        txt = " ".join([w.lower() for w in txt.split()])

    

    # removing stop words

    if remove_stops:

        txt = " ".join([w for w in txt.split() if w not in stops])

    

    # stemming

    if stemming:

        txt = " ".join([st.stem(w) for w in txt.split()])

        

    if lemma:

        txt = " ".join([lemm_.lemmatize(w) for w in txt.split()])



    return txt

df['clean_question_text'] = df['question_text'].map(lambda x: cleanData(x))

test['clean_question_text'] = test['question_text'].map(lambda x: cleanData(x))

test['clean_question_text']
max_features = 50000  ##More than this would filter in noise also

tfidf_vectorizer = TfidfVectorizer(ngram_range =(2,4) , max_df=0.90, min_df=5, max_features=max_features) ##4828 features found

#tfidf_feature_names = tfidf_vectorizer.get_feature_names()
X = tfidf_vectorizer.fit_transform(df['clean_question_text'])

X_te = tfidf_vectorizer.transform(test['clean_question_text'])

tfidf_feature_names = tfidf_vectorizer.get_feature_names()

from gensim.models import KeyedVectors



news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)
y = df["target"]
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,random_state=42)
# Classification and prediction

clf = LogisticRegression(C=10, penalty='l1')

clf.fit(X_train, y_train)
clf.score(X_val, y_val)
p_test = clf.predict_proba(X_te)[:, 0]

y_te = (p_test > 0.5).astype(np.int)
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

import nltk



stop = stopwords.words('english')

stemmer = PorterStemmer()

lem = nltk.WordNetLemmatizer()

eng_stopwords = set(stopwords.words("english"))
#df['question_text'] = df['question_text'].apply(lambda x: x.lower())

df['question'] = df['question_text'].str.replace(r"[^a-z0-9 ]", '')

df['tokens'] = df['question'].apply(word_tokenize)

df['tokens'] = df['tokens'].map(lambda x: [word for word in x if word not in eng_stopwords])

df['lems'] = df['tokens'].map(lambda x: [lem.lemmatize(word) for word in x])
df['lems']
from gensim.corpora import Dictionary



dicti = Dictionary(df['lems'])



bow = [dicti.doc2bow(line) for line in df['lems']]
# TODO: Compute TF-IDF

from gensim.models import TfidfModel



tfmodel = TfidfModel(bow)



tfidf = tfmodel[bow]
from gensim.models import LsiModel



lsa = LsiModel(corpus = tfidf, num_topics=10, id2word = dicti)
from pprint import pprint



pprint(lsa.print_topics(num_words=4))
from gensim.models import LdaModel

lda = LdaModel(corpus = tfidf, num_topics=6, id2word = dicti, passes=5)
pprint(lda.print_topics(num_words=4))
import pyLDAvis



import pyLDAvis.gensim



# Visualize the topics

pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda, bow, dicti)

vis
ax=sns.countplot(x='target',hue="target", data=df  ,linewidth=1,edgecolor=sns.color_palette("dark", 3))

plt.title('Data set distribution');
from sklearn.model_selection import train_test_split 

import nltk
def build_vocab(sentences, verbose =  True):

    """

    :param sentences: list of list of words

    :return: dictionary of words and their count

    """

    vocab = {}

    for sentence in tqdm(sentences, disable = (not verbose)):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab

def clean_text(x):



    x = str(x)

    for punct in "/-'":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    return x
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

import nltk



stop = stopwords.words('english')

stemmer = PorterStemmer()

lem = nltk.WordNetLemmatizer()

eng_stopwords = set(stopwords.words("english"))
df["question_text"] = df["question_text"].apply(lambda x: clean_numbers(x))

df["question_text"] = df["question_text"].apply(lambda x: clean_text(x))

df['lowered_question'] = df['question_text'].apply(lambda x: x.lower())

df['question'] = df['lowered_question'].str.replace(r"[^a-z0-9 ]", '')

df['tokens'] = df['question'].apply(nltk.word_tokenize)

df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in eng_stopwords])

df['lems'] = df['tokens'].apply(lambda x: [lem.lemmatize(word) for word in x])
df_train = pd.DataFrame(df['lems'])

df_train
def clean_numbers(x):



    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x
import re
sentences = df_train['lems'].apply(lambda x: x.split())

vocab = build_vocab(sentences)
import operator 



def check_coverage(vocab,embeddings_index):

    a = {}

    oov = {}

    k = 0

    i = 0

    for word in tqdm(vocab):

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:



            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x
oov = check_coverage(vocab,embeddings_index)
oov[:10]
train_text['lems'] = df['lems']

train_text['lems']
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
tfidf_vectorizer = TfidfVectorizer()

tfidf_vectorizer.fit(all_text)



count_vectorizer = CountVectorizer()

count_vectorizer.fit(all_text)



train_text_features_cv = count_vectorizer.transform(train_text)

test_text_features_cv = count_vectorizer.transform(test_text)



train_text_features_tf = tfidf_vectorizer.transform(train_text)

test_text_features_tf = tfidf_vectorizer.transform(test_text)
kfold = KFold(n_splits = 5, shuffle = True, random_state = 2018)

test_preds = 0

oof_preds = np.zeros([df.shape[0],])



for i, (train_idx,valid_idx) in enumerate(kfold.split(df['lems'])):

    x_train, x_valid = train_text_features_tf[train_idx,:], train_text_features_tf[valid_idx,:]

    y_train, y_valid = train_target[train_idx], train_target[valid_idx]

    classifier = LogisticRegression()

    print('fitting.......')

    classifier.fit(x_train,y_train)

    print('predicting......')

    print('\n')

    oof_preds[valid_idx] = classifier.predict_proba(x_valid)[:,1]

    test_preds += 0.2*classifier.predict_proba(test_text_features_tf)[:,1]
pred_train = (oof_preds > .25).astype(np.int)

f1_score(train_target, pred_train)