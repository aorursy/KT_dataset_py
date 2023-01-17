from glob import glob

import pandas as pd

from re import sub

#Primeiro Modelo

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt





from nltk import word_tokenize, RegexpTokenizer, TweetTokenizer, PorterStemmer



#Avaliação dos Modelos

from sklearn.metrics import classification_report, accuracy_score, f1_score

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline
abbrev = pd.read_csv("../input/abbrev/abbrev.csv")

data_train = pd.read_csv('../input/nlp-getting-started/train.csv')

data_test = pd.read_csv('../input/nlp-getting-started/test.csv')

submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
def plot(config):

    plt.figure(figsize = (15, 8))

    plt.grid()

    plt.plot(results['config'], results['f1'], 'o-')

    plt.ylabel('F1 Score')

    plt.xlabel('Max_features')

    plt.title('Relação Parâmetro x F1 Score')

    plt.show();

    

def remove_URL(text):

    url = RegexpTokenizer(r'https?://\S+|www\.\S+', gaps = True)

    return " ".join(url.tokenize(text))



def stopWords(tweet):  

    stop_words, toker = stopwords.words('english'), TweetTokenizer()

    words_tokens = toker.tokenize(tweet)

    return " ".join([word for word in  words_tokens if not word in stop_words])



def remove_pontucations(text):

    tokenizer_dots = RegexpTokenizer(r'\w+')

    return " ".join(tokenizer_dots.tokenize(text))



def abbrev2normal(tweet):



    abbrevs = list(abbrev.abbrev)

    noabbrev = list(abbrev.nobbrev)

    

    for this in TweetTokenizer().tokenize(tweet):

        

        for idx, value in enumerate(abbrevs):

            if this == value:

                tweet = tweet.replace(this, noabbrev[idx])

                break

            

    return tweet



def remove_words_min(text):

    tmp = text

    

    for x in tmp.split():

        if len(x) < 2:

            tmp = tmp.replace(x, '')

    return " ".join(tmp.split())
data_train.text = data_train.text.apply(lambda x: x.lower()) #transforma tetxo em minúsculo

 

data_train.text = data_train.text.apply(lambda x: abbrev2normal(x))  

data_train.text = data_train.text.apply(lambda x: sub(r'\d+', '', x)) #deleta números

data_train.text = data_train.text.apply(lambda x: remove_pontucations(x)) #remove pontuações e caracteres especiais

data_train.text = data_train.text.apply(lambda x: stopWords(x))

data_train.text = data_train.text.apply(lambda x: remove_words_min(x))
X_train, X_test, y_train, y_test = train_test_split(

    data_train.text, 

    data_train.target.values, 

    test_size=0.33

)
def make_model(clf):

    

    pipe_clf = Pipeline(

        [('vect', CountVectorizer(analyzer='word', stop_words='english', tokenizer=word_tokenize)),

         ('tfidf', TfidfTransformer()),

         ('clf', clf),

        ]

    )



    pipe_clf = pipe_clf.fit(X_train, y_train)

    y_preds = pipe_clf.predict(X_test)



    print(classification_report(y_test, y_preds))

    print('Acurácia: ', accuracy_score(y_test, y_preds))

    print('F1 Score: ', f1_score(y_test, y_preds))

    

    return pipe_clf
pipe_clf = make_model(MultinomialNB())
clf = pipe_clf.fit(data_train.text, data_train.target)

y_preds = clf.predict(data_test.text)

submission.target = y_preds

submission.to_csv("submission.csv", index=False)