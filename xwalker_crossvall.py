import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from re import sub

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt

from nltk import word_tokenize, RegexpTokenizer, TweetTokenizer, PorterStemmer
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test =  pd.read_csv('../input/nlp-getting-started/test.csv')

submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
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
train.text = train.text.apply(lambda x: x.lower()) #transforma tetxo em minúsculo

train.text = train.text.apply(lambda x: " ".join(x.split())) #deleta excesso de espaços

train.text = train.text.apply(lambda x: sub(r'\d+', '', x)) #deleta números

train.text = train.text.apply(lambda x: remove_pontucations(x)) #remove pontuações e caracteres especiais

train.text = train.text.apply(lambda x: stopWords(x))

train.text = train.text.apply(lambda x: x.replace('_', ' '))

train.text = train.text.apply(lambda x: remove_words_min(x))
train.head()
text_transformer = TfidfVectorizer(

    stop_words='english', 

    ngram_range=(1, 2), 

    lowercase=True, 

    max_features=150000

)
X_train_text = text_transformer.fit_transform(train.text)

X_test_text = text_transformer.transform(test.text)
logit = MultinomialNB()



skf = StratifiedKFold(

    n_splits=5, 

    shuffle=True,

    random_state=17

)



cv_results = cross_val_score(

    logit, 

    X_train_text, 

    train.target, 

    cv=skf, 

    scoring='f1_micro'

)
cv_results, cv_results.mean()
logit = logit.fit(X_train_text, train.target)

y_preds = logit.predict(X_test_text)

submission.target = y_preds

submission.to_csv("submission.csv", index=False)