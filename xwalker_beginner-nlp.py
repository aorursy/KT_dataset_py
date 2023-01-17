import pandas as pd

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from re import sub

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt

from nltk import word_tokenize, RegexpTokenizer, TweetTokenizer, PorterStemmer

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report, accuracy_score, f1_score

path = '../input/amazon-pet-product-reviews-classification/'

data_train = pd.read_csv(path + 'train.csv')

val_df = pd.read_csv(path + 'valid.csv')

train  = pd.concat([data_train, val_df])

test = pd.read_csv(path + 'test.csv')
def names2labels(labels):

    

    if labels == 0: return 'birds'

    if labels == 1: return 'bunny rabbit central'

    if labels == 2: return 'cats'

    if labels == 3: return 'dogs'

    if labels == 4: return 'fish aquatic pets'

    if labels == 5: return 'small animals'

    else: return labels

    

def labels2value(labels):

    

    if labels == 'birds': return 0

    if labels == 'bunny rabbit central': return 1

    if labels == 'cats': return 2

    if labels == 'dogs': return 3

    if labels == 'fish aquatic pets': return 4

    if labels == 'small animals': return 5

    else: return labels

    

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



def remove_words_min(text, param):

    tmp = text

    

    for x in tmp.split():

        if len(x) < param:

            tmp = tmp.replace(x, '')

    return " ".join(tmp.split())
def clean(data):

    data.text = data.text.apply(lambda x: x.lower()) #transforma tetxo em minúsculo

    data.text = data.text.apply(lambda x: " ".join(x.split())) #deleta excesso de espaços

    data.text = data.text.apply(lambda x: sub(r'\d+', '', x)) #deleta números

    data.text = data.text.apply(lambda x: remove_pontucations(x)) #remove pontuações e caracteres especiais

    data.text = data.text.apply(lambda x: stopWords(x))

    data.text = data.text.apply(lambda x: x.replace('_', ' '))

    data.text = data.text.apply(lambda x: remove_words_min(x, 2))
clean(data_train)

clean(test)
data_train.head()
test.head()
#train.label =  train.label.apply(lambda x: labels2value(x))
X_train, X_test, y_train, y_test = train_test_split(train.text, train.label, test_size=0.33, random_state=42)
pipe_clf = Pipeline(

    [('vect', CountVectorizer(analyzer='word', stop_words='english', tokenizer=word_tokenize, max_features=20000)),

     ('tfidf', TfidfTransformer()),

     ('clf', LogisticRegression(

         C=5e1, 

         solver='lbfgs', 

         multi_class='multinomial', 

         random_state=17, 

         n_jobs=-1)

     )]

)
pipe_clf = pipe_clf.fit(X_train, y_train)

y_preds = pipe_clf.predict(X_test)
print(classification_report(y_test, y_preds))

print('Acurácia: ', accuracy_score(y_test, y_preds))
pipe_clf = pipe_clf.fit(train.text, train.label)

y_preds = pipe_clf.predict(test.text)
test['label'] = y_preds

submission = test[['id', 'label']]

submission.to_csv("submission.csv", index=False)