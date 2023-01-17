# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")

from gensim.models import Doc2Vec

from sklearn import utils

from sklearn.model_selection import train_test_split

import gensim

from sklearn.linear_model import LogisticRegression

from gensim.models.doc2vec import TaggedDocument

import re

import seaborn as sns

import matplotlib.pyplot as plt

import string

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer


def remove_punctuation(text):

    '''a function for removing punctuation'''

    import string

    # replacing the punctuations with no space, 

    # which in effect deletes the punctuation marks 

    translator = str.maketrans('', '', string.punctuation)

    # return the text stripped of punctuation marks

    return text.translate(translator)





# extracting the stopwords from nltk library

sw = stopwords.words('english')

# displaying the stopwords

np.array(sw);





def stopwords(text):

    '''a function for removing the stopword'''

    # removing the stop words and lowercasing the selected words

    text = [word.lower() for word in text.split() if word.lower() not in sw]

    # joining the list of words with space separator

    return " ".join(text)



stemmer = SnowballStemmer("english")



def stemming(text):    

    '''a function which stems each word in the given text'''

    text = [stemmer.stem(word) for word in text.split()]

    return " ".join(text) 

    





def clean_loc(x):

    if x == 'None':

        return 'None'

    elif x == 'Earth' or x =='Worldwide' or x == 'Everywhere':

        return 'World'

    elif 'New York' in x or 'NYC' in x:

        return 'New York'    

    elif 'London' in x:

        return 'London'

    elif 'Mumbai' in x:

        return 'Mumbai'

    elif 'Washington' in x and 'D' in x and 'C' in x:

        return 'Washington DC'

    elif 'San Francisco' in x:

        return 'San Francisco'

    elif 'Los Angeles' in x:

        return 'Los Angeles'

    elif 'Seattle' in x:

        return 'Seattle'

    elif 'Chicago' in x:

        return 'Chicago'

    elif 'Toronto' in x:

        return 'Toronto'

    elif 'Sacramento' in x:

        return 'Sacramento'

    elif 'Atlanta' in x:

        return 'Atlanta'

    elif 'California' in x:

        return 'California'

    elif 'Florida' in x:

        return 'Florida'

    elif 'Texas' in x:

        return 'Texas'

    elif 'United States' in x or 'USA' in x:

        return 'USA'

    elif 'United Kingdom' in x or 'UK' in x or 'Britain' in x:

        return 'UK'

    elif 'Canada' in x:

        return 'Canada'

    elif 'India' in x:

        return 'India'

    elif 'Kenya' in x:

        return 'Kenya'

    elif 'Nigeria' in x:

        return 'Nigeria'

    elif 'Australia' in x:

        return 'Australia'

    elif 'Indonesia' in x:

        return 'Indonesia'

    elif x in top_loc:

        return x

    else: return 'Others'

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)

# importing data



new_df = pd.read_csv('../input/nlp-getting-started/train.csv')

final_test = pd.read_csv('../input/nlp-getting-started/test.csv')

new_df['keyword'] = new_df['keyword'].fillna('unknown')

new_df['location'] = new_df['location'].fillna('unknown')





new_df = new_df[['target', 'location', 'text', 'keyword']]

final_test = final_test[['location', 'text', 'keyword']]







new_df['text'] = new_df['text'].apply(remove_punctuation)

new_df['text'] = new_df['text'].apply(stopwords)

new_df['text'] = new_df['text'].apply(stemming)

new_df['text'] = new_df['text'].apply(remove_URL)

new_df['text'] = new_df['text'].apply(remove_html)

new_df['text'] = new_df['text'].apply(remove_emoji)

new_df['text'] = new_df['text'].apply(remove_punct)







final_test['text'] = final_test['text'].apply(remove_punctuation)

final_test['text'] = final_test['text'].apply(stopwords)

final_test['text'] = final_test['text'].apply(stemming)

final_test['text'] = final_test['text'].apply(remove_URL)

final_test['text'] = final_test['text'].apply(remove_html)

final_test['text'] = final_test['text'].apply(remove_emoji)

final_test['text'] = final_test['text'].apply(remove_punct)

raw_loc = new_df.location.value_counts()

top_loc = list(raw_loc[raw_loc>=10].index)

new_df['location_clean'] = new_df['location'].apply(lambda x: clean_loc(str(x)))

final_test['location_clean'] = final_test['location'].apply(lambda x: clean_loc(str(x)))

from bs4 import BeautifulSoup



def cleanText(text):

    text = BeautifulSoup(text, "lxml").text

    text = re.sub(r'\|\|\|', r' ', text) 

    text = re.sub(r'http\S+', r'<URL>', text)

    text = text.lower()

    text = text.replace('x', '')

    return text

new_df['text'] = new_df['text'].apply(cleanText)

new_df['keyword'] = new_df['keyword'].apply(cleanText)

new_df['location_clean'] = new_df['location_clean'].apply(cleanText)



final_test['text'] = final_test['text'].apply(cleanText)

final_test['keyword'] = final_test['keyword'].fillna('unknown')

final_test['keyword'] = final_test['keyword'].apply(cleanText)

final_test['location_clean'] = final_test['location_clean'].apply(cleanText)

keyword_df = new_df.groupby(['keyword']).count().reset_index()

keyword_test = final_test.groupby(['keyword']).count().reset_index()



from gensim.test.utils import common_texts, get_tmpfile

from gensim.models import Word2Vec

path = get_tmpfile("word2vec.model")

model = Word2Vec(common_texts, size=100, window=1, min_count=1, workers=4)



model = Word2Vec([list(keyword_df['keyword']) + list(keyword_test['keyword'])], min_count=1)

# traing and test split

train, test = train_test_split(new_df, test_size=0.2, random_state=42)





# another encoding technique for location and keyword, with the event rate

keyword_val = train.groupby('keyword').agg({'target': 'mean'})

location_val = train.groupby('location_clean').agg({'target': 'mean'})





new_train = pd.merge(train, keyword_val, how='left', on = 'keyword')

new_train = pd.merge(new_train, location_val, how='left', on = 'location_clean')



new_test = pd.merge(test, keyword_val, how='left', on = 'keyword')

new_test = pd.merge(new_test, location_val, how='left', on = 'location_clean')

new_train['target_y'].fillna(new_train['target_y'].mean(), inplace=True)

new_train['target'].fillna(new_train['target'].mean(), inplace=True)



new_test['target_y'].fillna(new_train['target_y'].mean(), inplace=True)

new_test['target'].fillna(new_train['target'].mean(), inplace=True)

# now back to creating word embeddings vector for keywords

words = list(model.wv.vocab)



train_w2v = []

test_w2v = []

final_test_w2v = []



for elem in train['keyword']:

    train_w2v.append(model.wv[elem])

    

for elem in test['keyword']:

    test_w2v.append(model.wv[elem])

    

for elem in final_test['keyword']:

    final_test_w2v.append(model.wv[elem])





# below code to create doc2vec vector for text variable



import multiprocessing

cores = multiprocessing.cpu_count()



import nltk

from nltk.corpus import stopwords



def tokenize_text(text):

    tokens = []

    for sent in nltk.sent_tokenize(text):

        for word in nltk.word_tokenize(sent):

            if len(word) < 2:

                continue

            tokens.append(word.lower())

    return tokens



train_tagged = train.apply(lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.target]), axis=1)

test_tagged = test.apply(lambda r: TaggedDocument(words=tokenize_text(r['text']),tags=[r.target]), axis=1)





model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)

model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])



%%time

for epoch in range(30):

    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)

    model_dbow.alpha -= 0.002

    model_dbow.min_alpha = model_dbow.alpha





def vec_for_learning(model, tagged_docs):

    sents = tagged_docs.values

    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])

    return targets, regressors



y_train, X_train = vec_for_learning(model_dbow, train_tagged)

y_test, X_test = vec_for_learning(model_dbow, test_tagged)







# now combining the doc2vec vector, with word2vec vector and keyword and location encoding 







from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



vectorizer = CountVectorizer(analyzer='word', binary=True)

vectorizer.fit(new_df['text'])
X_train_cvec = vectorizer.transform(train['text']).todense()

X_test_cvec = vectorizer.transform(test['text']).todense()



# y = tweets['target'].values

# X.shape, y.shape
X_train_cvec.shape, X_test_cvec.shape
tfidf = TfidfVectorizer(analyzer='word', binary=True)

tfidf.fit(new_df['text'])
X_train_tfidf = tfidf.transform(train['text']).todense()

X_test_tfidf = tfidf.transform(test['text']).todense()
new_X_train = list(map(lambda x,y: np.append(x,y),X_train, new_train['target_y']))

new_X_train_2 = list(map(lambda x,y: np.append(x,y),new_X_train, new_train['target']))

new_X_train_3 = list(map(lambda x,y: np.append(x,y),new_X_train_2, train_w2v))





new_X_test = list(map(lambda x,y: np.append(x,y),X_test, new_test['target_y']))

new_X_test_2 = list(map(lambda x,y: np.append(x,y),new_X_test, new_test['target']))

new_X_test_3 = list(map(lambda x,y: np.append(x,y),new_X_test_2, test_w2v))

# CountVectorizer





new_X_train_4 = list(map(lambda x,y: np.append(x,y),new_X_train_3, X_train_cvec))

new_X_test_4 = list(map(lambda x,y: np.append(x,y),new_X_test_3, X_test_cvec))





# TFIDF



new_X_train_5 = list(map(lambda x,y: np.append(x,y),new_X_train_4, X_train_tfidf))

new_X_test_5 = list(map(lambda x,y: np.append(x,y),new_X_test_4, X_test_tfidf))



new_X_test_5[0].dtype
# Simple logistic regression



from sklearn.metrics import accuracy_score, f1_score





logreg = LogisticRegression(n_jobs=1, C=1e5)

logreg.fit(new_X_train_5, y_train)

y_pred = logreg.predict(new_X_test_5)

print ('Testing accuracy : {}'.format(accuracy_score(y_test, y_pred)))

print ('Testing F1 score : {}'.format(f1_score(y_test, y_pred, average='weighted')))



from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_jobs=50, random_state=0)

clf.fit(new_X_train_5, y_train)

from sklearn.metrics import accuracy_score, f1_score



y_pred = clf.predict(new_X_test_5)

print ('Testing accuracy : {}'.format(accuracy_score(y_test, y_pred)))

print ('Testing F1 score : {}'.format(f1_score(y_test, y_pred, average='weighted')))



var_lst = ['var_'+str(i) for i in range(len(new_X_train_4[0]))]



import xgboost as xgb

xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'binary:logistic',

    'silent': 1,

    'seed' : 0,

    'n_estimators': 200,

    'eval_metric': 'logloss'

}

dtrain = xgb.DMatrix(new_X_train_4, y_train, feature_names=var_lst)

xgb_model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)





dtest = xgb.DMatrix(new_X_test_4,feature_names = var_lst )



y_pred = xgb_model.predict(dtest)

# print ('Testing accuracy : {}'.format(accuracy_score(y_test, y_pred)))

print ('Testing F1 score : {}'.format(f1_score(y_test, y_pred.round(), average='weighted')))
new_X_all_4 = new_X_train_4+new_X_test_4

new_y_all_4 = y_train+y_test
new_X_all_4_np = np.array(new_X_all_4)

new_y_all_4_np = np.array(new_y_all_4)
from sklearn.model_selection import StratifiedKFold



models = []

n_splits = 5

fold = 0 

for train_index, test_index in StratifiedKFold(n_splits=n_splits).split(new_X_all_4_np, new_y_all_4_np):



    X_train, X_test = new_X_all_4_np[train_index], new_X_all_4_np[test_index]

    y_train, y_test = new_y_all_4_np[train_index], new_y_all_4_np[test_index]



    clf = LogisticRegression(max_iter=400)



    clf.fit(X_train,y_train)





    models.append(clf)

    fold += 1

    print(fold)
from sklearn.model_selection import StratifiedKFold



models = []

n_splits = 5

fold = 0 

for train_index, test_index in StratifiedKFold(n_splits=n_splits).split(new_X_all_4_np, new_y_all_4_np):



    X_train, X_test = new_X_all_4_np[train_index], new_X_all_4_np[test_index]

    y_train, y_test = new_y_all_4_np[train_index], new_y_all_4_np[test_index]



    clf = RandomForestClassifier(n_jobs=50, random_state=0)



    clf.fit(X_train,y_train)





    models.append(clf)

    fold += 1

    print(fold)
final_test['target'] = 0



final_test = pd.merge(final_test, keyword_val, how='left', on = 'keyword')

final_test = pd.merge(final_test, location_val, how='left', on = 'location_clean')

final_test['target_y'].fillna(new_train['target_y'].mean(), inplace=True)

final_test['target'].fillna(new_train['target'].mean(), inplace=True)



final_test_tagged = final_test.apply(lambda r: TaggedDocument(words=tokenize_text(r['text']),tags=[r.target]), axis=1)

f_y_test, f_X_test = vec_for_learning(model_dbow, final_test_tagged)



final_X_test = list(map(lambda x,y: np.append(x,y), f_X_test, final_test['target_y']))

final_X_test_2 = list(map(lambda x,y: np.append(x,y),final_X_test, final_test['target']))

final_X_test_3 = list(map(lambda x,y: np.append(x,y),final_X_test_2, final_test_w2v))

X_f_test_cvec = vectorizer.transform(final_test['text']).todense()

X_f_test_tfidf = tfidf.transform(final_test['text']).todense()





final_X_test_4 = list(map(lambda x,y: np.append(x,y),final_X_test_3, X_f_test_cvec))

# final_X_test_5 = list(map(lambda x,y: np.append(x,y),final_X_test_4, X_f_test_tfidf))
new_final_4 = list((lambda x: map(x, float), final_X_test_4))
new_final_4_np = np.array(final_X_test_4)



# new_y_all_4_np = np.array(new_y_all_4)

# y_hat = clf.predict(new_final_4_np)



final = np.zeros((new_final_4_np.shape[0]))



for i in range(n_splits):

        clf = models[i]

        preds = clf.predict(new_final_4_np)

        

        final += preds/n_splits



    

final = np.where(final>=0.5,1,0)
# xgboost



# dtest = xgb.DMatrix(final_X_test_4,feature_names = var_lst )



# y_pred_xgb = xgb_model.predict(dtest)
# creating the submissions file

sub_sample = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")



submit = sub_sample.copy()

submit.target = final

submit.to_csv('submit_rf_cv.csv',index=False)


