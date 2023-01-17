# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_path = '/kaggle/input/bbcnewsclassification/BBC News Train.csv'
df = pd.read_csv(train_path)
print(df.shape)
df.head()
df.Text[1]
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import sent_tokenize, word_tokenize, pos_tag


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
    
wnl = WordNetLemmatizer()
ls = ['running', 'ran', 'happily', 'checking']
[wnl.lemmatize(i) for i in ls]
from re import findall

pattern = r"(?u)\b\w\w+\b"
test_str = """ Winnie's Umbrealla is amainx 3i!"""

findall(pattern,test_str)
def tokenizeText(text,includePunctuation=True, \
               includeStopWords=False,isLowerCase=True, \
               isRemoveNumbers=False,lemmatized=False):
    '''
    Given text, return a list of tokens (words or punctuation)
    
    Options:
        includePunctuation = True (default) if the bag of words include punctuation as token
                           = False if the bag of words exclude punctuation.
        includeStopWords = True if stop words are not cleaned from bag of words
                         = False (default) to return clean words without stop words.
        isLowerCase = True (default) if all words are transformed into lower case
                    = False if no transformation of case
        isRemoveNumbers = True to strip all numbers from the text
                        = False (default) if no numbers to be stripped off the text
        lemmatized = True to carry out lemmeatization
                   = False (default) to keep word endings
    '''
    if isRemoveNumbers==True:
        import re
        text = re.sub("\d+", " ", text)

    if includePunctuation==True:
        # include punctuation as part of token or word
        tokens = [word for token in nltk.sent_tokenize(text) 
                  for word in nltk.word_tokenize(token)]
    else:
        # remove punctuation, words only
#        tokenizer=nltk.tokenize.RegexpTokenizer(r'\w+')
        pattern = r"(?u)\b\w\w+\b"
        tokens = findall(pattern,text)
#        tokens = [word for token in nltk.sent_tokenize(text) 
#                  for word in tokenizer.tokenize(token)]
        
    if isLowerCase==True:
        tokens=[word.lower() for word in tokens]
        
    if includeStopWords==False:
        stopWords=set(nltk.corpus.stopwords.words('english'))  # load stop words
        tokens = [t for t in tokens if t not in stopWords]     # cleaning word from stop words 
    
    if lemmatized==True:
        tmp = []
        for word, pos in pos_tag(tokens):
            wordnet_pos = get_wordnet_pos(pos)
            if wordnet_pos == '':
                tmp.append(word)
            else:
                word_lemmatized = wnl.lemmatize(word,wordnet_pos)
                tmp.append(word_lemmatized)
        tokens = tmp
    
    return tokens
myTokenizerConfig=[False,False,True,True,False]

all_results = []
meta_counter = Counter()
for i,text in enumerate(df.Text):
    if i % 50 ==0:
        print("processing {}th text now".format(i))
    clean_tokens = tokenizeText(text,includePunctuation=myTokenizerConfig[0], \
                                          includeStopWords=myTokenizerConfig[1],isLowerCase=myTokenizerConfig[2], \
                                          isRemoveNumbers=myTokenizerConfig[3],lemmatized=myTokenizerConfig[4])
    meta_counter.update(clean_tokens)

vocab_size = len(list(meta_counter.keys()))
vocab_list = list(meta_counter.keys())
word_count = sorted(meta_counter.items(), key=lambda item: item[1],reverse=True)
print(word_count[:20])
print("========================================")
print("Total Vocabulary Size: {}".format(len(word_count)))
print(vocab_size)
prob_topic = np.log(df.groupby(['Category'])['ArticleId'].count()/len(df))
all_categories = list(prob_topic.index)
prob_topic

vocab_size
from collections import Counter,defaultdict

## Split dataframe by category
d = dict(tuple(df.groupby('Category')))
category_word_count = dict()


for m,category in enumerate(all_categories):
    print('------COUNTING for {} topic now------'.format(category))
    cur_df = d[category]

    # Initialize Counter
    word_counter = Counter()
    total_tokens = 0
    all_prob_count = dict()

    # tokenize
    for i,text in enumerate(cur_df.Text):
        if i % 50 ==0:
            print("processing {}th text now".format(i))
        clean_tokens = tokenizeText(text,includePunctuation=myTokenizerConfig[0], \
                                              includeStopWords=myTokenizerConfig[1],isLowerCase=myTokenizerConfig[2], \
                                              isRemoveNumbers=myTokenizerConfig[3],lemmatized=myTokenizerConfig[4])
        # update count
        word_counter.update(clean_tokens)

    # need to calculate probability
    for vocab in vocab_list:
        cur_count = word_counter.get(vocab,0) +1
        total_tokens += cur_count
        
    for vocab in vocab_list:
        cur_count = word_counter.get(vocab,0) +1
        cur_prob = np.log(cur_count) - np.log(total_tokens) #- np.log(vocab_size)
        all_prob_count[vocab] = cur_prob
    category_word_count[category] = {'counter':word_count,'prob_table':all_prob_count}
    

#array([86558., 73840., 88081., 83092., 93050.])
len(list(all_prob_count.keys()))
n_classes = 5
np.full(n_classes, -np.log(n_classes))
import operator
from scipy.special import logsumexp

val_path = '/kaggle/input/bbcnewsclassification/BBC News Test.csv'
val_df = pd.read_csv(val_path)

final_prediction = []

for idx,text in enumerate(val_df.Text):
    article_id = val_df.loc[idx,'ArticleId']

    if idx % 50 ==0:
        print("processing {}th text now".format(idx))
    clean_tokens = tokenizeText(text,includePunctuation=myTokenizerConfig[0], \
                                          includeStopWords=myTokenizerConfig[1],isLowerCase=myTokenizerConfig[2], \
                                          isRemoveNumbers=myTokenizerConfig[3],lemmatized=myTokenizerConfig[4])
    ##### HOW COME LAST CATEGORY IS ALWAYS WITH THE HIGHEST PROBABILITY
    ##### TROUBLESHOOT

    all_probs = np.zeros(5)
    for q,category in enumerate(all_categories):
        cur_total_prob = -1.60943791
        for token in clean_tokens:
            cur_prob = category_word_count[category]['prob_table'].get(token)
            if cur_prob:
                cur_total_prob += cur_prob
        all_probs[q] = cur_total_prob
    
    all_probs = all_probs - logsumexp(all_probs)
    pred = np.argmax(all_probs)
    final_prediction.append([article_id,all_categories[pred]])

#        category_map = max(prob_count.items(), key=operator.itemgetter(1))
#        
#        category_name = category_map[0]
#        log_prob = category_map[1]
#        final_prediction.append([article_id,category_name,log_prob])
    


val_pred_df = pd.DataFrame(final_prediction)
val_pred_df.columns = ['ArticleId','Category_Pred']
val_pred_df
val_label_path = '/kaggle/input/bbcnewsclassification/BBC News Sample Solution.csv'
val_label_df = pd.read_csv(val_label_path)
assert(val_pred_df.shape[0]<=val_label_df.shape[0])
merged_df = val_pred_df.merge(val_label_df, left_on="ArticleId", right_on="ArticleId")
merged_df
from sklearn.metrics import accuracy_score
y_true = merged_df.Category
y_pred = merged_df.Category_Pred

accuracy_score(y_true, y_pred)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred, labels=all_categories)
# countVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(stop_words = 'english')
vec.fit(df.Text)
X_transformed = vec.transform(df.Text)
X_transformed
X_test_transformed = vec.transform(val_df.Text)
X_test_transformed
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_transformed, df.Category)
smoothed_cc = (nb.feature_count_ + 1).sum(axis=1)

np.log(smoothed_cc.reshape(-1, 1))
# predict class
y_pred_class = nb.predict(X_test_transformed)

# predict probabilities
y_pred_proba = nb.predict_log_proba(X_test_transformed)
merged_df.loc[:,'sk_pred'] = y_pred_proba
pd.DataFrame(np.concatenate([merged_df,y_pred_proba],axis=1))