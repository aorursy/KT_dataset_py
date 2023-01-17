import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

%matplotlib inline



import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

init_notebook_mode(connected = True)



import urllib.request

from PIL import Image

from wordcloud import WordCloud ,STOPWORDS

from IPython.display import Markdown

def bold(string):

    display(Markdown(string))



from tqdm import tqdm

import time

import re

import string

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from scipy.sparse import csr_matrix, hstack

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score as auc

import gc

from collections import defaultdict

import os

import psutil
# read the data file

df_train = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

df_train1 = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv')

df_valid = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

df_test = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test.csv')

sub = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
# preview the date file

bold('**TRAINING DATA**')

display(df_train.head(3))

bold('**BAIS TRAINING DATA**')

display(df_train1.head(3))

bold('**VALIDATION DATA**')

display(df_valid.head(3))

bold('**TEST DATA**')

display(df_test.head(3))
# shape of the datasets

print('Shape of training data:', df_train.shape)

print('Shape of validation data:', df_valid.shape)

print('Shape of test data:', df_test.shape)
# check the missing values

print("Check for missing values in Train dataset")

null_check=df_train.isnull().sum()

print(null_check)

print('\n')

print("Check for missing values in Validation dataset")

null_check=df_valid.isnull().sum()

print(null_check)

print('\n')

print("Check for missing values in Test dataset")

null_check=df_test.isnull().sum()

print(null_check)
fig, axes = plt.subplots(ncols=2, figsize=(16, 7), dpi=100)



temp = df_train.toxic.value_counts()

sns.barplot(temp.index, temp, ax=axes[0], palette='Dark2')



temp = df_valid.toxic.value_counts()

sns.barplot(temp.index, temp, ax=axes[1], palette='Dark2')





axes[0].set_ylabel('Count ')

axes[1].set_ylabel(' ')

axes[0].set_xticklabels(["Non-toxic (90.4%) [0's]", "toxic (9.6%) [1's]"])

axes[1].set_xticklabels(["Non-toxic (84.6%) [0's]", "toxic (15.4%) [1's]"])



axes[0].set_title('Target Distribution of Train Dataset', fontsize=13)

axes[1].set_title('Target Distribution of Valid Dataset', fontsize=13)



plt.tight_layout()

plt.show()
print('Toxic Comment')

print('\n')

print(df_train[df_train.toxic==1].iloc[3,1])
print('Non-toxic Comment')

print('\n')

print(df_train[df_train.toxic==0].iloc[7,1])
fig, axes = plt.subplots(ncols=3, figsize=(17, 7), dpi=100)



temp = df_valid.lang.value_counts()

sns.barplot(temp.index, temp, ax=axes[0], palette='Set1')



temp = df_test.lang.value_counts()

sns.barplot(temp.index, temp, ax=axes[1], palette='Set1')



sns.countplot(data=df_valid, x="lang", hue="toxic" ,ax=axes[2], palette='Set1')



axes[0].set_ylabel(' Count ')

axes[1].set_ylabel(' ')

axes[2].set_ylabel(' ')

axes[2].set_xlabel(' ')



axes[0].set_title('Language Distribution of Valid Dataset', fontsize=13)

axes[1].set_title('Language Distribution of Test Dataset', fontsize=13)

axes[2].set_title('Language Distribution by Taget of Valid dataset', fontsize=13)



plt.tight_layout()

plt.show()
non_toxic_mask=np.array(Image.open(urllib.request.urlopen(url='https://image.flaticon.com/icons/png/512/99/99665.png')))

#wordcloud for non-toxic comments

subset=df_train[df_train.toxic==0]

text=subset.comment_text.values

wc= WordCloud(background_color="black",max_words=2000,mask=non_toxic_mask,stopwords=STOPWORDS)

wc.generate(" ".join(text))

plt.figure(figsize=(20,10))

plt.axis("off")

plt.title("Words frequented in non-toxic Comments", fontsize=20)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()
stopwords = STOPWORDS

toxic_mask=np.array(Image.open(urllib.request.urlopen(url='https://cdn4.vectorstock.com/i/1000x1000/81/98/radiation-hazard-symbol-vector-23088198.jpg')))

#wordcloud for toxic comments

subset=df_train[df_train.toxic==1]

text=subset.comment_text.values

wc= WordCloud(background_color="black",max_words=2000,mask=toxic_mask,stopwords=stopwords)

wc.generate(" ".join(text))

plt.figure(figsize=(20,10))

plt.axis("off")

plt.title("Words frequented in toxic Comments", fontsize=20)

plt.imshow(wc.recolor(colormap= 'Reds' , random_state=17), alpha=0.98)

plt.show()
#source: https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove

def get_top_bigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



def get_top_threegrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
#plt.style.use('ggplot')

nt_comment = df_train[df_train.toxic==0]["comment_text"]

t_comment = df_train[df_train.toxic==1]["comment_text"]

fig, axes = plt.subplots(2, 2, figsize=(18, 20), dpi=100)

           

top_unigrams=get_top_bigrams(nt_comment)[:20]

x,y=map(list,zip(*top_unigrams))

sns.barplot(x=y,y=x, ax=axes[0,0], color='c')





top_bigrams=get_top_bigrams(t_comment)[:20]

x,y=map(list,zip(*top_bigrams))

sns.barplot(x=y,y=x, ax=axes[0,1], color='red')



top_threegrams=get_top_threegrams(nt_comment)[:20]

x,y=map(list,zip(*top_threegrams))

sns.barplot(x=y,y=x, ax=axes[1, 0], color='c')



top_fourgrams=get_top_threegrams(t_comment)[:20]

x,y=map(list,zip(*top_fourgrams))

sns.barplot(x=y,y=x, ax=axes[1, 1], color='red')





axes[0, 0].set_ylabel(' ')

axes[0, 1].set_ylabel(' ')

axes[1, 0].set_ylabel(' ')

axes[1, 1].set_ylabel(' ')



axes[0, 0].yaxis.set_tick_params(labelsize=15)

axes[0, 1].yaxis.set_tick_params(labelsize=15)

axes[1, 0].yaxis.set_tick_params(labelsize=15)

axes[1, 1].yaxis.set_tick_params(labelsize=15)



axes[0, 0].set_title('Top 20 most common bigrams in Non-toxic', fontsize=15)

axes[0, 1].set_title('Top 20 most common bigrams in toxic', fontsize=15)

axes[1, 0].set_title('Top 20 most common threegrams in Non-toxic', fontsize=15)

axes[1, 1].set_title('Top 20 most common threegrams in toxic', fontsize=15)



plt.tight_layout()

plt.show()
# Contraction replacement patterns

#https://www.pythonforbeginners.com/regex/regular-expressions-in-python

cont_patterns = [

    (b'(W|w)on\'t', b'will not'),

    (b'(C|c)an\'t', b'can not'),

    (b'(I|i)\'m', b'i am'),

    (b'(A|a)in\'t', b'is not'),

    (b'(\w+)\'ll', b'\g<1> will'),

    (b'(\w+)n\'t', b'\g<1> not'),

    (b'(\w+)\'ve', b'\g<1> have'),

    (b'(\w+)\'s', b'\g<1> is'),

    (b'(\w+)\'re', b'\g<1> are'),

    (b'(\w+)\'d', b'\g<1> would'),

]

patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]



def count_regexp_occ(regexp="", text=None):

    """ Simple way to get the number of occurence of a regex"""

    return len(re.findall(regexp, text))
def prepare_for_char_n_gram(text):

    """ Simple text clean up process"""

    # 1. Go to lower case (only good for english)

    # Go to bytes_strings as I had issues removing all \n in r""

    clean = bytes(text.lower(), encoding="utf-8")

    # 2. Drop \n and  \t

    clean = clean.replace(b"\n", b" ")

    clean = clean.replace(b"\t", b" ")

    clean = clean.replace(b"\b", b" ")

    clean = clean.replace(b"\r", b" ")

    # 3. Replace english contractions

    for (pattern, repl) in patterns:

        clean = re.sub(pattern, repl, clean)

    # 4. Drop puntuation

    # I could have used regex package with regex.sub(b"\p{P}", " ")

    exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))

    clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])

    # 5. Drop numbers - as a scientist I don't think numbers are toxic ;-)

    clean = re.sub(b"\d+", b" ", clean)

    # 6. Remove extra spaces - At the end of previous operations we multiplied space accurences

    clean = re.sub(b'\s+', b' ', clean)

    # Remove ending space if any

    clean = re.sub(b'\s+$', b'', clean)

    # 7. Now replace words by words surrounded by # signs

    # e.g. my name is bond would become #my# #name# #is# #bond#

    #clean = re.sub(b"([a-z]+)", b"#\g<1>#", clean)

    clean = re.sub(b" ", b"# #", clean)  # Replace space

    clean = b"#" + clean + b"#"  # add leading and trailing #



    return str(clean, 'utf-8')
def get_indicators_and_clean_comments(df):

    """

    Check all sorts of content as it may help find toxic comment

    Though I'm not sure all of them improve scores

    """

    # Count number of \n

    df["ant_slash_n"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\n", x))

    # Get length in words and characters

    df["raw_word_len"] = df["comment_text"].apply(lambda x: len(x.split()))

    df["raw_char_len"] = df["comment_text"].apply(lambda x: len(x))

    # Check number of upper case, if you're angry you may write in upper case

    df["nb_upper"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[A-Z]", x))

    # Number of F words - f..k contains folk, fork,

    df["nb_fk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))

    # Number of S word

    df["nb_sk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))

    # Number of D words

    df["nb_dk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[dD]ick", x))

    # Number of occurence of You, insulting someone usually needs someone called : you

    df["nb_you"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))

    # Just to check you really refered to my mother ;-)

    df["nb_mother"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wmother\W", x))

    # Just checking for toxic 19th century vocabulary

    df["nb_ng"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))

    # Some Sentences start with a <:> so it may help

    df["start_with_columns"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"^\:+", x))

    # Check for time stamp

    df["has_timestamp"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))

    # Check for dates 18:44, 8 December 2010

    df["has_date_long"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))

    # Check for date short 8 December 2010

    df["has_date_short"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))

    # Check for http links

    df["has_http"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))

    # check for mail

    df["has_mail"] = df["comment_text"].apply(

        lambda x: count_regexp_occ(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x)

    )

    # Looking for words surrounded by == word == or """" word """"

    df["has_emphasize_equal"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\={2}.+\={2}", x))

    df["has_emphasize_quotes"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x))



    # Now clean comments

    df["clean_comment"] = df["comment_text"].apply(lambda x: prepare_for_char_n_gram(x))



    # Get the new length in words and characters

    df["clean_word_len"] = df["clean_comment"].apply(lambda x: len(x.split()))

    df["clean_char_len"] = df["clean_comment"].apply(lambda x: len(x))

    # Number of different characters used in a comment

    # Using the f word only will reduce the number of letters required in the comment

    df["clean_chars"] = df["clean_comment"].apply(lambda x: len(set(x)))

    df["clean_chars_ratio"] = df["clean_comment"].apply(lambda x: len(set(x))) / df["clean_comment"].apply(

        lambda x: 1 + min(99, len(x)))
# Dropinf the unwanted columns and rename text column

keep_cols = ['id', 'comment_text', 'toxic']

df_train = df_train[keep_cols]

df_test.rename(columns = {"content": "comment_text"}, inplace = True)
%%time

# performing the function

get_indicators_and_clean_comments(df_train)

get_indicators_and_clean_comments(df_test)
'''Function to plot histogram'''

def histogram_plot(x1, x2, title, end,size):

    trace1 = go.Histogram(x = x1,

                        name ='Non-Toxic', 

                        xbins = dict(end=end,size=size),

                        marker = dict(color = '#1bd902'))

    trace2 = go.Histogram(x = x2,

                        name='Toxic',

                        xbins = dict(end=end,size=size),

                        marker = dict(color = '#ff0307'))

    layout = go.Layout(barmode='stack', 

                       title = title, 

                       width=600, height=500,

                       template="ggplot2",

                       yaxis = dict(title = "Count"),

                       xaxis = dict(title = "Value"),

                       font=dict(family="Arial, Balto, Courier New, Droid Sans",

                                 color='black'))

    fig = go.Figure(data = [trace1, trace2], layout = layout,)

    return iplot(fig)
nt = df_train[df_train.toxic==0]['ant_slash_n']

t  = df_train[df_train.toxic==1]['ant_slash_n']

histogram_plot(nt, t, "Count of Newline", 30, 1)
nt = df_train[df_train.toxic==0]['raw_word_len']

t  = df_train[df_train.toxic==1]['raw_word_len']

histogram_plot(nt, t, "Count of Word length", 400, 5)
nt = df_train[df_train.toxic==0]['raw_char_len']

t  = df_train[df_train.toxic==1]['raw_char_len']

histogram_plot(nt, t, "Count of Character length", 1000, 10)
nt = df_train[df_train.toxic==0]['nb_upper']

t  = df_train[df_train.toxic==1]['nb_upper']

histogram_plot(nt, t, "Count of Upper Case Length", 100, 2)
nt = df_train[df_train.toxic==0]['nb_fk']

t  = df_train[df_train.toxic==1]['nb_fk']

histogram_plot(nt, t, "Count of F**K word contains folk, fork", 10, 1)
nt = df_train[df_train.toxic==0]['nb_sk']

t  = df_train[df_train.toxic==1]['nb_sk']

histogram_plot(nt, t, "Count of S**K word", 10, 1)
nt = df_train[df_train.toxic==0]['nb_dk']

t  = df_train[df_train.toxic==1]['nb_dk']

histogram_plot(nt, t, "Count of D**K word", 10, 1)
nt = df_train[df_train.toxic==0]['nb_you']

t  = df_train[df_train.toxic==1]['nb_you']

histogram_plot(nt, t, "Number of Occurence of You", 10, 1)
nt = df_train[df_train.toxic==0]['nb_mother']

t  = df_train[df_train.toxic==1]['nb_mother']

histogram_plot(nt, t, "Number of Occurence of Mother", 10, 1)
nt = df_train[df_train.toxic==0]['nb_ng']

t  = df_train[df_train.toxic==1]['nb_ng']

histogram_plot(nt, t, "Number of Occurence of Nigger", 10, 1)
nt = df_train[df_train.toxic==0]['has_timestamp']

t  = df_train[df_train.toxic==1]['has_timestamp']

histogram_plot(nt, t, "Number of Occurence of Time Stamp", 15, 1)
nt = df_train[df_train.toxic==0]['has_date_long']

t  = df_train[df_train.toxic==1]['has_date_long']

histogram_plot(nt, t, "Number of Occurence of Date Long", 10, 1)
nt = df_train[df_train.toxic==0]['has_http']

t  = df_train[df_train.toxic==1]['has_http']

histogram_plot(nt, t, "Number of Occurence of http Links", 10, 1)
nt = df_train[df_train.toxic==0]['has_mail']

t  = df_train[df_train.toxic==1]['has_mail']

histogram_plot(nt, t, "Number of Occurence of Mail", 10, 1)
nt = df_train[df_train.toxic==0]['clean_word_len']

t  = df_train[df_train.toxic==1]['clean_word_len']

histogram_plot(nt, t, "Word Lenght After Cleaning", 800, 10)
nt = df_train[df_train.toxic==0]['clean_char_len']

t  = df_train[df_train.toxic==1]['clean_char_len']

histogram_plot(nt, t, "Character Lenght After Cleaning", 3000, 30)
# correlation between meta featrue

num_features = [f for f in df_train.columns

                        if f not in ["comment_text", "clean_comment", "id", "toxic"]]

corr = df_train[num_features].corr()

sns.set_style("white")

plt.rcParams['figure.figsize'] = (20,15)

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask=mask, cmap='RdYlGn',square=True, linewidths=.5,annot=True)

plt.title("Correlation Between Meta Featrue", fontsize=25)

plt.show()
%%time

#Scaling numerical features

skl = MinMaxScaler()

train_num_features = csr_matrix(skl.fit_transform(df_train[num_features]))

test_num_features = csr_matrix(skl.fit_transform(df_test[num_features]))



#Get TF-IDF features

train_text = df_train['comment_text']

test_text = df_test['comment_text']

all_text = pd.concat([train_text, test_text])



 #  On real words

word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 2),

    max_features=20000)

word_vectorizer.fit(all_text)

train_word_features = word_vectorizer.transform(train_text)

test_word_features = word_vectorizer.transform(test_text)



del word_vectorizer

gc.collect()



# On character

char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='char',

    stop_words='english',

    ngram_range=(2, 6),

    max_features=50000)

char_vectorizer.fit(all_text)

train_char_features = char_vectorizer.transform(train_text)

test_char_features = char_vectorizer.transform(test_text)



del train_text

del test_text

del all_text

del char_vectorizer

gc.collect()





# Now stack TF IDF matrices

train_features = hstack([train_char_features, 

                         train_word_features, 

                         train_num_features]).tocsr()



del train_char_features

del train_word_features

gc.collect()



test_features = hstack([test_char_features, 

                        test_word_features, 

                        test_num_features]).tocsr()

del test_char_features 

del test_word_features,

gc.collect()
%%time



target =  df_train["toxic"]



# Model

def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):

    # Stratified k-fold 

    kf = StratifiedKFold(n_splits=5)

    fold_splits = kf.split(train, target)

    cv_scores = []

    pred_full_test = 0

    pred_train = np.zeros((train.shape[0]))

    i = 1

    for dev_index, val_index in fold_splits:

        print('Started ' + label + ' fold ' + str(i) + '/5')

        dev_X, val_X = train[dev_index], train[val_index]

        dev_y, val_y = target[dev_index], target[val_index]

        params2 = params.copy()

        pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, test, params2)

        pred_full_test = pred_full_test + pred_test_y

        pred_train[val_index] = pred_val_y

        if eval_fn is not None:

            cv_score = eval_fn(val_y, pred_val_y)

            cv_scores.append(cv_score)

            print(label + ' cv score {}: {}'.format(i, cv_score))

        i += 1

    print('{} cv scores : {}'.format(label, cv_scores))

    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))

    print('{} cv std score : {}'.format(label, np.std(cv_scores)))

    pred_full_test = pred_full_test / 5.0

    results = {'label': label,

              'train': pred_train, 'test': pred_full_test,

              'cv': cv_scores}

    return results





def runLR(train_X, train_y, test_X, test_y, test_X2, params):

    print('Train LR')

    model = LogisticRegression(**params)

    model.fit(train_X, train_y)

    print('Predict 1/2')

    pred_test_y = model.predict_proba(test_X)[:, 1]

    print('Predict 2/2')

    pred_test_y2 = model.predict_proba(test_X2)[:, 1]

    return pred_test_y, pred_test_y2





lr_params = {'solver': 'sag', 'C':  0.1, 'max_iter': 1000}

results = run_cv_model(train_features, test_features, target, runLR, lr_params, auc, 'lr')
%%time

sub_id = sub['id']

submission = pd.DataFrame({'id': sub_id, 'toxic': results['test']})

submission.to_csv('submission.csv', index=False)