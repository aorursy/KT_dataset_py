# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

from matplotlib import rcParams

import nltk

from nltk.corpus import stopwords

import string

stop = stopwords.words('english')

punctuation = list(string.punctuation)

stop.append(punctuation)

from wordcloud import WordCloud

pd.set_option('display.max_colwidth', -1)

import matplotlib.pyplot as plt

import re

from nltk.stem import WordNetLemmatizer

from nltk import pos_tag

from nltk.corpus import wordnet

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv(filepath_or_buffer="/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv")
df.head()
df.isna().sum()
rcParams["figure.figsize"] = 15,20

df["author"].value_counts()[1:20].plot(kind="bar")
df["over_18"].value_counts()
df.over_18.replace(True,1,inplace = True)

df.over_18.replace(False,0,inplace = True)
rcParams["figure.figsize"] = 10,10

df["over_18"].value_counts().plot(kind="pie")
train_false = df[df.over_18 == 0.0].title

train_true = df[df.over_18 == 1.0].title
def tokenizeandstopwords(text):

    tokens = nltk.word_tokenize(text)

    # taken only words (not punctuation)

    token_words = [w for w in tokens if w.isalpha()]

    meaningful_words = [w for w in token_words if not w in stop]

    joined_words = ( " ".join(meaningful_words))

    return joined_words
def generate_word_cloud(text):

    wordcloud = WordCloud(

        width = 3000,

        height = 2000,

        max_words=3000,min_font_size=4,

        background_color = 'black').generate(str(text))

    fig = plt.figure(

        figsize = (40, 30),

        facecolor = 'k',

        edgecolor = 'k')

    plt.imshow(wordcloud, interpolation = 'bilinear')

    plt.axis('off')

    plt.tight_layout(pad=0)

    plt.show()
true_pre_processed = train_true[:1000].apply(tokenizeandstopwords)
generate_word_cloud(true_pre_processed)
false_pre_processed = train_false[:1000].apply(tokenizeandstopwords)

generate_word_cloud(false_pre_processed)
true_bigrams_series = (pd.Series(nltk.ngrams(true_pre_processed, 2)).value_counts())[:20]
true_bigrams_series.sort_values().plot.barh(color='blue', width=.9, figsize=(15, 15))

plt.title('20 Most Frequently Occuring Bigrams')

plt.ylabel('Bigram')

plt.xlabel('# of Occurances')
df["text"] = df["title"] + ' ' + df['author']
df = df.drop(["id","author","author_flair_text","removed_by","created_utc","awarders","full_link","title"],axis=1)
df.shape

df.text.fillna(" ",inplace = True)
def remove_emoji(string):

    emoji_pattern = re.compile("["

                               u"\U0001F600-\U0001F64F"  # emoticons

                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                               u"\U0001F680-\U0001F6FF"  # transport & map symbols

                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                               u"\U00002500-\U00002BEF"  # chinese char

                               u"\U00002702-\U000027B0"

                               u"\U00002702-\U000027B0"

                               u"\U000024C2-\U0001F251"

                               u"\U0001f926-\U0001f937"

                               u"\U00010000-\U0010ffff"

                               u"\u2640-\u2642"

                               u"\u2600-\u2B55"

                               u"\u200d"

                               u"\u23cf"

                               u"\u23e9"

                               u"\u231a"

                               u"\ufe0f"  # dingbats

                               u"\u3030"

                               "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', string)
df["text"] = df["text"].apply(remove_emoji)

df.head()
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):

    final_text = []

    for i in text.split():

        if i.strip().lower() not in stop:

            pos = pos_tag([i.strip()])

            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))

            final_text.append(word.lower())

    return final_text        
def get_simple_pos(tag):

    if tag.startswith('J'):

        return wordnet.ADJ

    elif tag.startswith('V'):

        return wordnet.VERB

    elif tag.startswith('N'):

        return wordnet.NOUN

    elif tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN
def join_text(text):

    string = ''

    for i in text:

        string += i.strip() +' '

    return string  
df.text = df.text.apply(lemmatize_words)

df.text = df.text.apply(join_text)
df.head()
# rcParams['figure.figsize'] = 15,10

# sns.countplot(x=df["text"],hue=df["over_18"])
df["over_18"].value_counts()
train_message = df.text[:150000]

test_message = df.text[150000:]

train_category = df.over_18[:150000]

test_category = df.over_18[150000:]
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,2))

#transformed train reviews

cv_train_reviews=cv.fit_transform(train_message)

#transformed test reviews

cv_test_reviews=cv.transform(test_message)
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,2))

#transformed train reviews

tv_train_reviews=tv.fit_transform(train_message)

#transformed test reviews

tv_test_reviews=tv.transform(test_message)

print('Tfidf_train:',tv_train_reviews.shape)

print('Tfidf_test:',tv_test_reviews.shape)
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)

#Fitting the model for Bag of words

lr_bow=lr.fit(cv_train_reviews,train_category)

print(lr_bow)

#Fitting the model for tfidf features

lr_tfidf=lr.fit(tv_train_reviews,train_category)

print(lr_tfidf)
#Predicting the model for bag of words

lr_bow_predict=lr.predict(cv_test_reviews)

##Predicting the model for tfidf features

lr_tfidf_predict=lr.predict(tv_test_reviews)





#Accuracy score for bag of words

lr_bow_score=accuracy_score(test_category,lr_bow_predict)

print("lr_bow_score :",lr_bow_score)

#Accuracy score for tfidf features

lr_tfidf_score=accuracy_score(test_category,lr_tfidf_predict)

print("lr_tfidf_score :",lr_tfidf_score)







#Classification report for bag of words

lr_bow_report=classification_report(test_category,lr_bow_predict,target_names=['0','1'])

print(lr_bow_report)



#Classification report for tfidf features

lr_tfidf_report=classification_report(test_category,lr_tfidf_predict,target_names=['0','1'])

print(lr_tfidf_report)



plot_confusion_matrix(lr_bow, cv_test_reviews, test_category,display_labels=['0','1'],cmap="Blues",values_format = '')

plot_confusion_matrix(lr_tfidf, tv_test_reviews, test_category,display_labels=['0','1'],cmap="Blues",values_format = '')
