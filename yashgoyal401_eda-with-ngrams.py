import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context="notebook", style="darkgrid", palette="deep", font="sans-serif", font_scale=1, color_codes=True)

import nltk

from nltk import word_tokenize

from nltk import ngrams

import warnings

warnings.filterwarnings("ignore")

Train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

Test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

Sub = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
print(Train.head(1))
print(Test.head(2))
print(Sub.head(1))
### Example of Jaccard score

def jac(str1,str2):

    a = set(str1.lower().split())

    b= set(str2.lower().split())

    c = a.intersection(b)

    

    d = float(len(c))/(len(a)+len(b)-len(c))

    return d



Sentence_1 = 'Life well spent is life good'

Sentence_2 = 'Life is an art and it is good so far'

Sentence_3 = 'Life is good'



print(jac(Sentence_1,Sentence_2))

print(jac(Sentence_1,Sentence_3))

#### EDA

print(Train.shape)

print(Test.shape)
Train.isnull().sum()
Train  = Train.dropna()

sns.heatmap(Train.isnull(),cmap="viridis")
Train["sentiment"].value_counts()
Train["sentiment"].value_counts(normalize=True)
x = Train["sentiment"].value_counts(normalize=True)

plt.pie(x,labels=x.index,autopct='%1.1f%%',explode = (0, 0.1,0.1))

plt.title("Sentiment distribution")

sns.countplot(x="sentiment",data=Train,order=Train["sentiment"].value_counts().index)
sns.countplot(x="sentiment",data=Test,order=Test["sentiment"].value_counts().index)

import re

import string

import nltk

from nltk.corpus import stopwords

def clean_text(text):

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



def text_preprocessing(text):

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    nopunc = clean_text(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(remove_stopwords)

    return combined_text
Train["text_clean"] = Train["text"].astype(str).apply(lambda x: text_preprocessing(x))

Test["text_clean"] = Test["text"].astype(str).apply(lambda x: text_preprocessing(x))

print("original_text :",Train["text"].values[1])

print("cleaned_text :",Train["text_clean"].values[1])

Train["clean_sl_tx"] = Train["selected_text"].astype(str).apply(lambda x: text_preprocessing(x))



print("Original_selected_text :", Train["selected_text"].values[2])

print("cleaned_selected_text :", Train["clean_sl_tx"].values[2])

Train["text_len"] = Train["text_clean"].astype(str).apply(len)

Train["text_word_count"] = Train['text_clean'].apply(lambda x: len(str(x).split()))  

print(Train["text_len"][1])

print(Train["text_word_count"][1])
Train["sl_text_len"]  = Train["clean_sl_tx"].astype(str).apply(len)

Train["sl_text_word_count"] = Train['clean_sl_tx'].apply(lambda x: len(str(x).split())) 

print(Train["sl_text_len"][1])

print(Train["sl_text_word_count"][1])
Train["difference of length"]  = Train["text_len"] - Train["sl_text_len"]

Train["Difference of word count"] = Train["text_word_count"] - Train["sl_text_word_count"]

print(Train["difference of length"][1])

print(Train["Difference of word count"][1])
Train.columns
Train.describe()
print(Train.groupby("sentiment").count()["text"])
positive_data = Train[Train["sentiment"]=="positive"]

negative_data = Train[Train["sentiment"]=="negative"]

neutral_data = Train[Train["sentiment"]=="neutral"]

## text length analysis of training data

fig = plt.figure(1, figsize=(10, 10))

plt.hist(positive_data["text_len"],bins=50,color="red")

plt.title("Positive text length distribution")

plt.xlabel("Text_length")

plt.ylabel("Count")
fig = plt.figure(1, figsize=(10, 10))

plt.hist(negative_data["text_len"],bins=50,color="green")

plt.title("negative_data text length distribution")

plt.xlabel("Text_length")

plt.ylabel("Count")
fig = plt.figure(1, figsize=(10, 10))

plt.hist(neutral_data["text_len"],bins=50,color="blue")

plt.title("neutral_data text length distribution")

plt.xlabel("Text_length")

plt.ylabel("Count")
kwargs = dict(hist_kws={'alpha':.4}, kde_kws={'linewidth':5})

plt.figure(figsize=(10,7), dpi= 80)

sns.distplot(positive_data["text_len"], color="dodgerblue", label="Positive", **kwargs)

sns.distplot(negative_data["text_len"], color="orange", label="Negative", **kwargs)

sns.distplot(neutral_data["text_len"], color="deeppink", label="Neutral", **kwargs)

plt.xlim(0,120)

plt.legend()

## Text word count analysis

fig = plt.figure(1, figsize=(10, 10))

plt.hist(positive_data["text_word_count"],bins=20,color="red")

plt.title("Positive word count distribution")

plt.xlabel("word count")

plt.ylabel("Count")

fig = plt.figure(1, figsize=(10, 10))

plt.hist(negative_data["text_word_count"],bins=20,color="green")

plt.title("negative_data word count distribution")

plt.xlabel("word count")

plt.ylabel("Count")

fig = plt.figure(1, figsize=(10, 10))

plt.hist(neutral_data["text_word_count"],bins=20,color="blue")

plt.title("neutral_data word count distribution")

plt.xlabel("word count")

plt.ylabel("Count")

kwargs = dict(hist_kws={'alpha':.4}, kde_kws={'linewidth':5})

plt.figure(figsize=(10,10), dpi= 80)

sns.distplot(positive_data["text_word_count"], color="dodgerblue", label="Positive", **kwargs)

sns.distplot(negative_data["text_word_count"], color="orange", label="Negative", **kwargs)

sns.distplot(neutral_data["text_word_count"], color="deeppink", label="Neutral", **kwargs)

plt.xlim(0,20)

plt.legend()
fig = plt.figure(1, figsize=(10, 10))

sns.boxplot(x="sentiment",y="text_word_count",data=Train)

plt.title("Word count of text")
fig = plt.figure(1, figsize=(10, 10))

sns.boxplot(x="sentiment",y="text_len",data=Train)

plt.title("text length")

## Function for plotting Top 50 words of each category

import heapq

from operator import itemgetter

from collections import Counter



def Top50(data,title=None):

    token_data= []

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    for i in data:

        l = tokenizer.tokenize(i)

        token_data.append(l)

    corpus = []

    for i in token_data:

       for j in i:

         corpus.append(j)

    c = Counter(corpus)

    Di = dict(c)

    TOp_50 = dict(heapq.nlargest(50, Di.items(), key=itemgetter(1)))

    dd = pd.DataFrame(TOp_50.items(),columns=["word","frequency"])

    fig = plt.figure(1, figsize=(15, 15))

    plt.bar(range(len(TOp_50)),TOp_50.values(),align='center')        

    plt.xticks(range(len(TOp_50)), list(TOp_50.keys()))

    plt.tick_params(axis="x",rotation=90) 

    if title==None:

        plt.title("Top 50 words")

    else:

        plt.title(title)

    return dd.head(10)
Top50(positive_data["text_clean"],title="Top 50 Positive words")
Top50(negative_data["text_clean"],title="Top 50 negative words")
Top50(neutral_data["text_clean"],title="Top 50 Neutral words")

Top50(Train["text_clean"],title="most common words in whole data")

## Function for plotting Ngrams

def Ngram(data,num,title=None):

    token_data= []

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    for i in data:

        n_grams = ngrams(tokenizer.tokenize(i), num)

        p = [ ' '.join(grams) for grams in  n_grams]

        token_data.append(p)

    corpus = []

    for i in token_data:

       for j in i:

         corpus.append(j)

    c = Counter(corpus)

    Di = dict(c)

    TOp_50 = dict(heapq.nlargest(50, Di.items(), key=itemgetter(1)))

    fig = plt.figure(1, figsize=(12, 12))

    plt.bar(range(len(TOp_50)),TOp_50.values(),align='center')        

    plt.xticks(range(len(TOp_50)), list(TOp_50.keys()))

    plt.tick_params(axis="x",rotation=90) 

    

    if title == None:

        plt.title("Ngram")

    else:

        plt.title(title)
## Biagram

Ngram(positive_data["text_clean"],2,title="Bigram of Positive Tweets")
Ngram(negative_data["text_clean"],2,title="Bigram of Negative Tweets")
Ngram(neutral_data["text_clean"],2,title="Bigram of Neutral Tweets")

## trigram

Ngram(positive_data["text_clean"],3,title="Trigram of Positive Tweets")
Ngram(negative_data["text_clean"],3,title="Trigram of Negative Tweets")
Ngram(neutral_data["text_clean"],3,title="Trigram of Neutral Tweets")
### Wordcloud

from wordcloud import WordCloud

def show_wordcloud(data,title=None):

    if title == None:

        fig = plt.figure(1, figsize=(12, 12))

        plt.title("Wordclud")

    else :

        fig = plt.figure(1, figsize=(12, 12))

        plt.title(title)

    wordcloud = WordCloud(background_color='white',max_font_size=60,max_words=2000, random_state=1,width=600,height=400).generate(str(data))

    plt.axis('off')

    plt.imshow(wordcloud,interpolation="bilinear")

    plt.show()
show_wordcloud(Train["text_clean"],title="whole data")
show_wordcloud(Train.loc[Train["sentiment"]=="positive","text_clean"],title="Positive wordcloud")
show_wordcloud(Train.loc[Train["sentiment"]=="negative","text_clean"],title="Negative wordcloud")

show_wordcloud(Train.loc[Train["sentiment"]=="neutral","text_clean"],title="Neutral wordcloud")

## Gonna deal with selected text

Top50(positive_data["clean_sl_tx"],title="most common positive words")
Top50(negative_data["clean_sl_tx"],title="most common negative words")
Top50(neutral_data["clean_sl_tx"],title="most common neutral words")
Top50(Train["clean_sl_tx"],title="most common selected words in whole data")
kwargs = dict(hist_kws={'alpha':.4}, kde_kws={'linewidth':5})

plt.figure(figsize=(10,7), dpi= 80)

sns.distplot(positive_data["sl_text_len"], color="dodgerblue", label="Positive", **kwargs)

sns.distplot(negative_data["sl_text_len"], color="orange", label="Negative", **kwargs)

sns.distplot(neutral_data["sl_text_len"], color="deeppink", label="Neutral", **kwargs)

plt.xlim(0,100)

plt.legend()

plt.title("selected text length distribution")

kwargs = dict(hist_kws={'alpha':.5}, kde_kws={'linewidth':3})

plt.figure(figsize=(10,7), dpi= 80)

sns.distplot(positive_data["sl_text_word_count"], color="dodgerblue", label="Positive", **kwargs)

sns.distplot(negative_data["sl_text_word_count"], color="orange", label="Negative", **kwargs)

sns.distplot(neutral_data["sl_text_word_count"], color="deeppink", label="Neutral", **kwargs)

plt.xlim(0,15)

plt.legend()

plt.title("Selected text word count distribution")

sns.kdeplot(Train["text_len"],shade=True,color="r")

sns.kdeplot(Train["sl_text_len"],shade=True,color="b")

plt.xlabel("length of text")

plt.title("length distribution")

sns.kdeplot(Train["text_word_count"],shade=True,color="r")

sns.kdeplot(Train["sl_text_word_count"],shade=True,color="b")

plt.xlabel("Word count")

plt.title("Word count distribution")
sns.boxplot(x="sentiment",y="sl_text_word_count",data=Train)

plt.title("Selected text word count")

sns.boxplot(x="sentiment",y="sl_text_len",data=Train)

plt.title("selected text length")

jaccard_score = []

for i,j in Train.iterrows():

    str1 = j.text

    str2 = j.selected_text

    

    JC_score = round(jac(str1,str2),2)

    jaccard_score.append(JC_score)



Train["Jaccard_score"] = jaccard_score
jc_0 = Train[Train["Jaccard_score"]==0]

jc_1 = Train[Train["Jaccard_score"]==1]

print(jc_0.shape)

print(jc_1.shape)