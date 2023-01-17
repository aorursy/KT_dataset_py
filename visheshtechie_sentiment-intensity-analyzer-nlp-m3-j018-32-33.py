import pandas as pd

import scipy.io

from array import *

import numpy as np

import re

import nltk

from nltk.corpus import wordnet,stopwords

from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer

lem=WordNetLemmatizer()

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from PIL import Image

from wordcloud import WordCloud

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="whitegrid")

import collections
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
print('Training Data shape is:', train.shape)

print('Testing Data shape is:',test.shape)

train.head()
#Function to get unique list elements

def unique_list(l):

    ulist = []

    [ulist.append(x) for x in l if x not in ulist]

    return ulist



#Function to obtain Jaccard Index. It is a statistic used in understanding the similarities between 2 texts.

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



#Function to make wordcloud    

def MakeCloud(array , title = 'Word Cloud' , w = 16 , h = 13, my_mask='null',my_colormap='null',border_size=0):

    plt.figure(figsize=(w,h))

    if my_mask is not 'null':

        wc = WordCloud(background_color="black",stopwords=STOPWORDS, max_words=10000, 

                       max_font_size= 40, mask = my_mask,contour_width=border_size, contour_color='white')

    else:

        wc = WordCloud(background_color="black",stopwords=STOPWORDS, max_words=10000, 

                       max_font_size= 40, contour_width=border_size, contour_color='white')

    wc.generate(" ".join(array))

    if my_colormap is not 'null':

        plt.imshow(wc.recolor( colormap= my_colormap , random_state=17), alpha=0.98)

    else:

        plt.imshow(wc)

    plt.axis("off")

    plt.title(title)

    plt.show()



#Function that tags our tokens for lemmatizing

def get_wordnet_pos(word):

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)



#Function that pipelines and prepares data for Prediction Models

def pipeline(text):

    text = str(text).strip()

    text = re.sub(r'http\S+', '', text)

    stop_free = ' '.join([word for word in text.lower().split() if ((word not in STOPWORDS))])

    punc_free=re.sub('[^a-zA-Z]', " ", str(stop_free))

    text = ' '.join(lem.lemmatize(word, get_wordnet_pos(word)) for word in nltk.word_tokenize(punc_free))

    return text



#Function to create a bar plot with 2 axis representing 2 kinds of data

def BPlot(feature_1,feature_2,dataframe) :

    sns.barplot(x=feature_1, y=feature_2 , data=dataframe)

    

#EDA function to get wordcount of a column in dataset

def CommonWords(text ,show = True , kk=10) : 

    all_words = []



    for i in range(text.shape[0]) : 

        this_phrase = list(text)[i]

        for word in this_phrase.split() : 

            all_words.append(word)

    common_words = collections.Counter(all_words).most_common()

    k=0

    word_list =[]

    for word, i in common_words : 

        if not word.lower() in  STOPWORDS :

            if show : 

                print(f'The word    {word}   is repeated   {i}  times')

            word_list.append(word)

            k+=1

        if k==kk : 

            break

            

    return word_list
#Function to obtain words with maximum polarity

def choosing_selectedword(df_process):

    train_data = df_process['text']

    train_data_sentiment = df_process['sentiment']

    #This list will contain our predictions for the dataframe

    selected_text_processed = []

    #Initializing our Sentiment Analyzer

    analyser = SentimentIntensityAnalyzer()

    for j in range(0 , len(train_data)):

        #Removing hyperlink from tweets

        text = re.sub(r'http\S+', '', str(train_data.iloc[j]))

        #If the sentiment is neutral, we return the text as it is. Because the words labelled neutral will have low polarity

        if(train_data_sentiment.iloc[j] == "neutral" or len(text.split()) < 2):

            selected_text_processed.append(str(text))

        if(train_data_sentiment.iloc[j] == "positive" and len(text.split()) >= 2):

            #Tokenizing the text

            aa = re.split(' ', text)

            #This string will contain our high polarity words of each text

            ss_arr = ""

            polar = 0

            #Looking for high polarity tokens

            for qa in range(0,len(aa)):

                score = analyser.polarity_scores(aa[qa])

                if score['compound'] >polar:

                    polar = score['compound']

                    ss_arr = aa[qa]

            #If we find high polarity words, we return the ss_arr string containing high polarity words, else we return the initial text

            if len(ss_arr) != 0:

                selected_text_processed.append(ss_arr)   

            if len(ss_arr) == 0:

                selected_text_processed.append(text)

        if(train_data_sentiment.iloc[j] == "negative"and len(text.split()) >= 2):

            #Tokenizing the text

            aa = re.split(' ', text)

            #This string will contain our high polarity words of each text

            ss_arr = ""

            polar = 0

            #Looking for high polarity tokens

            for qa in range(0,len(aa)):

                score = analyser.polarity_scores(aa[qa])

                if score['compound'] <polar:

                    polar = score['compound']

                    ss_arr = aa[qa]

            #If we find high polarity words, we return the ss_arr string containing high polarity words, else we return the initial text

            if len(ss_arr) != 0:

                selected_text_processed.append(ss_arr)   

            if len(ss_arr) == 0:

                selected_text_processed.append(text)  

    return selected_text_processed
#Pipelining

train['pipelined_text']=train['text'].apply(pipeline)

train.head()
#Seeing distribution of rows on the basis of sentiment

temp = train.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)

temp.style.background_gradient(cmap='Blues')
#Seeing graph of rows v/s sentiment

BPlot(train['sentiment'].value_counts().index , train['sentiment'].value_counts().values,train)
#Seeing the word frequency in the texts of dataset

AllCommon = CommonWords(train['pipelined_text'])
#Importing images that will act as masks for our wordclouds

pos = np.array(Image.open('/kaggle/input/tweet-sentiment-extraction-masks/happy.jpg'))

neu = np.array(Image.open('/kaggle/input/tweet-sentiment-extraction-masks/neutral.jpg'))

neg = np.array(Image.open('/kaggle/input/tweet-sentiment-extraction-masks/sad.jpg'))
#Creating 3 arrays which contain all texts from each author respectively. 

#They are used further for wordcloud represnting each author.

positive = train[train.sentiment=="positive"]["pipelined_text"].values

negative = train[train.sentiment=="negative"]["pipelined_text"].values

neutral = train[train.sentiment=="neutral"]["pipelined_text"].values
#Creating a wordcloud for Positive Sentiment.

MakeCloud(positive,title="WordCloud for Sentiment : Positive",my_mask=pos,my_colormap='Reds')
#Creating a wordcloud for Neutral Sentiment.

MakeCloud(neutral,title="WordCloud for Sentiment : Neutral",my_mask=neu,my_colormap='Greys')
#Creating a wordcloud for Negative Sentiment.

MakeCloud(negative,title="WordCloud for Sentiment : Negative",my_mask=neg,my_colormap='Blues')
#Fetching predictions for train data

selected_text_train = choosing_selectedword(train)
#Checking accuracy of our Sentiment Analyser on Training Set

train_selected_data = train['selected_text']

average = 0;

#Fetching Jaccard Scores

for i in range(0,len(train_selected_data)):

    ja_s = jaccard(str(selected_text_train[i]),str(train_selected_data[i]))

    average = ja_s+average

print('Training Data accuracey')

print(average/len(selected_text_train))
#Fetching predictions for test data

selected_text_test = choosing_selectedword(test)
#Preparing dataframe in the form of sample submissions

test_textid = test['textID']

text_id_list = []

for kk in range(0,len(test_textid)):

    text_id_list.append(test_textid.iloc[kk])

final_result = pd.DataFrame({'textID':text_id_list,'selected_text':selected_text_test})

final_result.head()
#Creating submission file from dataframe

final_result.to_csv('submission.csv',index=False)