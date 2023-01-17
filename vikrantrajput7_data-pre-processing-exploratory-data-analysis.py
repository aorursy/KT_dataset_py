import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
df=pd.read_csv("train.csv")
df.drop("id",axis=1,inplace=True)      #Dropping the ID column
df.head(50)
print(df.info())
i=df['target'].isnull().sum()
print("Total {} null values and {} non-null values in Keyword feature".format(i,len(df)-i))
df['target'].value_counts()
i=df['keyword'].isnull().sum()
print("Total {} null values and {} non-null values in Keyword feature".format(i,len(df)-i))
# filling null values with empty keyword
df['keyword']=df['keyword'].fillna(value='empty')
uniq_key=df["keyword"].unique()
print("No of Unique Keywords in Keyword Feature : {} \n\n".format(len(uniq_key)))
print(uniq_key)

from nltk.stem import WordNetLemmatizer
lemm=WordNetLemmatizer()
kywd=[re.sub('%20',' ',str(i)) for i in df['keyword']]
final_kywd=[]
for i in range(len(kywd)):
    if len(kywd[i].split()) <2:
        x=lemm.lemmatize(kywd[i],'v')
        x=lemm.lemmatize(x,'n')
        final_kywd.append(x)
    else:
        splitted=[]
        for word in kywd[i].split():
            x=lemm.lemmatize(word,'v')
            x=lemm.lemmatize(x,'n')
            splitted.append(x)
        final_kywd.append(' '.join(splitted))
df['keyword']=final_kywd            
uniq_key=df["keyword"].unique()
print("No of Unique Keywords in Keyword Feature : {} \n\n".format(len(uniq_key)))
print(uniq_key)

freq=df['keyword'].value_counts()
x=[]
y=[]
for word in freq.keys():
    x.append(word)
    y.append(freq[word])
    print("{} : {}".format(word,freq[word]))
import matplotlib.pyplot as plt
plt.barh(x,y)
plt.xlabel("Document Frequency")
plt.ylabel("Keyword")
plt.show()
i=df['location'].isnull().sum()
print("Total {} null values and {} non-null values in location feature".format(i,len(df)-i))
uniq_key=df["location"].unique()
print("No of Unique places in Location Feature : {} \n\n".format(len(uniq_key)))
print(uniq_key)

df.drop('location',axis=1,inplace=True)
df.head(5)
i=df['text'].isnull().sum()
print("Total {} null values and {} non-null values in text feature".format(i,len(df)-i))
i=1
for line in df['text']:
    print("{} : {}".format(i,line))
    i+=1
contraction_mapping = {"that'll":"that will","ain’t": "is not", "aren’t": "are not","can’t": "cannot", "’cause": "because", "could’ve": "could have", "couldn’t": "could not",
                          "didn’t": "did not", "doesn’t": "does not", "don’t": "do not", "hadn’t": "had not", "hasn’t": "has not", "haven’t": "have not",
                          "he’d": "he would","he’ll": "he will", "he’s": "he is", "how’d": "how did", "how’d’y": "how do you", "how’ll": "how will", "how’s": "how is",
                          "I’d": "I would", "I’d’ve": "I would have", "I’ll": "I will", "I’ll’ve": "I will have","I’m": "I am", "I’ve": "I have", "i’d": "i would",
                          "i’d’ve": "i would have", "i’ll": "i will",  "i’ll’ve": "i will have","i’m": "i am", "i’ve": "i have", "isn’t": "is not", "it’d": "it would",
                          "it’d’ve": "it would have", "it’ll": "it will", "it’ll’ve": "it will have","it’s": "it is", "let’s": "let us", "ma’am": "madam",
                          "mayn’t": "may not", "might’ve": "might have","mightn’t": "might not","mightn’t’ve": "might not have", "must’ve": "must have",
                          "mustn’t": "must not", "mustn’t’ve": "must not have", "needn’t": "need not", "needn’t’ve": "need not have","o’clock": "of the clock",
                          "oughtn’t": "ought not", "oughtn’t’ve": "ought not have", "shan’t": "shall not", "sha’n’t": "shall not", "shan’t’ve": "shall not have",
                          "she’d": "she would", "she’d’ve": "she would have", "she’ll": "she will", "she’ll’ve": "she will have", "she’s": "she is",
                          "should’ve": "should have", "shouldn’t": "should not", "shouldn’t’ve": "should not have", "so’ve": "so have","so’s": "so as",
                          "this’s": "this is","that’d": "that would", "that’d’ve": "that would have", "that’s": "that is", "there’d": "there would",
                          "there’d’ve": "there would have", "there’s": "there is", "here’s": "here is","they’d": "they would", "they’d’ve": "they would have",
                          "they’ll": "they will", "they’ll’ve": "they will have", "they’re": "they are", "they’ve": "they have", "to’ve": "to have",
                          "wasn’t": "was not", "we’d": "we would", "we’d’ve": "we would have", "we’ll": "we will", "we’ll’ve": "we will have", "we’re": "we are",
                          "we’ve": "we have", "weren’t": "were not", "what’ll": "what will", "what’ll’ve": "what will have", "what’re": "what are",
                          "what’s": "what is", "what’ve": "what have", "when’s": "when is", "when’ve": "when have", "where’d": "where did", "where’s": "where is",
                          "where’ve": "where have", "who’ll": "who will", "who’ll’ve": "who will have", "who’s": "who is", "who’ve": "who have",
                          "why’s": "why is", "why’ve": "why have", "will’ve": "will have", "won’t": "will not", "won’t’ve": "will not have",
                          "would’ve": "would have", "wouldn’t": "would not", "wouldn’t’ve": "would not have", "y’all": "you all",
                          "y’all’d": "you all would","y’all’d’ve": "you all would have","y’all’re": "you all are","y’all’ve": "you all have",
                          "you’d": "you would", "you’d’ve": "you would have", "you’ll": "you will", "you’ll’ve": "you will have",
                          "you’re": "you are", "you’ve": "you have","ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                          "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                          "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                          "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                          "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                          "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                          "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                          "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                          "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                          "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                          "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                          "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                          "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                          "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                          "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                          "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                          "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                          "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                          "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                          "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                          "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                          "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                          "you're": "you are", "you've": "you have","n't":'not'}
def clean_data(document,contraction_mapping):
    clean_data=[]
    for line in tqdm(document):
        x=line.lower()              #lower-casing the words
        word_list=[]
        for word in x.split():      # contraction mapping
            if word in contraction_mapping:
                word_list.append(contraction_mapping[word])
            else:
                word_list.append(word)
        jnt=' '.join(word_list)
        x=re.sub("'s",' ',jnt)
        x=re.sub('\n',' ',x)
        x=re.sub(r'http.*?\s|http.*',' ',x)       #removing web urls
        x=re.sub(r'@.*?\s|@(.*)?',' ',x)          #removing tagged usernames
        x=re.sub(r'\(.*?\)',' ',x)
        x=re.sub(r'\[.*?\]',' ',x)
        x=re.sub(r'\{.*?\}',' ',x)
        x=re.sub(r'&.*;',' ',x)
        x=re.sub(r'[^a-z]',' ',x)
        x=re.sub(r'\s+',' ',x)
        
        
        clean_data.append(x.strip())
    return clean_data
i=1
tweets=clean_data(df['text'],contraction_mapping)
for line in tweets:
    print("{} : {}".format(i,line))
    i+=1
from nltk.corpus import stopwords
stopword=set(stopwords.words("english"))
cleaned_tweets=[]
for line in tweets:
    word_list=line.split()
    new_word_list=[]
    for word in word_list:
        if len(word) > 2:
            if word not in stopword:
                x=lemm.lemmatize(word,'v')
                x=lemm.lemmatize(x,'n')
                if len(x) > 2:
                    new_word_list.append(x)
    cleaned_tweets.append(' '.join(new_word_list))
df['text']=cleaned_tweets
df.head(5)
count=0
index=[]
for line in df['text']:
    if len(line.split()) > 3:
        index.append(count)
        count+=1
    else:
        count+=1

ndf=df.loc[index]
ndf.tail()
(ndf.target).value_counts()
words_list=[]
word_count=[]
for line in ndf['text']:
    word_count.append(len(line.split()))
    words_list.extend(line.split())
print("There are total {} unique words".format(len(set(words_list))))
plt.hist(word_count,bins=10)
plt.title('Word distribution of tweets')
plt.xlabel("Range of words")
plt.ylabel("No of tweets")
plt.show()
# Creating vocabulary
vocab=set(words_list)
word_count_dict={}
for word in vocab:
    word_count_dict[word]=words_list.count(word)
# sorting dictionary by words
import operator
sorted_word_count=dict(sorted(word_count_dict.items(), key=operator.itemgetter(1),reverse=True))
sorted_word_count
full_str=''
for line in ndf['text']:
    full_str+=line+' '
from wordcloud import WordCloud, STOPWORDS 
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                min_font_size = 1).generate(full_str) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 
ndf.to_csv('cleaned_train.csv',index=False)