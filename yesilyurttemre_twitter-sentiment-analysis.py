import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
%matplotlib inline
DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "text"]

dataset = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv', encoding="ISO-8859-1", names = DATASET_COLUMNS)
dataset.head()

dataset.info()
dataset['sentiment'].unique()
dataset = dataset[['sentiment', 'text']]
dataset['sentiment'] = dataset['sentiment'].replace(4,1)
dataset['sentiment'].unique()
"""
If stop words and word are not installed, comment lines can be downloaded by running.
"""
#stop_words = nltk.download('stopwords')
#word_net = nltk.download('wordnet')


emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

ps = PorterStemmer()

text, sentiment = list(dataset['text']), list(dataset['sentiment'])
def preprocess(data_text):
    processed_text = []
    
    word_lem = nltk.WordNetLemmatizer()
    
    url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    user_pattern = '@[^\s]+'
    alpha_pattern = "[^a-zA-Z0-9]"
    sequence_pattern = r"(.)\1\1+"
    seq_replace_pattern = r"\1\1"
    
    for tweet in data_text:
        tweet = tweet.lower()
        
        tweet = re.sub(url_pattern, ' ', tweet)
        
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
            
        tweet = re.sub(user_pattern, " ", tweet)
        
        tweet = re.sub(alpha_pattern, " ", tweet)

        tweet = re.sub(sequence_pattern, seq_replace_pattern, tweet)

        tweet_words = ''

        for word in tweet.split():
            if word not in nltk.corpus.stopwords.words('english'):
                if len(word) > 1:
                    word = word_lem.lemmatize(word)
                    tweet_words += (word + ' ')
        processed_text.append(tweet_words)
      
    return processed_text
t = time.time()
processed_text = preprocess(text)
print(f'Text Preprocessing complete.')
print(f'Time Taken: {round(time.time()-t)} seconds')
processed_text[0:25]
data_pos = processed_text[800000:]
wc = WordCloud(max_words = 300000,background_color ='white', width = 1920 , height = 1080,
              collocations=False).generate(" ".join(data_pos))
plt.figure(figsize = (40,40))
plt.imshow(wc)
data_pos = processed_text[:800000]
wc = WordCloud(max_words = 300000,background_color ='white', width = 1920 , height = 1080,
              collocations=False).generate(" ".join(data_pos))
plt.figure(figsize = (40,40))
plt.imshow(wc)
X_train, X_test, y_train, y_test = train_test_split(processed_text, sentiment, test_size = 0.05, random_state = 0)
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features = 500000)
vectoriser.fit(X_train)
X_train = vectoriser.transform(X_train)
X_test = vectoriser.transform(X_test)
def model_evaluate(model):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    
    categories = ['Negative', 'Positive']
    group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten() / np.sum(cm)] 
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cm, annot = labels, cmap = 'Blues',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
t = time.time()
model = LogisticRegression()
model.fit(X_train, y_train)
model_evaluate(model)
print(f'Logistic Regression complete.')
print(f'Time Taken: {round(time.time()-t)} seconds')
file = open('vectoriser-ngram-(1,2).pickle','wb')
pickle.dump(vectoriser, file)
file.close()

file = open('sentiment_logistic.pickle','wb')
pickle.dump(model, file)
file.close()
if __name__=="__main__":
    
    def load_models():

        file = open('vectoriser-ngram-(1,2).pickle', 'rb')
        vectoriser = pickle.load(file)
        file.close()

        file = open('sentiment_logistic.pickle', 'rb')
        log_model = pickle.load(file)
        file.close()

        return vectoriser, log_model
    
    def predict(vectoriser, model, text):

        textdata = vectoriser.transform(preprocess(text))
        sentiment = model.predict(textdata)

        data = []
        for text, pred in zip(text, sentiment):
            data.append((text,pred))

        df = pd.DataFrame(data, columns = ['text','sentiment'])
        df = df.replace([0,1], ["Negative","Positive"])
        return df

    vectoriser, log_model = load_models()
    
    text = ["Data science is a very enjoyable job.",
            "Twitter is unnecessary.",
            "I dont feel good."]
    
    df = predict(vectoriser, log_model, text)
    print(df.head())
