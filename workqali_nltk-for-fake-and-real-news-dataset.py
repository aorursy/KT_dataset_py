import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from collections import Counter
import nltk
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
#df = pd.read_pickle('../saved_files/cleaned_df.pkl')
df_fake = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')
df_true = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')
df_fake.head()
df_true.head()
plt.plot(param_C,df_acc1.mean_test_score.astype(float), marker = 'o')
Title = 'Model Accuracy vs C Parameter with l1 Regulation'
Xlab = "log10 of C"
Ylab = "Accuracy"
plt.title(Title)
plt.xlabel(Xlab)
plt.ylabel(Ylab)
plt.show()
df_true['label'] = 0
df_fake['label'] = 1
df = pd.concat([df_true, df_fake],axis=0)
df = df.sample(frac = 1).reset_index(drop=True)
df.head()
# Preparing the target and predictors for modeling
# Keep the title and body text separated for different models
#X_body_text = df['clean_text'].values
#X_title_text = df['clean_title'].values
#y = df['label'].values
np.unique(df['subject'],return_counts = True)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
plt.rcParams['figure.figsize'] = (14.0, 6.0)
plt.rcParams['font.family'] = "serif"
news_cat_count = sns.countplot(data=df, x = 'subject')
#news_cat_count.axes.set_title("Title",fontsize=30)
#news_cat_count.set_xlabel("Subject",fontsize=12)
news_cat_count.set_ylabel("Count",fontsize=12)
news_cat_count.tick_params(labelsize=12)
#sns.plt.show()
plt.rcParams['figure.figsize'] = (8.0, 4.0)
plt.rcParams['font.family'] = "serif"
news_true_cat_count = sns.countplot(data=df_true,x = 'subject')
plt.rcParams['figure.figsize'] = (9.0, 4.0)
plt.rcParams['font.family'] = "serif"
news_fake_cat_count = sns.countplot(data=df_fake, x = 'subject')
df[['title', 'label']].groupby('label').agg('count')
df['text_len']=df['text'].apply(len)
df.hist(column='text_len',bins=50,figsize=(8,5),grid=False)
df.hist(column='text_len',by='label',bins=50,figsize=(14,4))
import re 
# using regex (findall()) 
# to count words in string 
df['num_words'] = df['text'].apply(lambda x: len(re.findall(r'\w+', x)))
df.sort_values(by='num_words',ascending = False)['num_words'][:1000]
#df.sort_values(by='num_words',ascending = False)
df[df['label']==0].hist(column='num_words',bins=50,figsize=(8,6),xlabelsize=12, ylabelsize=12)
plt.title('Real News', fontdict=None, loc='center', pad=None, fontsize=16)
plt.xlabel("Number of Words", fontsize=14)
plt.ylabel("Number of Articles",fontsize=14)
plt.xlim([0,2000])
plt.ylim([0,6000])
#df.sort_values(by='num_words',ascending = False)
df[df['label']==1].hist(column='num_words',bins=80,figsize=(8,6),xlabelsize=12, ylabelsize=12)
plt.title('Fake News', fontdict=None, loc='center', pad=None, fontsize=16)
plt.xlabel("Number of Words", fontsize=14)
plt.ylabel("Number of Articles",fontsize=14)
plt.xlim([0,2000])
plt.ylim([0,6000])
df2 = df.copy()
df2.head()
news_text = df2['text']
print(news_text[1])
#use regular expression to replace some specific text
# replace email address
processed_text = news_text.str.replace(r'^([a-zA-Z0-9_\-\.]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([a-zA-Z0-9\-]+\.)+))([a-zA-Z]{2,4}|[0-9]{1,3})(\]?)$', 'email_address')

#replace 10 digit phone number
#processed_text = processed_text.str.replace(r'^[^0-9]*(?:(\d)[^0-9]*){10}$', 'phone_number')

#replace normal number with numbr
#processed_text = processed_text.str.replace(r'\d+(\.\d+)?', 'numbr')

#remove punctuation
processed_text = processed_text.str.replace(r'[^\w\d\s]', ' ')

#remove whitespace between terms with a single space
processed_text = processed_text.str.replace(r'\s+', ' ')

#remove leading and trailing whitespace
processed_text = processed_text.str.replace(r'^\s+|\s+?$', '')
print(processed_text[1])
processed_text = processed_text.str.lower()
print(processed_text[1])
#remove stopwords from text
#import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords
              
stop_words = set(stopwords.words('english'))
processed_text = processed_text.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
print(processed_text[1])
text_before_stemming  = processed_text
porter_stemmer = nltk.PorterStemmer()
processed_text = processed_text.apply(lambda x: ' '.join(porter_stemmer.stem(term) for term in x.split()))
print(processed_text[1])
print(processed_text.shape)
processed_text.head()
import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

#creating a bag of words
all_words =[]

for news in processed_text:
    words = word_tokenize(news)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)
#word_tokenize(processed_text[1])
all_words.plot(20,cumulative=False)
#print the total number of words and the 20 most common words
print('Number of words: {}'.format(len(all_words)))
print('Most common words: {}'.format(all_words.most_common(20)))
#use the 2000 most common words as feature
word_features = list(all_words.keys())[:2000]
#word_features
def find_features(news):
    words = word_tokenize(news)
    features = {}
    for word in word_features:
        features[word]=(word in words)
        
    return features
processed_text[1]
features = find_features(processed_text[1])
for key, value in features.items():
    if value == True:
        print (key)
features
y = df2['label']
#find features for all news
news = list(zip(processed_text, y))

#define a seed for reproducibility
seed = 12
np.random.seed = seed
#np.random.shuffle(news)

#call find_features function for each news article
featuresets = [(find_features(text), label) for (text, label) in news]
from sklearn import model_selection
training, testing = model_selection.train_test_split(featuresets, \
                                                    test_size=0.25,random_state = seed)
print(np.shape(featuresets))
#featuresets[3]
np.shape(news)
df_news = pd.DataFrame(news, columns=['text', 'label'])
df_news.head()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
#list sklearn models to train
model_names = ['KNeighbors','Decision Tree', 'Random Forest',\
             'Logistic Regression','SGD Classifier','Naive Bayes','SVM Linear']
sklearn_classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list(zip(model_names, sklearn_classifiers))
print(models)
#wrap models in nltk
from nltk.classify.scikitlearn import SklearnClassifier
import time

for model_name, model in models:
    start = time.time()
    
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing) * 100
    print ('{}: Accuracy: {}'.format(model_name, accuracy))
    
    stop = time.time()
    print("Model run time: {}s".format(stop - start))
# ensemble method - Voting classifier
from sklearn.ensemble import VotingClassifier

#list models to train
model_names = ['Decision Tree', 'Random Forest','Logistic Regression',\
               'SGD Classifier','Naive Bayes','SVM Linear']
sklearn_classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list(zip(model_names, sklearn_classifiers))

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models,\
                                    voting = 'hard',n_jobs = 4))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_ensemble,testing)*100
print('Ensemble Method Accuracy: {}'.format(accuracy))
# predict class label for testing dataset
text_features, labels = zip(*testing)

prediction = nltk_ensemble.classify_many(text_features)
# print a classification report and a confusion matrix
print(classification_report(labels, prediction))

pd.DataFrame(confusion_matrix(labels, prediction),
            index = [['actual','actual'], ['Positive','Negative']],
            columns = [['predicted','predicted'], ['Positive','Negative']])
y = df2['label']
news_before_stemming = list(zip(text_before_stemming, y))
df_news = pd.DataFrame(news_before_stemming, columns = ['text','label'])
fake_news = df_news[df_news['label']==1]
fake_news.head()
import nltk
nltk.download('punkt')
fake_news_words = nltk.word_tokenize(" ".join(fake_news['text'].values.tolist()))
fake_counter = Counter(fake_news_words)
print(fake_counter.most_common(50))
fake_wordcloud = WordCloud(width=800, height=800, random_state = 42).generate(" ".join(fake_news_words))

fig = plt.figure(figsize=(8,8), facecolor = 'k')
plt.imshow(fake_wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
fake_bigrams = nltk.bigrams(fake_news_words)
fake_counter = Counter(fake_bigrams)
print(fake_counter.most_common(10))
#true news
true_news = df_news[df_news['label']==0]
true_news.head()
true_news_words = nltk.word_tokenize(" ".join(true_news['text'].values.tolist()))
true_news_counter = Counter(true_news_words)
print(true_news_counter.most_common(50))
true_news_wordcloud = WordCloud(width=800, height=800, random_state = 42).generate(" ".join(true_news_words))

fig = plt.figure(figsize=(8,8), facecolor = 'k' )
plt.imshow(true_news_wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
true_news_bigrams = nltk.bigrams(true_news_words)
true_news_counter = Counter(true_news_bigrams)
print(true_news_counter.most_common(10))



