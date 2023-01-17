## import libraries

import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## load train test data sets

data_raw = pd.read_csv("/kaggle/input/topicmodel/train.csv")

test = pd.read_csv("/kaggle/input/topicmodel/test.csv")
## shape of datasets

data_raw.shape,test.shape
data_raw.head(10)
data_raw['ABSTRACT'][10]
## check for missing values

missing_values_check = data_raw.isnull().sum()

print(missing_values_check)
# Comments with no label are considered to be clean comments.

# Creating seperate column in dataframe to identify clean comments.



# We use axis=1 to count row-wise and axis=0 to count column wise



rowSums = data_raw.iloc[:,3:].sum(axis=1)

clean_comments_count = (rowSums==0).sum(axis=0)



print("Total number of comments = ",len(data_raw))

print("Number of clean comments = ",clean_comments_count)

print("Number of comments with labels =",(len(data_raw)-clean_comments_count))
### find different categories

categories = list(data_raw.columns.values)

categories = categories[3:]

print(categories)
# Calculating number of comments in each category

counts = []

for category in categories:

    counts.append((category, data_raw[category].sum()))

df_stats = pd.DataFrame(counts, columns=['category', 'number of comments'])

df_stats
### lets plot comments in each category

sns.set(font_scale = 1)

plt.figure(figsize=(15,8))



ax= sns.barplot(categories, data_raw.iloc[:,3:].sum().values)



plt.title("Comments in each category", fontsize=20)

plt.ylabel('Number of comments', fontsize=15)

plt.xlabel('Comment Type ', fontsize=15)



#adding the text labels

rects = ax.patches

labels = data_raw.iloc[:,3:].sum().values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)



plt.show()
## lets plot Comments having multiple labels

rowSums = data_raw.iloc[:,3:].sum(axis=1)

multiLabel_counts = rowSums.value_counts()

multiLabel_counts = multiLabel_counts.iloc[1:]



sns.set(font_scale = 1)

plt.figure(figsize=(10,6))



ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)



plt.title("Comments having multiple labels ")

plt.ylabel('Number of comments', fontsize=15)

plt.xlabel('Number of labels', fontsize=15)



#adding the text labels

rects = ax.patches

labels = multiLabel_counts.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()
#!pip install wordcloud

from wordcloud import WordCloud,STOPWORDS



plt.figure(figsize=(40,25))



# Computer Science

subset = data_raw[data_raw['Computer Science']==1]

text = subset.TITLE.values

cloud_Computer_Science = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 1)

plt.axis('off')

plt.title("Computer_Science",fontsize=40)

plt.imshow(cloud_Computer_Science)





# Physics

subset = data_raw[data_raw.Physics==1]

text = subset.TITLE.values

cloud_Physics = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 2)

plt.axis('off')

plt.title("Physics",fontsize=40)

plt.imshow(cloud_Physics)



# Mathematics

subset = data_raw[data_raw.Mathematics==1]

text = subset.TITLE.values

cloud_Mathematics = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 3)

plt.axis('off')

plt.title("Mathematics",fontsize=40)

plt.imshow(cloud_Mathematics)





# Statistics

subset = data_raw[data_raw.Statistics==1]

text = subset.TITLE.values

cloud_Statistics = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 4)

plt.axis('off')

plt.title("Statistics",fontsize=40)

plt.imshow(cloud_Statistics)



# Quantitative Finance

subset = data_raw[data_raw['Quantitative Finance']==1]

text = subset.TITLE.values

cloud_Quantitative_Finance = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 5)

plt.axis('off')

plt.title("Quantitative_Finance",fontsize=40)

plt.imshow(cloud_Quantitative_Finance)





# Quantitative Biology

subset = data_raw[data_raw['Quantitative Biology']==1]

text = subset.TITLE.values

cloud_Quantitative_Biology = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 6)

plt.axis('off')

plt.title("Quantitative_Biology",fontsize=40)

plt.imshow(cloud_Quantitative_Biology)



plt.show()
#!pip install wordcloud

from wordcloud import WordCloud,STOPWORDS



plt.figure(figsize=(40,25))



# Computer Science

subset = data_raw[data_raw['Computer Science']==1]

text = subset.ABSTRACT.values

cloud_Computer_Science = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 1)

plt.axis('off')

plt.title("Computer_Science",fontsize=40)

plt.imshow(cloud_Computer_Science)





# Physics

subset = data_raw[data_raw.Physics==1]

text = subset.ABSTRACT.values

cloud_Physics = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 2)

plt.axis('off')

plt.title("Physics",fontsize=40)

plt.imshow(cloud_Physics)



# Mathematics

subset = data_raw[data_raw.Mathematics==1]

text = subset.ABSTRACT.values

cloud_Mathematics = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 3)

plt.axis('off')

plt.title("Mathematics",fontsize=40)

plt.imshow(cloud_Mathematics)





# Statistics

subset = data_raw[data_raw.Statistics==1]

text = subset.ABSTRACT.values

cloud_Statistics = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 4)

plt.axis('off')

plt.title("Statistics",fontsize=40)

plt.imshow(cloud_Statistics)



# Quantitative Finance

subset = data_raw[data_raw['Quantitative Finance']==1]

text = subset.ABSTRACT.values

cloud_Quantitative_Finance = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 5)

plt.axis('off')

plt.title("Quantitative_Finance",fontsize=40)

plt.imshow(cloud_Quantitative_Finance)





# Quantitative Biology

subset = data_raw[data_raw['Quantitative Biology']==1]

text = subset.ABSTRACT.values

cloud_Quantitative_Biology = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 6)

plt.axis('off')

plt.title("Quantitative_Biology",fontsize=40)

plt.imshow(cloud_Quantitative_Biology)



plt.show()
### import nltk libraries for text processing

import nltk

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

from nltk.stem import PorterStemmer

from nltk.stem import LancasterStemmer

import re



import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")
### lets create functions to deal with texts cleaning



def cleanHtml(sentence):

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, ' ', str(sentence))

    return cleantext





def cleanPunc(sentence): #function to clean the word of any punctuation or special characters

    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)

    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)

    cleaned = cleaned.strip()

    cleaned = cleaned.replace("\n"," ")

    return cleaned





def keepAlpha(sentence):

    alpha_sent = ""

    for word in sentence.split():

        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)

        alpha_sent += alpha_word

        alpha_sent += " "

    alpha_sent = alpha_sent.strip()

    return alpha_sent
### apply data cleaning functions.



data_raw['ABSTRACT'] = data_raw['ABSTRACT'].str.lower()

data_raw['ABSTRACT'] = data_raw['ABSTRACT'].apply(cleanHtml)

#data_raw['ABSTRACT'] = data_raw['ABSTRACT'].apply(cleanPunc)

data_raw['ABSTRACT'] = data_raw['ABSTRACT'].apply(keepAlpha)





data_raw['TITLE'] = data_raw['TITLE'].str.lower()

data_raw['TITLE'] = data_raw['TITLE'].apply(cleanHtml)

#data_raw['TITLE'] = data_raw['TITLE'].apply(cleanPunc)

data_raw['TITLE'] = data_raw['TITLE'].apply(keepAlpha)



data_raw.head()
### apply data cleaning functions



test['ABSTRACT'] = test['ABSTRACT'].str.lower()

test['ABSTRACT'] = test['ABSTRACT'].apply(cleanHtml)

#test['ABSTRACT'] = test['ABSTRACT'].apply(cleanPunc)

test['ABSTRACT'] = test['ABSTRACT'].apply(keepAlpha)





test['TITLE'] = test['TITLE'].str.lower()

test['TITLE'] = test['TITLE'].apply(cleanHtml)

#test['TITLE'] = test['TITLE'].apply(cleanPunc)

test['TITLE'] = test['TITLE'].apply(keepAlpha)



test.head()
data_raw['ABSTRACT'][6]
stop_words = set(stopwords.words('english'))



### define few my own stop words



stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across',

              'among','beside','however','yet','within','a', 'about', 'above', 'after', 'again', 'against', 'all', 'also', 

              'am', 'an', 'and','any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',

              'between', 'both', 'but', 'by', 'can', "can't", 'cannot', 'com', 'could', "couldn't", 'did',

              "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'else', 'ever',

              'few', 'for', 'from', 'further', 'get', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having',

              'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how',

              "how's", 'however', 'http', 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it',

              "it's", 'its', 'itself', 'just', 'k', "let's", 'like', 'me', 'more', 'most', "mustn't", 'my', 'myself',

              'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'otherwise', 'ought', 'our', 'ours',

              'ourselves', 'out', 'over', 'own', 'r', 'same', 'shall', "shan't", 'she', "she'd", "she'll", "she's",

              'should', "shouldn't", 'since', 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs',

              'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're",

              "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't",

              'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',

              "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't",

              'www', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])



re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

def removeStopWords(sentence):

    global re_stop_words

    return re_stop_words.sub(" ", sentence)



# additional cleaning

def clean_text(text):

    

    #Remove Unicode characters

    text = re.sub(r'[^\x00-\x7F]+', '', text)

    #remove double spaces 

    text = re.sub('\s+', ' ',text) 

    return text



data_raw['ABSTRACT'] = data_raw['ABSTRACT'].apply(removeStopWords)

data_raw['TITLE'] = data_raw['TITLE'].apply(removeStopWords)



#data_raw['TITLE'] = data_raw['TITLE'].apply(lambda x: clean_text(x))

#data_raw['ABSTRACT'] = data_raw['ABSTRACT'].apply(lambda x: clean_text(x))



data_raw.head()
## remove stop words from data

test['ABSTRACT'] = test['ABSTRACT'].apply(removeStopWords)

test['TITLE'] = test['TITLE'].apply(removeStopWords)



#test['TITLE'] = test['TITLE'].apply(lambda x: clean_text(x))

#test['ABSTRACT'] = test['ABSTRACT'].apply(lambda x: clean_text(x))



test.head()
data_raw['TITLE'][0]
### apply stemmer tried multiple stemmer like porter lancaster but snowball works better than others. did not try lemmatizer



stemmer = SnowballStemmer("english")

def stemming(sentence):

    stemSentence = ""

    for word in sentence.split():

        stem = stemmer.stem(word)

        stemSentence += stem

        stemSentence += " "

    stemSentence = stemSentence.strip()

    return stemSentence



data_raw['ABSTRACT'] = data_raw['ABSTRACT'].apply(stemming)

data_raw['TITLE'] = data_raw['TITLE'].apply(stemming)



data_raw.head()
## apply stemmer

test['ABSTRACT'] = test['ABSTRACT'].apply(stemming)

test['TITLE'] = test['TITLE'].apply(stemming)

test.head()
### just take a copy of test data

test_fin = test.copy()
# from sklearn.model_selection import train_test_split



# train, test = train_test_split(data_raw, random_state=42, test_size=0.30, shuffle=True)



# print(train.shape)

# print(test.shape)
## lets merge texts from multiple columns into one column

data_raw['ABSTRACT'] = data_raw['TITLE'] + data_raw['ABSTRACT'] 

test_fin['ABSTRACT'] = test_fin['TITLE'] + test_fin['ABSTRACT'] 
data_raw['ABSTRACT'][10]
data_raw.head()
# train_text = train['ABSTRACT']

# test_text = test['ABSTRACT']

train_text = data_raw['ABSTRACT']
test_final = test_fin['ABSTRACT']
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2',max_df=2.5)

#vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2',sublinear_tf=True,stop_words="english",token_pattern=r'\w{1,}')

vectorizer.fit(train_text)
x_train = vectorizer.transform(train_text)

#y_train = train.drop(labels = ['ID','TITLE','ABSTRACT'], axis=1)

y_train = data_raw.drop(labels = ['ID','TITLE','ABSTRACT'], axis=1)

#x_test = vectorizer.transform(test_text)

#y_test = test.drop(labels = ['ID','TITLE','ABSTRACT'], axis=1)
test_f = vectorizer.transform(test_final)
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC

from xgboost import XGBClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
from IPython.display import Markdown, display

def printmd(string):

    display(Markdown(string))

    

# Using pipeline for applying logistic regression and one vs rest classifier

#LogReg_pipeline = Pipeline([

#                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),

#            ])



# Using pipeline for applying logistic regression and one vs rest classifier

LogReg_pipeline = Pipeline([

                ('clf', OneVsRestClassifier(LinearSVC(class_weight='balanced',random_state=0,tol=1e-1,C=8.385), n_jobs=-1)),

            ])





# Using pipeline for applying logistic regression and one vs rest classifier

#LogReg_pipeline = Pipeline([

#                ('clf', OneVsRestClassifier(MultinomialNB(alpha=1e-2,fit_prior=True, class_prior=None), n_jobs=-1)),

#            ])





printmd('**Processing {} comments...**'.format('Computer Science'))

    

    # Training logistic regression model on train data

LogReg_pipeline.fit(x_train, data_raw['Computer Science'])

    

    # calculating test accuracy

pred_df = pd.DataFrame()

pred_df['Computer Science'] = LogReg_pipeline.predict(test_f)



printmd('**Processing {} comments...**'.format('Physics'))

# Training logistic regression model on train data

LogReg_pipeline.fit(x_train, data_raw['Physics'])

# calculating test accuracy

#pred_df = pd.DataFrame()

pred_df['Physics'] = LogReg_pipeline.predict(test_f)



printmd('**Processing {} comments...**'.format('Mathematics'))

# Training logistic regression model on train data

LogReg_pipeline.fit(x_train, data_raw['Mathematics'])

# calculating test accuracy

#pred_df = pd.DataFrame()

pred_df['Mathematics'] = LogReg_pipeline.predict(test_f)



printmd('**Processing {} comments...**'.format('Statistics'))

# Training logistic regression model on train data

LogReg_pipeline.fit(x_train, data_raw['Statistics'])

# calculating test accuracy

#pred_df = pd.DataFrame()

pred_df['Statistics'] = LogReg_pipeline.predict(test_f)



printmd('**Processing {} comments...**'.format('Quantitative Biology'))

# Training logistic regression model on train data

LogReg_pipeline.fit(x_train, data_raw['Quantitative Biology'])

# calculating test accuracy

#pred_df = pd.DataFrame()

pred_df['Quantitative Biology'] = LogReg_pipeline.predict(test_f)



printmd('**Processing {} comments...**'.format('Quantitative Finance'))

# Training logistic regression model on train data

LogReg_pipeline.fit(x_train, data_raw['Quantitative Finance'])

# calculating test accuracy

#pred_df = pd.DataFrame()

pred_df['Quantitative Finance'] = LogReg_pipeline.predict(test_f)

    

    #print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))

    #print("\n")

print(pred_df.shape)

pred_df.head()
### save results in submission file



pred_df['ID'] = test_fin['ID'].copy()

pred_df = pred_df.set_index('ID')

#pred_df.head()

pred_df.to_csv('submission.csv')
# # using classifier chains

# from sklearn.multioutput import ClassifierChain

# from sklearn.linear_model import LogisticRegression



# #%%time



# # initialize classifier chains multi-label classifier

# classifier = ClassifierChain(LinearSVC(class_weight='balanced',random_state=0,tol=1e-2,C=8.385))



# # Training logistic regression model on train data

# classifier.fit(x_train, y_train)



# # # predict

# predictions = classifier.predict(test_f).astype(int)

# predictions_df = pd.DataFrame(predictions)

# predictions_df['ID'] = test_fin['ID'].copy()

# predictions_df = predictions_df.set_index('ID')

# predictions_df = predictions_df.rename(columns={0:'Computer Science',1:'Physics',2:'Mathematics',3:'Statistics',4:'Quantitative Biology',5:'Quantitative Finance'})

# predictions_df.head()

# predictions_df.to_csv('submission_chain.csv')