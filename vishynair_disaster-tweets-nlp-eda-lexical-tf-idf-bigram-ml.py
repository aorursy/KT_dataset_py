#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#Importing the train and test datasets into respective dataframes with read_csv function
df = pd.read_csv('../input/nlp-getting-started/train.csv')
df_test = pd.read_csv("../input/nlp-getting-started/test.csv")
#To display maximum column width
pd.set_option('display.max_colwidth',None)
#Observing the first few records of the train dataset
df.head(10)
#Checking the dimension of the train and test datasets
print(df.shape)
print(df_test.shape)
#Basic information on the dataframes
df.info()
#NULL count in train and test
print(df.isnull().sum())
print(df_test.isnull().sum())
#Initial statistics for the columns
#We will see describe function even for the object column, so that we can get some pointers on duplicate values.
df.describe(include='all')
df.head()
#count for each category target
#Target 1 stands for real disaster and 0 stands for any other tweets
classes = df['target'].value_counts()
classes
#Creating a simple dataframe with percentage of each class
class_0 = classes[0]/df['target'].count()*100
class_1 = classes[1]/df['target'].count()*100
dist_df = pd.DataFrame({'Percentage':[class_0,class_1]},index=['Normal_Tweets','Disaster_Tweets'])
dist_df.style.background_gradient(cmap='coolwarm')
#Barplot for the classes
plt.title("Percentage of Tweet Classes",fontweight='bold')
sns.barplot(x=dist_df.index,y=dist_df['Percentage'],palette='Blues')
plt.show()
df['keyword'].value_counts()[:10]
df_not_disaster = df.loc[df['target']==0]
df_disaster = df.loc[df['target']==1]
#Top 10 keywords in the disaster tweets
df_disaster['keyword'].value_counts()[:10]
#Top 10 keywords in the other tweets
df_not_disaster['keyword'].value_counts()[:10]
#Barplots for the above
#Interpretation is always easier with the plots
plt.figure(figsize=(16,5))
plt.subplot(121)
plt.xlabel('Mentions')
plt.title('Top 10 keywords - DISASTER',fontweight='bold')
sns.barplot(y=df_disaster['keyword'].value_counts()[:10].index,x=df_disaster['keyword'].value_counts()[:10])
plt.subplot(122)
plt.xlabel('Mentions')
plt.title('Top 10 keywords - OTHER',fontweight='bold')
sns.barplot(y=df_not_disaster['keyword'].value_counts()[:10].index,x=df_not_disaster['keyword'].value_counts()[:10])
plt.show()
#Top 10 locations based on the count
df['location'].value_counts()[:10]
#Maximum Tweet Length
df['text'].str.len().max()
#Tweet with maximum length present in the train set
df.loc[df['text'].str.len()==df['text'].str.len().max()]['text']
#Minimum Tweet length
df['text'].str.len().min()
df.loc[df['text'].str.len()==df['text'].str.len().min()]
#Average tweet length
df['text'].str.len().mean()
#Average Word length of the tweet of our train corpus
df['text'].str.split().apply(lambda x: len(x)).mean()
#Average Word length of the tweet for seperate Disaster and other set
wl_not_disaster = df_not_disaster['text'].str.split().apply(lambda x: len(x))
wl_disaster = df_disaster['text'].str.split().apply(lambda x: len(x))
print(wl_not_disaster.mean())
print(wl_disaster.mean())
#Tweets with maximum word count in our train set
df.loc[df['text'].str.split().apply(lambda x: len(x))==df['text'].str.split().apply(lambda x: len(x)).max()]
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.title("Mean Word count-Other Tweets",fontweight='bold')
sns.distplot(wl_not_disaster.map(lambda x: np.mean(x)),color='grey')
plt.xlabel('count')
plt.subplot(122)
plt.title("Mean Word count-Disaster Tweets",fontweight='bold')
sns.distplot(wl_disaster.map(lambda x: np.mean(x)))
plt.xlabel('count')
plt.show()
#Trying to find if there are tweets present with a web link.
df.loc[df['text'].str.contains('http')]
#Trying to find if there are hashtags alongside
df.loc[df['text'].str.contains('#')]
def corpus_build(column):
    """Function to create a corpus list for all the words present in the tweets.Pass in the 
    dataframe column"""
    text_corpus = []
    for i in column.str.split():
        for word in i:
            text_corpus.append(word)
    return text_corpus
#Text_corpus for the train dataset tweets
text_corpus = corpus_build(df['text'])
#Text corpus for the test dataset tweets
text_corpus_test = corpus_build(df_test['text'])
#Total Number of words present in the tweets
len(text_corpus)
#Importing the stopwords
from nltk.corpus import stopwords
#We can see the stopwords listed with this.
print(stopwords.words('english'))
#Count of stopwords present in our tweets
corpus_stopwords = {}
for word in text_corpus:
    if word in stopwords.words('english'):
        if word in corpus_stopwords:
            corpus_stopwords[word] += 1
        else:
            corpus_stopwords.update({word: 1})
corpus_stopwords
#We will try to sort this words in terms of frequency - higher to lower and find out top10 frequent stopwords
corpus_stopwords_sorted = sorted(corpus_stopwords.items(),key=lambda x:x[1],reverse=True)
corpus_stopwords_10 = corpus_stopwords_sorted[:10]
corpus_stopwords_10
top_corpus_stopwords = pd.DataFrame(corpus_stopwords_10,columns=["Word","Frequency"])
top_corpus_stopwords.style.background_gradient(cmap='Blues')
#Treemap for the corpus top stopwords
fig = px.treemap(top_corpus_stopwords,path=['Word'],values='Frequency',title="Top 10 stopwords in the corpus")
fig.show()
keys = []
values = []
for i in corpus_stopwords_10:
    keys.append(i[0])
    values.append(i[1])
#Plotting the top appearing stopwords and their corresponding frequency
plt.title("Top appearing STOPWORDS",fontweight='bold')
plt.bar(keys,values,color='grey')
plt.show()
# The top appearing stopwords in our corpus are -
keys
#Example tweet with link present. We will try to check function on this
link ="Link to Regex basics - https://www.w3schools.com/python/python_regex.asp"
#Importing the regular expression function
import re
#Function to remove the links in the text
def remove_url(input):
    """Function to remove the URLs present in the text. Feed in the text data as input to function"""
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',input)
remove_url(link)
#Some of the records with urls before our process
df.iloc[31:34]
df['text'] = df['text'].apply(remove_url)
df_test['text'] = df_test['text'].apply(remove_url)
#After url removal
df.iloc[31:34]
#Importing the String module
import string
#Python provides a constant called string.punctuation that provides a great list of punctuation characters. 
print(string.punctuation)
def remove_punctuation(input1):
    """To remove all the punctuations present in the text. Input the text to the function"""
    table = str.maketrans('','',string.punctuation)
    return input1.translate(table)
#Some of the records with hash before our process
df.iloc[3:6]
df['text'] = df['text'].apply(remove_punctuation)
df_test['text'] = df_test['text'].apply(remove_punctuation)
#After Punctuation removal
df.iloc[3:6]
#Converting text column to all lowercase
df['text'] = df['text'].str.lower()
df_test['text'] = df_test['text'].str.lower()
df.loc[df['text'].str.contains("\n")][:5]
def remove_linebreaks(input1):
    """Function to remove the line breaks  present in the text. Feed in the text data as input to function"""
    text = re.compile(r'\n')
    return text.sub(r' ',input1)
df['text'] = df['text'].apply(remove_linebreaks)
df_test['text'] = df_test['text'].apply(remove_linebreaks)
#Importing the word_tokenize
from nltk.tokenize import word_tokenize
#We can tokenize all the tweets using word_tokenize
df['text'] = df['text'].apply(word_tokenize)
df_test['text'] = df_test['text'].apply(word_tokenize)
df.head()
def remove_stopwords(input1):
    """Function to remove the stopwords present in the text. Feed in the text data as input to function"""
    words = []
    for word in input1:
        if word not in stopwords.words('english'):
            words.append(word)
    return words
df['text']=df['text'].apply(remove_stopwords)
df_test['text'] = df_test['text'].apply(remove_stopwords)
df.head(10)
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
def lemma_wordnet(input1):
    """Lemmatization function"""
    return [lem.lemmatize(w) for w in input1]
df['text'].apply(lemma_wordnet)[:10]
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def stemming_porter(input1):
    """Stemming using Porter Stemmer"""
    return [stemmer.stem(w) for w in input1]
df['text'].apply(stemming_porter)[:10]
from nltk.stem.snowball import SnowballStemmer
stemmer_snowball = SnowballStemmer("english")
def stemming_snowball(input1):
    """Stemming using Snowball Stemmer"""
    return [stemmer_snowball.stem(w) for w in input1]
df['text'].apply(stemming_snowball)[:10]
df['text'] = df['text'].apply(lemma_wordnet)
df_test['text'] = df_test['text'].apply(lemma_wordnet)
df.head()
def combine_text(input1):
    """Function to combine the list words"""
    combined = ' '.join(input1)
    return combined
df['text'] = df['text'].apply(combine_text)
df_test['text'] = df_test['text'].apply(combine_text)
df.head()
#Importing the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#Object for the CountVectorizer function
vectorizer = CountVectorizer()
bow_model_train = vectorizer.fit_transform(df['text'])
bow_model_test = vectorizer.transform(df_test['text'])
#Complete sparse array
bow_model_train.toarray()
#Importing TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_tfidf = TfidfVectorizer()
tfidf_model_train = vectorizer_tfidf.fit_transform(df['text'])
tfidf_model_test = vectorizer_tfidf.transform(df_test['text'])
#Complete sparse array
tfidf_model_train.toarray()
#CountVectorizer with ngram_range=(2,2) will give us bigrams. We will fit_transform our text column with this.
vectorizer_bigram = CountVectorizer(ngram_range=(2,2),analyzer='word')
sparse_matrix = vectorizer_bigram.fit_transform(df['text'])
#We are creating here a dataframe for the bigrams which shows the frequency of this bigrams
frequencies = sum(sparse_matrix).toarray()[0]
bigram_df = pd.DataFrame(frequencies,index=vectorizer_bigram.get_feature_names(),columns=['frequency'])
#Sorting the bigram dataframe based on the frequency
bigram_df.sort_values(['frequency'],axis=0,ascending=False,inplace=True)
#Top bigrams from our train tweets
bigram_df[:10].style.background_gradient(cmap='Purples')
bigram_df.reset_index(inplace=True)
bigram_df_top20 = bigram_df[:20]
fig = px.treemap(bigram_df_top20,path=['index'],values='frequency',title='Tree of most occuring Bigrams')
fig.show()
#Importing the xgboost
import xgboost as xgb
#Setting the hyperparameters for the xgb model
xgb_param = xgb.XGBClassifier(max_depth=5, n_estimators=300, colsample_bytree=0.8, 
                                subsample=0.8, nthread=10, learning_rate=0.1)
#Importing the model_selection
from sklearn import model_selection
#Cross Validation scores with XGBoost model and bag of words representaion
scores = model_selection.cross_val_score(xgb_param, bow_model_train, df["target"], cv=5, scoring="f1")
scores
#Cross Validation scores with XGBoost model and TF-IDF representaion
scores = model_selection.cross_val_score(xgb_param, tfidf_model_train, df["target"], cv=5, scoring="f1")
scores
#Importing
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
scores = model_selection.cross_val_score(mnb, bow_model_train, df["target"], cv=5, scoring="f1")
scores
scores = model_selection.cross_val_score(mnb, tfidf_model_train, df["target"], cv=5, scoring="f1")
scores
mnb.fit(tfidf_model_train,df["target"])
df_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
df_submission.shape
df_submission['target'] = mnb.predict(tfidf_model_test)
df_submission.loc[df_submission['target']==1].shape[0]
df_submission.to_csv("submission.csv",index=False)