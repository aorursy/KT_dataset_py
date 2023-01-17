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

import re
import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
tf.__version__

fake = pd.read_csv("/kaggle/input/fake-news-detection/Fake.csv", parse_dates=['date'])
true = pd.read_csv("/kaggle/input/fake-news-detection/True.csv", parse_dates=['date'])
print(fake.info())
print(fake.head())
print(fake['subject'].value_counts())
fake[fake['date']=="https://100percentfedup.com/served-roy-moore-vietnamletter-veteran-sets-record-straight-honorable-decent-respectable-patriotic-commander-soldier/"]
fake.loc[9358]['date'] = 'December 31, 2017'


fake[fake['date']=="https://100percentfedup.com/video-hillary-asked-about-trump-i-just-want-to-eat-some-pie/"]
fake.loc[15507]['date'] = 'December 29, 2017'

fake[fake['date']=="https://100percentfedup.com/12-yr-old-black-conservative-whose-video-to-obama-went-viral-do-you-really-love-america-receives-death-threats-from-left/"]
fake.loc[15508]['date'] = 'December 30, 2017'

 
fake[fake['date']=="https://fedup.wpengine.com/wp-content/uploads/2015/04/hillarystreetart.jpg"]
fake.loc[15839]['date'] = 'December 30, 2017'
fake.loc[17432]['date'] = 'December 26, 2017'
fake.loc[21869]['date'] = 'December 25, 2017'

 
fake[fake['date']=="https://fedup.wpengine.com/wp-content/uploads/2015/04/entitled.jpg"]
fake.loc[15840]['date'] = 'December 29, 2017'
fake.loc[17433]['date'] = 'December 28, 2017'
fake.loc[21870]['date'] = 'December 27, 2017'

fake[fake['date']=="MSNBC HOST Rudely Assumes Steel Worker Would Never Let His Son Follow in His Footsteps…He Couldn’t Be More Wrong [Video]"]
fake.loc[18933]['date'] = 'December 24, 2017'
fake['date'] = pd.to_datetime(fake['date'], dayfirst = True)
print("Fake News dates: ",fake['date'].min(), fake['date'].max())
print("True News dates: ",true['date'].min(), true['date'].max())
print(true.info())
print(true.head())
print(true['subject'].value_counts())
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Data cleaning
def remove_tag(string):
    text=re.sub('<.*?>','',string)
    return text
def remove_mention(text):
    line=re.sub(r'@\w+','',text)
    return line
def remove_hash(text):
    line=re.sub(r'#\w+','',text)
    return line
def remove_newline(string):
    text=re.sub('\n','',string)
    return text
def remove_url(string): 
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',string)
    return text
def remove_number(text):
    line=re.sub(r'[0-9]+','',text)
    return line
def remove_punct(text):
    line = re.sub(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*','',text)
    #string="".join(line)
    return line
def text_strip(string):
    line=re.sub('\s{2,}', ' ', string.strip())
    return line   
fake['refine_text']=fake['text'].str.lower()
fake['refine_text']=fake['refine_text'].apply(lambda x:remove_tag(str(x)))
fake['refine_text']=fake['refine_text'].apply(lambda x:remove_mention(str(x)))
fake['refine_text']=fake['refine_text'].apply(lambda x:remove_hash(str(x)))
fake['refine_text']=fake['refine_text'].apply(lambda x:remove_newline(x))
fake['refine_text']=fake['refine_text'].apply(lambda x:remove_url(x))
fake['refine_text']=fake['refine_text'].apply(lambda x:remove_number(x))
fake['refine_text']=fake['refine_text'].apply(lambda x:remove_punct(x))
fake['refine_text']=fake['refine_text'].apply(lambda x:text_strip(x))
fake['text_length']=fake['refine_text'].str.split().map(lambda x: len(x))

true['refine_text']=true['text'].str.lower()
true['refine_text']=true['refine_text'].apply(lambda x:remove_tag(str(x)))
true['refine_text']=true['refine_text'].apply(lambda x:remove_mention(str(x)))
true['refine_text']=true['refine_text'].apply(lambda x:remove_hash(str(x)))
true['refine_text']=true['refine_text'].apply(lambda x:remove_newline(x))
true['refine_text']=true['refine_text'].apply(lambda x:remove_url(x))
true['refine_text']=true['refine_text'].apply(lambda x:remove_number(x))
true['refine_text']=true['refine_text'].apply(lambda x:remove_punct(x))
true['refine_text']=true['refine_text'].apply(lambda x:text_strip(x))
true['text_length']=true['refine_text'].str.split().map(lambda x: len(x))

fake['refine_title']=fake['title'].str.lower()
fake['refine_title']=fake['refine_title'].apply(lambda x:remove_tag(str(x)))
fake['refine_title']=fake['refine_title'].apply(lambda x:remove_mention(str(x)))
fake['refine_title']=fake['refine_title'].apply(lambda x:remove_hash(str(x)))
fake['refine_title']=fake['refine_title'].apply(lambda x:remove_newline(x))
fake['refine_title']=fake['refine_title'].apply(lambda x:remove_url(x))
fake['refine_title']=fake['refine_title'].apply(lambda x:remove_number(x))
fake['refine_title']=fake['refine_title'].apply(lambda x:remove_punct(x))
fake['refine_title']=fake['refine_title'].apply(lambda x:text_strip(x))
fake['title_length']=fake['refine_title'].str.split().map(lambda x: len(x))

true['refine_title']=true['title'].str.lower()
true['refine_title']=true['refine_title'].apply(lambda x:remove_tag(str(x)))
true['refine_title']=true['refine_title'].apply(lambda x:remove_mention(str(x)))
true['refine_title']=true['refine_title'].apply(lambda x:remove_hash(str(x)))
true['refine_title']=true['refine_title'].apply(lambda x:remove_newline(x))
true['refine_title']=true['refine_title'].apply(lambda x:remove_url(x))
true['refine_title']=true['refine_title'].apply(lambda x:remove_number(x))
true['refine_title']=true['refine_title'].apply(lambda x:remove_punct(x))
true['refine_title']=true['refine_title'].apply(lambda x:text_strip(x))
true['title_length']=true['refine_title'].str.split().map(lambda x: len(x))
fig, (ax2) = plt.subplots(1,1,figsize=[17, 10])
wordcloud2 = WordCloud(background_color='black',colormap="terrain_r",width=800,height=400).generate(" ".join(fake['title']))

ax2.imshow(wordcloud2,interpolation='bilinear')
ax2.axis('off')
ax2.set_title('Fake News - Most Used Words in Title',fontsize=35)
fig, (ax2) = plt.subplots(1,1,figsize=[17, 10])
wordcloud2 = WordCloud(background_color='white',colormap="spring", width=800,height=400).generate(" ".join(fake['title']))

ax2.imshow(wordcloud2,interpolation='bilinear')
ax2.axis('off')
ax2.set_title('True News - Most Used Words in Title',fontsize=35)
print("Average length of True News  : {}".format(round(true['text_length'].mean(),2)))
print("Average length of Fake News  : {}".format(round(fake['text_length'].mean(),2)))
print("Average title length of True News  : {}".format(round(true['title_length'].mean(),2)))
print("Average title length of Fake News  : {}".format(round(fake['title_length'].mean(),2)))
fig = go.Figure()

fig.add_trace(go.Violin(y=true['title_length'], box_visible=False, line_color='black', meanline_visible=True, fillcolor='magenta', opacity=0.6,name="True", x0='True News'))
fig.add_trace(go.Violin(y=fake['title_length'], box_visible=False, line_color='black', meanline_visible=True, fillcolor='skyblue', opacity=0.6,name="Fake", x0='Fake News') )

fig.update_traces(box_visible=False, meanline_visible=True)
fig.update_layout(title_text="Violin - News Title Length",title_x=0.5)
fig.show()
fig = go.Figure()

fig.add_trace(go.Violin(y=true['text_length'], box_visible=False, line_color='black', meanline_visible=True, fillcolor='green', opacity=0.6,name="True", x0='True News'))
fig.add_trace(go.Violin(y=fake['text_length'], box_visible=False, line_color='black', meanline_visible=True, fillcolor='red', opacity=0.6,name="Fake", x0='Fake News') )

fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(title_text="Violin - News Length",title_x=0.5)
fig.show()
def ngram_df(corpus,nrange,n=None):
    vec = CountVectorizer(stop_words = 'english',ngram_range=nrange).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    total_list=words_freq[:n]
    df=pd.DataFrame(total_list,columns=['text','count'])
    return df
unigram_df=ngram_df(true['refine_title'],(1,1),20)
bigram_df=ngram_df(true['refine_title'],(2,2),20)
trigram_df=ngram_df(true['refine_title'],(3,3),20)

unigram_fake_df=ngram_df(fake['refine_title'],(1,1),20)
bigram_fake_df=ngram_df(fake['refine_title'],(2,2),20)
trigram_fake_df=ngram_df(fake['refine_title'],(3,3),20)
fig = make_subplots(
    rows=3, cols=1,subplot_titles=("Unigram","Bigram",'Trigram'),
    specs=[[{"type": "scatter"}],
           [{"type": "scatter"}],
           [{"type": "scatter"}]
          ])

fig.add_trace(go.Bar(
    y=unigram_df['text'][::-1],
    x=unigram_df['count'][::-1],
    marker={'color': "blue"},  
    text=unigram_df['count'],
    textposition = "outside",
    orientation="h",
    name="Months",
),row=1,col=1)

fig.add_trace(go.Bar(
    y=bigram_df['text'][::-1],
    x=bigram_df['count'][::-1],
    marker={'color': "blue"},  
    text=bigram_df['count'],
     name="Days",
    textposition = "outside",
    orientation="h",
),row=2,col=1)

fig.add_trace(go.Bar(
    y=trigram_df['text'][::-1],
    x=trigram_df['count'][::-1],
    marker={'color': "blue"},  
    text=trigram_df['count'],
     name="Days",
    orientation="h",
    textposition = "outside",
),row=3,col=1)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text='Top True News N Grams',xaxis_title=" ",yaxis_title=" ", showlegend=False,title_x=0.5,height=1200,template="plotly_white")
fig.show()
fig = make_subplots(
    rows=3, cols=1,subplot_titles=("Unigram","Bigram",'Trigram'),
    specs=[[{"type": "scatter"}],
           [{"type": "scatter"}],
           [{"type": "scatter"}]
          ])

fig.add_trace(go.Bar(
    y=unigram_fake_df['text'][::-1],
    x=unigram_fake_df['count'][::-1],
    marker={'color': "blue"},  
    text=unigram_fake_df['count'],
    textposition = "outside",
    orientation="h",
    name="Months",
),row=1,col=1)

fig.add_trace(go.Bar(
    y=bigram_fake_df['text'][::-1],
    x=bigram_fake_df['count'][::-1],
    marker={'color': "blue"},  
    text=bigram_fake_df['count'],
     name="Days",
    textposition = "outside",
    orientation="h",
),row=2,col=1)

fig.add_trace(go.Bar(
    y=trigram_fake_df['text'][::-1],
    x=trigram_fake_df['count'][::-1],
    marker={'color': "blue"},  
    text=trigram_fake_df['count'],
     name="Days",
    orientation="h",
    textposition = "outside",
),row=3,col=1)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text='Top Fake titles N Grams',xaxis_title=" ",yaxis_title=" ", showlegend=False,title_x=0.5,height=1200,template="seaborn")
fig.show()
true['label'] = 1
fake['label'] = 0
news = pd.concat([true,fake],ignore_index=True)
y = news['label']
news = news.drop(['label'],axis = 1)
news

def remove_stopwords(text):
    ps = PorterStemmer()    
    #review = [ps.stem(word) for word in text.split() if not word in stopwords.words('english')]    
    review = [word for word in text.split() if not word in stopwords.words('english')]    
    review = " ".join(review)
    return review


corpus = news['refine_text'].apply(lambda x:remove_stopwords(x))
#corpus = news['refine_text'].values
corpus[:3]
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import itertools

cv = CountVectorizer(max_features = 500, ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()
feature_names = cv.get_feature_names()
print("Feature Names: ",feature_names[:20])
print("X shape: ",X.shape)
print("Get Params: ",cv.get_params())

X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.33, random_state=10)
print("X_train.shape, X_test.shape, y_train.shape, y_test.shape: ",X_train.shape, X_test.shape, y_train.shape, y_test.shape)

classifier = MultinomialNB()
classifier.fit(X_train,y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test,pred)
print("Score: ",score)

cm = metrics.confusion_matrix(y_test,pred)
plot_confusion_matrix(cm, classes = ['FAKE', 'TRUE'])
from prettytable import PrettyTable
pt = PrettyTable()
pt.field_names = ["S No.", "Alpha Value", "Score"]

previous_score = 0
i=1

# Hyperparameters with MultinomialNB
for alpha in np.arange(0,1,0.1):
    sclf = MultinomialNB(alpha = alpha)
    sclf.fit(X_train,y_train)
    pred = sclf.predict(X_test)
    score = metrics.accuracy_score(pred,y_test)
    if score > previous_score:
        clf = sclf
    #print("Alpha: {}, Score: {} ".format(alpha, score))
    pt.add_row([i,round(alpha,1),round(score,3)])
    i = i+1
    
print(pt)
# Most True
sorted(zip(clf.coef_[0], feature_names),reverse=True)[:20]
# Most Fake
sorted(zip(clf.coef_[0], feature_names),reverse=False)[:20]
from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier()

linear_clf.fit(X_train,y_train)
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test,pred)
print("Score: ",score)
cm = metrics.confusion_matrix(y_test,pred)
plot_confusion_matrix(cm, classes = ['FAKE data', 'TRUE data'])
from sklearn.feature_extraction.text import TfidfVectorizer

# create the transform
tfidf = TfidfVectorizer(max_features=500,ngram_range=(1,3))

# encode document
X = tfidf.fit_transform(corpus)

## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
print(tfidf.get_feature_names()[:20])
tfidf.get_params()

clf_tf = MultinomialNB()
clf_tf.fit(X_train, y_train)
pred_tf = clf_tf.predict(X_test)
score_tf = metrics.accuracy_score(y_test, pred_tf)
print("accuracy:   %0.3f" % score_tf)
cm = metrics.confusion_matrix(y_test, pred_tf)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
# Most True
feature_names_tf = tfidf.get_feature_names()
sorted(zip(clf_tf.coef_[0], feature_names_tf),reverse=True)[:20]
# Most Fake
feature_names_tf = tfidf.get_feature_names()
sorted(zip(clf_tf.coef_[0], feature_names_tf),reverse=False)[:20]
pta = PrettyTable()
pta.field_names = ["S No.", "Alpha Value", "Score"]

# Hyperparameters with MultinomialNB (fitted with TF-IDF)

previous_score = 0
for alpha in np.arange(0,1,0.1):
    sclf = MultinomialNB(alpha = alpha)
    sclf.fit(X_train,y_train)
    pred = sclf.predict(X_test)
    score = metrics.accuracy_score(pred,y_test)
    if score > previous_score:
        clf = sclf
    pta.add_row([i,round(alpha,3),round(score,3)])
    
print(pta)
from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier()

linear_clf.fit(X_train, y_train)
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])
from sklearn.feature_extraction.text import HashingVectorizer


# create the transform
hashVec = HashingVectorizer(n_features=1000, alternate_sign=False)

# encode document
X = hashVec.fit_transform(corpus.values)

## Divide the dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

clf_hash = MultinomialNB()
clf_hash.fit(X_train, y_train)
pred_hash = clf_hash.predict(X_test)
score_hash = metrics.accuracy_score(y_test, pred_hash)
print("accuracy:   %0.3f" % score_hash)
cm = metrics.confusion_matrix(y_test, pred_hash)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
from prettytable import PrettyTable
x = PrettyTable()

x.field_names = ["S No.", "Vectorizer", "Accuracy"]

x.add_row(["1","CountVectorizer", 0.956])
x.add_row(["2","PassiveAggressiveClassifier - CountVectorizer", 0.994])
x.add_row(["3","TfidfVectorizer", 0.946])
x.add_row(["4","PassiveAggressiveClassifier - TfidfVectorizer", 0.982])
x.add_row(["5","HashingVectorizer", 0.94])

print(x)
### Vocabulary size
voc_size=5000

onehot_repr=[one_hot(words,voc_size)for words in corpus] 
print(onehot_repr[0])
sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs[0])
len(embedded_docs),y.shape
## Creating model
embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
X_final=np.array(embedded_docs)
y_final=np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

### Finally Training
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)
y_pred = model.predict_classes(X_test)
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))
print("Accuracy Score: ",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
## Creating model - Bidirectional, no dropout
embedding_vector_features=40
model2 = Sequential()
model2.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model2.add(Bidirectional(LSTM(100)))
model2.add(Dense(1,activation='sigmoid'))
model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model2.summary())
print(len(embedded_docs),y.shape)
X_final=np.array(embedded_docs)
y_final=np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

# Training
history2 = model2.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)
y_pred2=model2.predict_classes(X_test)

print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred2))
print("Accuracy Score: ",accuracy_score(y_test,y_pred2))
print(classification_report(y_test,y_pred2))
## Creating model
embedding_vector_features=40
model1=Sequential()
model1.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model1.add(Bidirectional(LSTM(100)))
model1.add(Dropout(0.3))
model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model1.summary())
print(len(embedded_docs),y.shape)
X_final=np.array(embedded_docs)
y_final=np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

# Training
model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)
y_pred1=model1.predict_classes(X_test)

print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred1))
print("Accuracy Score: ",accuracy_score(y_test,y_pred1))
print(classification_report(y_test,y_pred1))
from prettytable import PrettyTable
x = PrettyTable()

x.field_names = ["S No.", "Deep Learning", "Accuracy"]

x.add_row(["1","LSTM", 0.9433])
x.add_row(["2","BiDirectional LSTM", 0.9431])
x.add_row(["3","BiDirectional LSTM + Dropout", 0.9367])

print(x)