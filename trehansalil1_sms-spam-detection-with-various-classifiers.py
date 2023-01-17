import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import stats

from sklearn.preprocessing import LabelEncoder, StandardScaler,MultiLabelBinarizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier
column = ['Class', 'Message']

#df = pd.read_csv('SMSSpamCollection')

df = pd.read_csv('../input/spam.csv',delimiter=',',encoding='latin-1')

df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

df.rename(columns={'v1':'Class', 'v2':'Message'}, inplace= True)

df.head()
df.info()
df.describe()
df[df['Class'] == 'spam'].describe()
df[df['Class'] == 'ham'].describe()
df.groupby('Class').describe().T
df.head()
df['Mes_len'] = df['Message'].apply(len)
df.head()
df.groupby('Class').describe().T
df['Mes_len'].plot(kind='hist',bins=50)

plt.show()
sns.barplot(df['Mes_len'],df['Class'])

plt.show()
plt.bar(df['Class'], df['Mes_len'])

#plt.xlabel()

#plt.legend()

label = ['spam', 'ham']

plt.show()
df.groupby('Class')['Mes_len'].max()
# lets check the message with the longest length for both classes

print(df.loc[df['Mes_len']==910,'Message'][1084])

print(df.loc[df['Mes_len']==224,'Message'][1020])
df.hist(column='Mes_len', by='Class', bins=50, figsize=(10,4))

plt.show()
#Removing Stopwords & Punctuations and applying Lemmatising/Stemming

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

from nltk import stem

import string



def token(example_sent):

    

    stemmer = stem.SnowballStemmer('english')

    

    stop_words = set(stopwords.words('english') + list(string.punctuation))



    word_tokens = word_tokenize(example_sent) 



    filtered_sentence = [w for w in word_tokens if w.lower() not in stop_words] 

    

    msg = [stemmer.stem(word) for word in filtered_sentence]

    return(msg)

df['Message'].apply(token)
df.head()
bag_of_words = CountVectorizer(token).fit(df['Message'])
message_bag_of_words = bag_of_words.transform(df['Message'])
message_bag_of_words.nnz

tf_idf_trans = TfidfTransformer().fit(message_bag_of_words)
messages_tfidf_trans = tf_idf_trans.transform(message_bag_of_words)
spam_detector_model = MultinomialNB().fit(messages_tfidf_trans,df['Message'])
msg_Train,msg_Test,Classification_train, Classification_test=train_test_split(df['Message'],df['Class'],test_size=0.3, random_state=123)



pipeline=Pipeline([

    ('bow',CountVectorizer(token)),

    ('Tfidf',TfidfTransformer()),

    ('NBMultinoial',MultinomialNB())

])

#Using Pipleline to perform differnt stepe in one go
pipeline.fit(msg_Train,Classification_train)
prediction=pipeline.predict(msg_Test)
print(classification_report(Classification_test,prediction))
#Using Random forest for the prediction

pipeline1=Pipeline([

    ('bow',CountVectorizer(token)),

    ('Tfidf',TfidfTransformer()),

    ('Randomforest',RandomForestClassifier(random_state=123))

])
pipeline1.fit(msg_Train,Classification_train)
prediction1=pipeline1.predict(msg_Test)
print(classification_report(Classification_test,prediction1))
#Using Light GBM for the prediction

pipeline2=Pipeline([

    ('bow',CountVectorizer(token)),

    ('Tfidf',TfidfTransformer()),

    ('LightGBM',LGBMClassifier(random_state=123))

])
pipeline2.fit(msg_Train,Classification_train)
prediction2=pipeline2.predict(msg_Test)
print(classification_report(Classification_test,prediction2))