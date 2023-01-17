import numpy as np

import pandas as pd 

import os

import re

import string



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('seaborn')

from plotly import graph_objs as go

import plotly.express as px

from collections import Counter

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS

from PIL import Image



import keras

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, Flatten, Dropout

from keras.optimizers import Adam
df1 = pd.read_csv('../input/nlp-getting-started/train.csv')

df2 = pd.read_csv('../input/nlp-getting-started/test.csv')

submit = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
print(df1.shape)

print(df2.shape)
df1.info()
df1.head()
temp = df1.groupby('target').count()['text'].reset_index()

temp['label'] = temp['target'].apply(lambda x : 'Disaster Tweet' if x==1 else 'Non Disaster Tweet')

temp
plt.figure(figsize=(7,5))

sns.countplot(x='target',data=df1)
fig = go.Figure(go.Funnelarea(

    text = temp.label,

    values = temp.text,

    title = {"position" : "top center", "text" : "Funnel Chart for target distribution"}

    ))

fig.show()
df1['target_mean'] = df1.groupby('keyword')['target'].transform('mean')



fig = plt.figure(figsize=(8, 78), dpi=100)



sns.countplot(y=df1.sort_values(by='target_mean', ascending=False)['keyword'],

              hue=df1.sort_values(by='target_mean', ascending=False)['target'])



plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=12)

plt.legend(loc=1)

plt.title('Target Distribution in Keywords')



plt.show()



df1.drop(columns=['target_mean'], inplace=True)
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

tweet_len=df1[df1['target']==1]['text'].str.split().map(lambda x: len(x))

ax1.hist(tweet_len,color='red')

ax1.set_title('Disaster Tweets')

tweet_len=df1[df1['target']==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(tweet_len,color='blue')

ax2.set_title('Non Disaster Tweets')

fig.suptitle('No.of words in a tweet')

plt.show()
def clean_text(text):

    text = str(text).lower()

    return text



df1['text_plot'] = df1['text'].apply(lambda x:clean_text(x))



df1['temp_list'] = df1['text_plot'].apply(lambda x:str(x).split())

top = Counter([item for sublist in df1['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(25))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Blues')
fig = px.bar(temp, x='count',y='Common_words',title='Common words in tweet',orientation='h',width=700,height=700,color='Common_words')

fig.show()
def remove_stopwords(x):

    return [y for y in x if y not in stopwords.words('english')]

df1['temp_list'] = df1['temp_list'].apply(lambda x : remove_stopwords(x))
top = Counter([item for sublist in df1['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(25))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Purples')
fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words')

fig.show()
text = df1['text'].values

twitter_logo = np.array(Image.open('../input/twitter-logo2/10wmt-articleLarge-v4.jpg'))

cloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          mask = twitter_logo,

                          max_words=200

                         ).generate(" ".join(text))



plt.imshow(cloud)

plt.axis('off')

plt.show()
del df1['text_plot']

del df1['temp_list']



df = pd.concat([df1,df2])
df.head()
for col in ['keyword', 'location']:

    df[col] = df[col].fillna(f'no_{col}')
df.head()
df['text']=df['text'].str.replace('https?://\S+|www\.\S+','').str.replace('<.*?>','')
def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)
df['text'] = df['text'].apply(lambda x : remove_emoji(x))
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)
df['text'] = df['text'].apply(lambda x : remove_punct(x))
def clean_text(text):

    text = re.sub('\s+', ' ', text).strip() 

    return text
df['text'] = df['text'].apply(lambda x : clean_text(x))
dfs = np.split(df, [len(df1)], axis=0)
train = dfs[0]

train.shape
test = dfs[1]

test.shape
test.drop('target',axis=1,inplace=True)
vocab_size = len(test)

text = train['text'].values

label = train['target'].values
encoded_docs = [one_hot(d,vocab_size) for d in text]

for x in range(5):

    print(encoded_docs[x])
max_len = len(train['text'].max())

pad_docs = pad_sequences(encoded_docs,maxlen=max_len,padding='post')
train.shape
model = Sequential()

model.add(Embedding(7613,100,input_length=max_len))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['acc'])
model.summary()
model.fit(pad_docs,label,epochs=20,batch_size=32)
prediction = model.predict(pad_docs)
train['prediction'] = prediction
train['prediction'] = train['prediction'].apply(lambda x : 0 if x<0.5 else 1)
train.head()
text2 = test['text'].values

encoded_docs2 = [one_hot(d,vocab_size) for d in text2]

pad_docs2 = pad_sequences(encoded_docs2,maxlen=max_len,padding='post')
prediction2 = model.predict(pad_docs2)
test['prediction'] = prediction2

test['prediction'] = test['prediction'].apply(lambda x : 0 if x<0.5 else 1)
test.head()
submit['target'] = test['prediction']
submit.head()
submit.to_csv('submission.csv',index=False)