import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import gc
import re
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Activation, Dense, Bidirectional
df_poetry=pd.read_csv("../input/poetry-analysis-with-machine-learning/all.csv", sep=",")
df_poetry.head()
df_poetry.rename(columns={"poem name":"poem_name"}, inplace=True)
df_poetry.age.unique()
df_poetry.type.unique()
df_poetry.author.unique()
def remove_special_chars(text, remove_digits=True):
    text=re.sub('[^a-zA-Z.\d\s]', '',text)
    return text
df_poetry.content=df_poetry.content.apply(remove_special_chars)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_poetry.age=le.fit_transform(df_poetry.age)
df_poetry
df_poetry.drop(columns=["author", "poem_name","type"])
from keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer(num_words=1009)
tokenizer.fit_on_texts(df_poetry.content)
sequences=tokenizer.texts_to_sequences(df_poetry.content)
tokenized=tokenizer.texts_to_matrix(df_poetry.content)
word_index=tokenizer.word_index
print("Found %s unique tokens."%len(word_index))
tokenized
tokenized.shape
X=tokenized
Y=df_poetry.age
tokenized.shape
df_poetry.age.shape
X_train, X_test, y_train, y_test =train_test_split(X,Y,test_size=0.2)
X_train=tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=300)
X_test=tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=300)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='my_log_dir')
max_features=10
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
model = tf.keras.Sequential([Embedding(input_dim=100, output_dim=128),
                            LSTM(128,activation='relu', dropout=0.05, return_sequences=True),
                            LSTM(128, activation="relu",dropout=0.05,recurrent_dropout=0.01, return_sequences=True),
                            LSTM(64, activation="relu",dropout=0.01,recurrent_dropout=0.01, return_sequences=True),
                            LSTM(32, activation="relu",dropout=0.01,recurrent_dropout=0.01),
                            Dense(2, activation="relu"),
                            Dense(1, activation="sigmoid")])
opt=tf.keras.optimizers.RMSprop()
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["acc"])
model.fit(X_train, y_train.values, epochs=100, batch_size=40, validation_split=0.1, callbacks=[callback, tensorboard])
model.evaluate(X_test, y_test)
df_poetry=pd.read_csv("../input/poetry-analysis-with-machine-learning/all.csv", sep=",")
df_poetry.head()
import plotly.graph_objects as go
from plotly.offline import iplot
words = df_poetry['content'].str.split(expand=True).unstack().value_counts()
data = [go.Bar(
            x = words.index.values[2:20],
            y = words.values[2:20],
            marker= dict(colorscale='RdBu',
                         color = words.values[2:40]
                        ),
            text='Word counts'
    )]

layout = go.Layout(
    title='Most used words excluding stopwords'
)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic-bar')
import matplotlib.pyplot as plt
def word_cloud(content, title):
    wc = WordCloud(background_color='white', max_words=200,
                  stopwords=STOPWORDS, max_font_size=50)
    wc.generate(" ".join(content))
    plt.figure(figsize=(16, 13))
    plt.title(title, fontsize=20)
    plt.imshow(wc.recolor(colormap='Pastel2', random_state=42), alpha=0.98)
    plt.axis('off')
word_cloud(df_poetry.content, "Word Cloud")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_poetry.type=le.fit_transform(df_poetry.type)
df_poetry.age=le.fit_transform(df_poetry.age)
df_poetry.author=le.fit_transform(df_poetry.author)
corr = df_poetry.corr()
corr
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
sns.catplot(x="age", y="author",hue="type", data=df_poetry);
y=df_poetry['author']
x=df_poetry["content"]
X_train, X_test, y_train, y_test =train_test_split(x,y,test_size=0.33, random_state=50)
print(X_train)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectrain = vectorizer.fit_transform(X_train)
vectest = vectorizer.transform(X_test)
vectest.shape
y_train.shape
dtclassifier=DecisionTreeClassifier(criterion="entropy", max_depth=None)
dtclassifier.fit(vectrain,y_train)
preddt = dtclassifier.predict(vectest)
accuracy= accuracy_score(preddt,y_test)
print(accuracy)
y=df_poetry['age']
x=df_poetry["content"]
X_train, X_test, y_train, y_test =train_test_split(x,y,test_size=0.33, random_state=50)
vectorizer = TfidfVectorizer()
vectrain = vectorizer.fit_transform(X_train)
vectest = vectorizer.transform(X_test)
dtclassifier=DecisionTreeClassifier(criterion="entropy", max_depth=None)
dtclassifier.fit(vectrain,y_train)
preddt = dtclassifier.predict(vectest)
accuracy= accuracy_score(preddt,y_test)
print(accuracy)
y=df_poetry['author']
X=df_poetry.loc[:, df_poetry.columns!="author"]
X_train, X_test, y_train, y_test =train_test_split(x,y,test_size=0.33, random_state=50)
vectorizer = TfidfVectorizer()
vectrain = vectorizer.fit_transform(X_train)
vectest = vectorizer.transform(X_test)
dtclassifier=DecisionTreeClassifier(criterion="gini", max_depth=None)
dtclassifier.fit(vectrain,y_train)
preddt = dtclassifier.predict(vectest)
accuracy= accuracy_score(preddt,y_test)
print(accuracy)