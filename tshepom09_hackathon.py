import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
%matplotlib inline
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import CountVectorizer
import os
print(os.listdir("../input"))
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
train = pd.read_csv('../input/sona-speeches/train.csv')
test = pd.read_csv('../input/sona-speeches/test.csv')
submission =  pd.read_csv('../input/sona-speeches/random_submission_example.csv')
train.head()
all_train = []
for i, row in train.iterrows():
    for post in row['text'].split('.'):
        all_train.append([row['president'], post])
all_train = pd.DataFrame(all_train, columns=['president', 'text'])
all_train['president'].value_counts().plot(kind = 'bar')
plt.show()
# first we make everything lower case to remove some noise from capitalisation
all_train['text'] = all_train['text'].str.lower()
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
def remove_stop_words(tokens):
    return [t for t in tokens if t not in stopwords.words('english')]
def remove_punctuation(post):
    return ''.join([l for l in post if l not in string.punctuation])
def join_string(tokens):
    return " ".join(str(token) for token in tokens) 
tokeniser = TreebankWordTokenizer()
all_train['text'] = all_train['text'].apply(remove_punctuation)
all_train['tokens'] = all_train['text'].apply(tokeniser.tokenize)
all_train['tokens'] = all_train['tokens'].apply(remove_stop_words)
all_train['join_stop'] = all_train['tokens'].apply(join_string)
df_klerk = all_train[all_train.president == 'deKlerk']
df_zuma = all_train[all_train.president == 'Mbeki']
df_mbeki = all_train[all_train.president == 'Mandela']
df_mandela = all_train[all_train.president == 'Zuma']
df_motlanthe = all_train[all_train.president == 'Motlanthe']
df_rama = all_train[all_train.president == 'Ramaphosa']
plt.figure(figsize=(20,10))
wordcloud = WordCloud(background_color='white', mode = "RGB", width = 2000, height=1000).generate(str(df_klerk['text']))
plt.title("De Klerk")
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.figure(figsize=(20,10))
wordcloud = WordCloud(background_color='white', mode = "RGB", width = 2000, height=1000).generate(str(df_mbeki['text']))
plt.title("Mbeki")
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.figure(figsize=(20,10))
wordcloud = WordCloud(background_color='white', mode = "RGB", width = 2000, height=1000).generate(str(df_zuma['text']))
plt.title("Zuma")
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.figure(figsize=(20,10))
wordcloud = WordCloud(background_color='white', mode = "RGB", width = 2000, height=1000).generate(str(df_mandela['text']))
plt.title("Mandela")
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.figure(figsize=(20,10))
wordcloud = WordCloud(background_color='white', mode = "RGB", width = 2000, height=1000).generate(str(df_mbeki['text']))
plt.title("Motlanthe")
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.figure(figsize=(20,10))
wordcloud = WordCloud(background_color='white', mode = "RGB", width = 2000, height=1000).generate(str(df_rama['text']))
plt.title("Ramaphosa")
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
all_train.head()
# vect = CountVectorizer(min_df= .01, lowercase=False, max_features=150)
# X = vect.fit_transform(all_train['join_stop'])
betterVect = CountVectorizer(stop_words='english',
                             min_df=2,
                             max_df=0.5,
                             ngram_range=(1, 2), lowercase=False, max_features=500)
X = betterVect.fit_transform(all_train['join_stop'])
df_dummies = pd.get_dummies(all_train['president'])
df_dummies.head()
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
y = df_dummies
ros = RandomOverSampler(ratio='auto', random_state=42)
#Balancing the Introvert vs Extrovert Class
#X_res, y_res = ros.fit_sample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train.shape

model = Sequential()
model.add(Dense(640, activation='relu', input_dim=X.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(100, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nb_epochs = 900
hist = model.fit(X_train, y_train, epochs= nb_epochs, batch_size=128, verbose=0)
#score = model.evaluate(x_test, y_test, batch_size=128
train_loss = hist.history['loss']
train_acc = hist.history['acc']
xc = range(nb_epochs)
plt.figure(1, figsize = (15,5))
plt.plot(xc, train_loss)
plt.plot(xc, train_acc)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend(['train_loss', 'accuracy'])
plt.figure(1, figsize = (15,5))
plt.plot(xc, train_loss)
plt.plot(xc, train_acc)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend(['train_loss', 'accuracy'])
test.head()
test['text'] = test['text'].str.lower()

test['text'] = test['text'].apply(remove_punctuation)

test['tokens'] = test['text'].apply(tokeniser.tokenize)

test['tokens'] = test['tokens'].apply(remove_stop_words)
test['join_stop'] = test['tokens'].apply(join_string)
X_infer = betterVect.fit_transform(test['join_stop'])
X_infer.shape[1]
predictions = model.predict(X_infer)
# de Klerk: 0
# Mandela: 1
# Mbeki: 2
# Mothlanthe: 3
# Zuma: 4
# Ramaphose: 5
def class_recall(x):
    if x==0:
        return 1
    elif x==1:
        return 2
    elif x==2:
        return 3
    elif x==3:
        return 5
    elif x==4:
        return 4
    elif x==5:
        return 0
y.head()
#pred = model.predict(X_test)
pred = np.argmax(predictions,axis=1)
subs = pd.DataFrame(pred)
result = pd.concat([test['sentence'], subs], axis = 1).values
result = pd.DataFrame(result, columns = ['sentence', 'president'])
result.head()
result['president'] = result['president'].apply(class_recall)
result.head()
result.to_csv('result5.csv', index=False)
from nltk.tokenize import word_tokenize, TreebankWordTokenizer#, Tokenizer
MAX_NUM_WORDS=1000 # how many unique words to use (i.e num rows in embedding vector)
MAX_SEQUENCE_LENGTH=100 # max number of words in a review to use
texts = all_train['tokens']


tokenizer = TreebankWordTokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
VALIDATION_SPLIT=0.2

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
nb_validation_samples = int(VALIDATION_SPLIT * X.shape[0])

x_train = X[:-nb_validation_samples]
y_train = y[:-nb_validation_samples]
x_val = X[-nb_validation_samples:]
y_val = y[-nb_validation_samples:]
GLOVE_DIR='../input/gloveglobalvectorsforwordrepresentation'

import os
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
EMBEDDING_DIM = 50 # how big is each word vector

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
from keras.layers import Bidirectional, GlobalMaxPool1D,Conv1D
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.models import Model


inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedded_sequences = embedding_layer(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(5, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
