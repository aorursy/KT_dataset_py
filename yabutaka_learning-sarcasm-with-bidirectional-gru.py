# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_json("../input/Sarcasm_Headlines_Dataset.json", lines = True)
data.head(10)
data["is_sarcastic"].value_counts()
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
data = data[["headline", "is_sarcastic"]]
data.head(10)
#https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings/data

## split to train and val
train_df, test_df = train_test_split(data, test_size=0.1, random_state=2019)
train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=2019)
print("Train size:{}".format(train_df.shape))
print("Validation size:{}".format(val_df.shape))
print("Test size:{}".format(test_df.shape))
#https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
plot_wordcloud(train_df["headline"], title="Word Cloud of Questions")
from collections import defaultdict

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

train1_df = train_df[train_df["is_sarcastic"]==1]
train0_df = train_df[train_df["is_sarcastic"]==0]

## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace

## Get the bar chart from sincere questions ##
freq_dict = defaultdict(int)
for sent in train0_df["headline"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

## Get the bar chart from insincere questions ##
freq_dict = defaultdict(int)
for sent in train1_df["headline"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of not sarcastic headlines", 
                                          "Frequent words of sarcastic headlines"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.iplot(fig, filename='word-plots')

#plt.figure(figsize=(10,16))
#sns.barplot(x="ngram_count", y="ngram", data=fd_sorted.loc[:50,:], color="b")
#plt.title("Frequent words for Insincere Questions", fontsize=16)
#plt.show()
## Number of words in the text ##
train_df["num_words"] = train_df["headline"].apply(lambda x: len(str(x).split()))
test_df["num_words"] = test_df["headline"].apply(lambda x: len(str(x).split()))

# ## Number of unique words in the text ##
# train_df["num_unique_words"] = train_df["headline"].apply(lambda x: len(set(str(x).split())))
# test_df["num_unique_words"] = test_df["headline"].apply(lambda x: len(set(str(x).split())))

# ## Number of characters in the text ##
# train_df["num_chars"] = train_df["headline"].apply(lambda x: len(str(x)))
# test_df["num_chars"] = test_df["headline"].apply(lambda x: len(str(x)))

# ## Number of stopwords in the text ##
# train_df["num_stopwords"] = train_df["headline"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
# test_df["num_stopwords"] = test_df["headline"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

# ## Number of punctuations in the text ##
# train_df["num_punctuations"] =train_df['headline'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
# test_df["num_punctuations"] =test_df['headline'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

# ## Number of title case words in the text ##
# train_df["num_words_upper"] = train_df["headline"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
# test_df["num_words_upper"] = test_df["headline"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

# ## Number of title case words in the text ##
# train_df["num_words_title"] = train_df["headline"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
# test_df["num_words_title"] = test_df["headline"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

# ## Average length of the words in the text ##
# train_df["mean_word_len"] = train_df["headline"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
# test_df["mean_word_len"] = test_df["headline"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
#https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings/data

## some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

## fill up the missing values
train_X = train_df["headline"].fillna("_na_").values
val_X = val_df["headline"].fillna("_na_").values
test_X = test_df["headline"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train_df['is_sarcastic'].values
val_y = val_df['is_sarcastic'].values
#https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings/data

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
# https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings/data

## Train the model 
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
# https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings/data

pred_noemb_val_y = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_noemb_val_y>thresh).astype(int))))
# https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings/data

pred_noemb_test_y = model.predict([test_X], batch_size=1024, verbose=1)