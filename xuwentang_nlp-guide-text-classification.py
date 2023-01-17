import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
!unzip ../input/nlp-data-corpus/1d2261e2276cbb0257a2ed6e2f1f4320464c7c07

!ls
!mv ad1d9c58d338e20d09ff26bcc06c4235-1d2261e2276cbb0257a2ed6e2f1f4320464c7c07 data
!ls -alh

!ls -alh data/*
data = open('data/corpus').read()



labels, texts = [], []



for i, line in enumerate(data.split("\n")):

  content = line.split()

  labels.append(content[0])

  texts.append(' '.join(content[1:]))



# create a dataFrame using texts and labels

trainDF = pd.DataFrame()

trainDF['text'] = texts

trainDF['label'] = labels



trainDF.head()
from sklearn import model_selection



train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

print(train_x.shape, train_y.shape)

print(valid_x.shape, valid_y.shape)

print(train_y[:5])
# Series对象，index -> value

# 获取 Series 的 index

train_x_index = train_x.index.tolist()

valid_x_index = valid_x.index.tolist()



print(max(train_x_index), len(train_x_index))

print(max(valid_x_index), len(valid_x_index))

print('train_x index:', sorted(train_x_index)[:15], sorted(train_x_index)[-5:])

print('valid_x index:', sorted(valid_x_index)[:15], sorted(valid_x_index)[-5:])
print(type(train_x))

train_x.head()
# 取 Series 前 2 个元素

print(train_x[:2], '\n')



# 通过 index 获取元素，第1个元素的 index 为 9040

print(9040, train_x[9040], '\n')



# 取 index 为 0 的元素

print(train_x[0], '\n')
from sklearn import preprocessing



encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_y)

valid_y = encoder.fit_transform(valid_y)

print(train_y.shape)

print(valid_y.shape)

print(train_y[:10])
from sklearn.feature_extraction import text
count_vect = text.CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

# Learn a vocabulary dictionary of all tokens in the raw documents.

# 学习 原始文档中所有标记 的词汇表

count_vect.fit(trainDF['text'])
vocab = count_vect.vocabulary_

reverse_vocab = { idx: word for word, idx in vocab.items() }



# 验证转换索引词表

print(reverse_vocab[1743])

print(vocab[reverse_vocab[1743]])

print('vocab length:', len(vocab))
xtrain_count = count_vect.transform(train_x)

xvalid_count = count_vect.transform(valid_x)

xvalid_count
train_x_0_1 = train_x[:10].tolist()

doc_id = 5



print('raw text document:\n', train_x_0_1[doc_id])

print()

print(xtrain_count[doc_id])
for word_idx, word_freq in enumerate(xtrain_count.A[doc_id]):

    if word_freq > 0:

        word = reverse_vocab[word_idx]  # 依据词ID从逆向词典中取出词

        origin_word_freq = sum(map(lambda x: x.lower() == word, train_x_0_1[doc_id].split()))

        if word_freq == origin_word_freq:

            print(word, word_freq, origin_word_freq)

        else:

            print(word, word_freq, origin_word_freq, 

                  [w for w in train_x_0_1[doc_id].split() if w.lower().startswith(word)])
from sklearn.feature_extraction import text
tfidf_vect = text.TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(trainDF['text'])
xtrain_tfidf = tfidf_vect.transform(train_x)

xtrain_tfidf
tfidf_vect_ngram = text.TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', 

                                        ngram_range=(2,3), 

                                        max_features=5000)

tfidf_vect_ngram.fit(trainDF['text'])

xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)

xtrain_tfidf_ngram
tfidf_vect_ngram_chars = text.TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}',

                                              ngram_range=(2,3), 

                                              max_features=5000)

tfidf_vect_ngram_chars.fit(trainDF['text'])

xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x)

xtrain_tfidf_ngram_chars
wiki_news_vec_file = open('../input/wikinews300d1mvec/wiki-news-300d-1M.vec')



# 加载预训练的词向量

embeddings_index = {}

for i, line in enumerate(wiki_news_vec_file):

    values = line.split()

    if i == 0:

        print('total words:', values[0], ', embeddings length:', values[1])

    else:

        embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')



print(len(embeddings_index))
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



# 分词

token = Tokenizer()

token.fit_on_texts(trainDF['text'])



# 获取词表， word -> word_index

word_index = token.word_index

# 逆向词表

reverse_word_index ={ i: w for w, i in word_index.items()}

print(len(word_index))



# 文档分词，词ID序列

# maxlen=70，截取文本后70个词，不足的左补零

# 即文档都被对齐到70个词长度，每个词用300维的词嵌入表示

train_seq_x = pad_sequences(token.texts_to_sequences(train_x), maxlen=70)

valid_seq_x = pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

train_seq_x.shape
# 创建 token-embedding 之间的映射

embedding_matrix = np.zeros(shape=(len(word_index) + 1, 300))

for word, index in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[index] = embedding_vector
print(embedding_matrix.shape)

print(len(word_index))

word_index # word_index 的 0 号索引没有使用 
doc_id = 1



print(train_seq_x[doc_id])

print([reverse_word_index[i] for i in train_seq_x[doc_id] if i > 0])

print(train_x.tolist()[doc_id])
text_seq_demo = pad_sequences(

    token.texts_to_sequences(

        [

            'HP workhorse: Great Printer, but does get a bit cranky at times.',

            'I feel I made a good choice for my photos.'

        ]

    ),

    maxlen=70)

print(text_seq_demo)

print([reverse_word_index[i] for i in text_seq_demo[0] if i > 0])

print([reverse_word_index[i] for i in text_seq_demo[1] if i > 0])
import string



trainDF['char_count'] = trainDF['text'].apply(len)

trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))

trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count'] + 1)

trainDF['punctuation_count'] = trainDF['text'].apply(

    lambda x: len([_ for _ in x if _ in string.punctuation])

)

trainDF['title_word_count'] = trainDF['text'].apply(

    lambda x: len([wrd for wrd in x.split() if wrd.istitle()])

)

trainDF['upper_case_word_count'] = trainDF['text'].apply(

    lambda x: len([wrd for wrd in x.split() if wrd.isupper()])

)

trainDF.head()
len(trainDF['text'][1])
list(map(lambda x: len(x.split()), trainDF['text'][:2]))
doc_demo = trainDF['text'][4]

print(doc_demo)

print([wrd for wrd in doc_demo.split() if wrd.istitle()])

print([wrd for wrd in doc_demo.split() if wrd.isupper()])
pos_family = {

    'noun' : ['NN','NNS','NNP','NNPS'],

    'pron' : ['PRP','PRP$','WP','WP$'],

    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],

    'adj' :  ['JJ','JJR','JJS'],

    'adv' : ['RB','RBR','RBS','WRB']

}



import textblob



def check_pos_tag(x, flag):

    cnt = 0

    try:

        wiki = textblob.TextBlob(x)

        for tup in wiki.tags:

            ppo = list(tup)[1]

            if ppo in pos_family[flag]:

                cnt += 1

    except:

        pass

    return cnt
trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))

trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))

trainDF['adj_count']  = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))

trainDF['adv_count']  = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))

trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))

trainDF.head()
from sklearn.feature_extraction import text



count_vect = text.CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

# Learn a vocabulary dictionary of all tokens in the raw documents.

# 学习 原始文档中所有标记 的词汇表

count_vect.fit(trainDF['text'])



vocab = count_vect.vocabulary_

reverse_vocab = { idx: word for word, idx in vocab.items() }



xtrain_count = count_vect.transform(train_x)
from sklearn import decomposition



# 训练一个 LDA 主题模型

lda_model = decomposition.LatentDirichletAllocation(n_components=20, 

                                                    learning_method='online', 

                                                    max_iter=20)

lda_model
X_topics = lda_model.fit_transform(xtrain_count)
topic_word = lda_model.components_ 

print('topic word matrix:', topic_word.shape)  # (20, 31666)

vocab = count_vect.get_feature_names()

print('vocab length:', len(vocab))



"""

共有20个主题，取每个主题最有代表性的 10 个词

"""

n_top_words = 10

topic_summaries = []

for i, topic_dist in enumerate(topic_word):

    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]

    topic_summaries.append(' '.join(topic_words))



print(len(topic_summaries))

topic_summaries
from sklearn import metrics 



def train_model(classifier, 

                feature_vector_train, label, 

                feature_vector_valid, 

                is_neural_net=False):

    """ 通用的模型训练方法

    """

    classifier.fit(feature_vector_train, label)

    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:

        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)

    
from sklearn import model_selection

from sklearn import preprocessing



train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], 

                                                                      trainDF['label'])

print(train_x.shape, train_y.shape)

print(valid_x.shape, valid_y.shape)

print(train_y[:5])



# Series labels 转换为 数值

encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_y)

valid_y = encoder.fit_transform(valid_y)

print(train_y.shape)

print(valid_y.shape)

print(train_y[:10])
from sklearn.feature_extraction import text



# Word Count

count_vect = text.CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

count_vect.fit(trainDF['text'])

xtrain_count = count_vect.transform(train_x)

xvalid_count = count_vect.transform(valid_x)



# TF-IDF (word level)

tfidf_vect = text.TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(trainDF['text'])

xtrain_tfidf = tfidf_vect.transform(train_x)

xvalid_tfidf = tfidf_vect.transform(valid_x)



# TF-IDF (word n-gram)

tfidf_vect_ngram = text.TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', 

                                        ngram_range=(2,3), 

                                        max_features=5000)

tfidf_vect_ngram.fit(trainDF['text'])

xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)

xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)



# TF-IDF (char n-gram)

tfidf_vect_ngram_chars = text.TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}',

                                              ngram_range=(2,3), 

                                              max_features=5000)

tfidf_vect_ngram_chars.fit(trainDF['text'])

xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x)

xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(valid_x)
from sklearn.naive_bayes import MultinomialNB



print('NB, Count Vectors:', 

      train_model(MultinomialNB(), xtrain_count, train_y, xvalid_count))



print('NB, Word Level TF-IDF:', 

      train_model(MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf))



print('NB, Word N-gram Level TF-IDF:', 

      train_model(MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram))



print('NB, Char N-gram Level TF-IDF:', 

      train_model(MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, 

                  xvalid_tfidf_ngram_chars))
from sklearn.linear_model import LogisticRegression
print('LR, Count Vectors:', 

      train_model(LogisticRegression(), xtrain_count, train_y, xvalid_count))



print('LR, Word Level TF-IDF:', 

      train_model(LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf))



print('LR, Word N-gram Level TF-IDF:', 

      train_model(LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram))



print('LR, Char N-gram Level TF-IDF:', 

      train_model(LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, 

                  xvalid_tfidf_ngram_chars))
from sklearn.svm import SVC



print('SVM, Count Vectors:', 

      train_model(SVC(), xtrain_count, train_y, xvalid_count))



print('SVM, Word Level TF-IDF:', 

      train_model(SVC(gamma=auto), xtrain_tfidf, train_y, xvalid_tfidf))



print('SVM, Word N-gram Level TF-IDF:', 

      train_model(SVC(gamma=auto), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram))



print('SVM, Char N-gram Level TF-IDF:', 

      train_model(SVC(gamma=auto), xtrain_tfidf_ngram_chars, train_y, 

                  xvalid_tfidf_ngram_chars))
from sklearn.ensemble import RandomForestClassifier



print('RF, Count Vectors:', 

      train_model(RandomForestClassifier(n_estimators=10), xtrain_count, train_y, 

                  xvalid_count))



print('RF, Word Level TF-IDF:', 

      train_model(RandomForestClassifier(n_estimators=10), xtrain_tfidf, train_y, 

                  xvalid_tfidf))



print('RF, Word N-gram Level TF-IDF:', 

      train_model(RandomForestClassifier(n_estimators=10), xtrain_tfidf_ngram, train_y, 

                  xvalid_tfidf_ngram))
RandomForestClassifier(n_estimators=10)
from xgboost import XGBClassifier



print('Xgb, Count Vectors:', 

      train_model(XGBClassifier(), xtrain_count, train_y, 

                  xvalid_count))



print('Xgb, Word Level TF-IDF:', 

      train_model(XGBClassifier(), xtrain_tfidf, train_y, 

                  xvalid_tfidf))



print('Xgb, Word N-gram Level TF-IDF:', 

      train_model(XGBClassifier(), xtrain_tfidf_ngram, train_y, 

                  xvalid_tfidf_ngram))
from keras.layers import Input, Dense

from keras.models import Model

from keras.optimizers import Adam

input_size = xtrain_tfidf_ngram.shape[1]



input_layer = Input(shape=(input_size,), sparse=True)

hidden_layer = Dense(units=100, activation='relu')(input_layer)

output_layer = Dense(units=1, activation='sigmoid')(hidden_layer)



classifier = Model(inputs=input_layer, outputs=output_layer)

classifier.summary()
classifier.compile(Adam(), loss='binary_crossentropy')

accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)

print('NN, N-gram Level IF-IDF Vectors:', accuracy)
wiki_news_vec_file = open('../input/wikinews300d1mvec/wiki-news-300d-1M.vec')



# 加载预训练的词向量

embeddings_index = {}

for i, line in enumerate(wiki_news_vec_file):

    values = line.split()

    if i == 0:

        print('total words:', values[0], ', embeddings length:', values[1])

    else:

        embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')



print(len(embeddings_index))





from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



# 分词

token = Tokenizer()

token.fit_on_texts(trainDF['text'])



# 获取词表， word -> word_index

word_index = token.word_index

reverse_word_index ={ i: w for w, i in word_index.items()}



# 文档分词，词ID序列

# maxlen=70，截取文本后70个词，不足的左补零

# 即文档都被对齐到70个词长度，每个词用300维的词嵌入表示

train_seq_x = pad_sequences(token.texts_to_sequences(train_x), maxlen=70)

valid_seq_x = pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)



# 创建 token-embedding 之间的映射

embedding_matrix = np.zeros(shape=(len(word_index) + 1, 300))

for word, index in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[index] = embedding_vector
def network_train_estimate(classifier):

    # train

    classifier.fit(x=train_seq_x, y=train_y, batch_size=128, epochs=5, validation_data=(valid_seq_x, valid_y))

    # predict

    predictions = classifier.predict(valid_seq_x)

    # score

    score = metrics.accuracy_score(valid_y, [1 if i >= 0.5 else 0 for i in predictions])

    return score
from keras import layers, models, optimizers
input_layer = layers.Input(shape=(70,))

embedding_layer = layers.Embedding(input_dim=len(word_index) + 1, output_dim=300, 

                                   weights=[embedding_matrix], 

                                   trainable=False)(input_layer)

embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)



# 卷积层及池化层

conv_layer = layers.Convolution1D(filters=100, kernel_size=3, activation="relu")(embedding_layer)

pooling_layer = layers.GlobalMaxPool1D()(conv_layer)





# Add the output Layers

output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)

output_layer1 = layers.Dropout(0.25)(output_layer1)

output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)



model = models.Model(inputs=input_layer, outputs=output_layer2)

model.summary()



# Compile the model

model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')





print('CNN, Word Embeddings:', network_train_estimate(model))
def create_rnn_lstm():

    input_layer = layers.Input(shape=(70,))

    embedding_layer = layers.Embedding(input_dim=len(word_index) + 1, output_dim=300, 

                                       weights=[embedding_matrix], 

                                       trainable=False)(input_layer)

    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)



    # lstm

    lstm_layer = layers.LSTM(100)(embedding_layer)





    # Add the output Layers

    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)

    output_layer1 = layers.Dropout(0.25)(output_layer1)

    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)



    model = models.Model(inputs=input_layer, outputs=output_layer2)

    model.summary()

    # Compile the model

    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model





classifier = create_rnn_lstm()

print('RNN-LSTM, Word Embeddings:', network_train_estimate(classifier))
def create_rnn_gru():

    # Add an Input Layer

    input_layer = layers.Input((70, ))



    # Add the word embedding Layer

    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)

    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)



    # Add the GRU Layer

    lstm_layer = layers.GRU(100)(embedding_layer)



    # Add the output Layers

    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)

    output_layer1 = layers.Dropout(0.25)(output_layer1)

    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)



    # construct the model

    model = models.Model(inputs=input_layer, outputs=output_layer2)

    model.summary()

    

    # Compile the model

    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    

    return model



classifier = create_rnn_gru()

print('RNN-GRU, Word Embeddings:', network_train_estimate(classifier))
def create_bidirectional_rnn():

    # Add an Input Layer

    input_layer = layers.Input((70, ))



    # Add the word embedding Layer

    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)

    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)



    # Add the LSTM Layer

    lstm_layer = layers.Bidirectional(layers.GRU(100))(embedding_layer)



    # Add the output Layers

    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)

    output_layer1 = layers.Dropout(0.25)(output_layer1)

    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)



    # Construct the model

    model = models.Model(inputs=input_layer, outputs=output_layer2)

    model.summary()

    # Compile the model

    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    

    return model



classifier = create_bidirectional_rnn()

print('RNN-Bidirectional, Word Embeddings:', network_train_estimate(classifier))