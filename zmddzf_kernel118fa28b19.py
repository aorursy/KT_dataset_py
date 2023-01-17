from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.utils import to_categorical

from keras.layers.core import Dense, Dropout, Activation

from keras.layers.embeddings import Embedding

from keras.layers.recurrent import SimpleRNN, LSTM

import pandas as pd

import jieba

import numpy as np
"""

读取数据 并将数据转化为可以训练的格式

"""

df = pd.read_excel('../input/train_data.xlsx')  # 读取评价信息



comments = []

for comment in df['分词前评价信息']:

    comments.append(' '.join(jieba.cut(str(comment))))  #分词，整理为list格式



# 将每一行连接成为一个长字符串 便于tokenizer建立词表 并进行文字序列化处理

tokenizer = Tokenizer(filters='!"#$%&()*+-/:;<=>?@[\]^_`{|}~【】？！', char_level=False)

tokenizer.fit_on_texts([' '.join(comments)])

sequences = tokenizer.texts_to_sequences(comments)



# 单词到索引的映射词典 索引到单词的映射词典

word_index = tokenizer.word_index

index_word = tokenizer.index_word



# 定义函数 将索引序列转化为词序列

def index2word(sequence):

    sentence = []

    for index in sequence:

        sentence.append(index_word[index])

    return sentence



print(''.join(index2word(sequences[0])))  #查看是否可以转化成功



vocab_size = len(index_word) + 1

seq_length = 2  # 通过两个词，来预测后一个词



# 将每一条评论转化成seq_length:1格式

X = []

Y = []

for sent in sequences:

    for i in range(0, len(sent) - seq_length, 1):

        X.append([[word] for word in sent[i: i + seq_length]])

        Y.append(sent[i+seq_length])

X = np.array(X)

X = np.reshape(X, (len(X), 2))

Y = np.array(Y)
# 建立模型

model = Sequential()

model.add(Embedding(vocab_size, 128, input_length=seq_length))

model.add(LSTM(256))

model.add(Dense(vocab_size, activation='softmax'))

print(model.summary())



# 编译网络

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练网络

model.fit(X, Y, epochs=100, batch_size=5000)
def generate_seq(model, tokenizer, max_length, seed_text, n_words):

	in_text = seed_text

	# generate a fixed number of words

	for _ in range(n_words):

		# encode the text as integer

		encoded = tokenizer.texts_to_sequences([in_text])[0]

		# pre-pad sequences to a fixed length

		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')

		# predict probabilities for each word

		yhat = model.predict_classes(encoded, verbose=0)

		# map predicted word index to word

		out_word = ''

		for word, index in tokenizer.word_index.items():

			if index == yhat:

				out_word = word

				break

		# append to input

		in_text += ' ' + out_word

	return in_text





generate_seq(model, tokenizer, 2, '重庆大学不错', 30)