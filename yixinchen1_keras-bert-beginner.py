# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install keras-bert
from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths



model_path = get_pretrained(PretrainedList.multi_cased_base)

paths = get_checkpoint_paths(model_path)

print(paths.config, paths.checkpoint, paths.vocab)
!ls /kaggle/input/bertenglish
from keras_bert import extract_embeddings



model_path = '/kaggle/input/bertenglish'

texts = ['Today is a fine day','I adore you','I hate you','you look beautiful']

template = ['I love you']



embeddings = extract_embeddings(model_path, texts)

tem = extract_embeddings(model_path, template)
emb = []

for i, embedding in enumerate(embeddings):

    emb.append(sum(np.array(embedding)))

tem = sum(np.array(tem[0]))
import gc

gc.collect()
from sklearn.metrics import mean_squared_error

for i,e in enumerate(emb):

    print('{}:{}'.format(texts[i],mean_squared_error(e,tem)))
for i,e in enumerate(emb):

    score = np.sum(np.array(e) * np.array(tem))/np.linalg.norm(np.array(tem))

    print('{}:{}'.format(texts[i],score))
import sys

import numpy as np

from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths



print('This demo demonstrates how to load the pre-trained model and check whether the two sentences are continuous')



if len(sys.argv) == 2:

    model_path = sys.argv[1]

else:

    from keras_bert.datasets import get_pretrained, PretrainedList

    model_path = get_pretrained(PretrainedList.multi_cased_base) #multi_cased_base;chinese_base



paths = get_checkpoint_paths(model_path)



model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, training=True, seq_len=None)

model.summary(line_length=120)

token_dict = load_vocabulary(paths.vocab)

token_dict_inv = {v: k for k, v in token_dict.items()}



tokenizer = Tokenizer(token_dict)

text = 'as well as meals the restaurant sells a range of drinks'

tokens = tokenizer.tokenize(text)

tokens[11] = tokens[12] = '[MASK]'

print('Tokens:', tokens)



indices = np.array([[token_dict[token] for token in tokens]])

segments = np.array([[0] * len(tokens)])

masks = np.array([[0, 1, 1] + [0] * (len(tokens) - 3)])



predicts = model.predict([indices, segments, masks])[0].argmax(axis=-1).tolist()

print('Fill with: ', list(map(lambda x: token_dict_inv[x], predicts[0][11:13])))
sentence_1 = 'as well as meals the restaurant sells a range of drinks'

sentence_2 = 'these include beers, red wines, white wines and soft drinks'

print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))

indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2)

masks = np.array([[0] * len(indices)])



predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]

print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))
sentence_2 = 'you should include some attributes that you think appropriate price.'

print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))

indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2)

masks = np.array([[0] * len(indices)])



predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]

print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))
sentence_1 = '数学是利用符号语言研究數量、结构、变化以及空间等概念的一門学科。'

sentence_2 = '从某种角度看屬於形式科學的一種。'

print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))

indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2)

masks = np.array([[0] * len(indices)])



predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]

print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))



sentence_2 = '任何一个希尔伯特空间都有一族标准正交基。'

print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))

indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2)

masks = np.array([[0] * len(indices)])



predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]

print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))