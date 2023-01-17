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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/cleaned-dataset-comprehensive/Cleaned_dataset_comprehensive.csv")
train
tweets = []
labels = []
for i, j in train.iterrows():
  if (j['freq']) >= 928:
    tweets.append(j['text'])
    labels.append(j['emoji'])
labels_remapped = []
output = []
for x in labels:
    if x not in output:
        output.append(x)
index_map = {}
i = 0;
for x in output:
  index_map[x] = i;
  i+=1;
for x in labels:
  labels_remapped.append(index_map[x])
x_train = np.array(tweets)
y_train = np.array(labels_remapped)
x_train.shape
import os
f = open('/kaggle/input/glove6b/glove.6B.50d.txt', encoding='utf8')
embedding_index = {}

for line in f:
    values = line.split()
    word = values[0]
    emb = np.array(values[1:], dtype ='float')
    embedding_index[word] = emb
def get_embedding_output(X):
    maxLen = 20
    embedding_output = np.zeros((len(X), maxLen, 50))
    
    for ix in range(X.shape[0]):
        my_example = X[ix].split()
        
#         print(my_example)       
        for ij in range(len(my_example)): 
            if (embedding_index.get(my_example[ij].lower()) is not None) and (ij<maxLen):
                embedding_output[ix][ij] = embedding_index[my_example[ij].lower()]
            
    return embedding_output
x_train_embed = get_embedding_output(x_train)
x_train_embed.shape
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_train[2]
from imblearn.over_sampling import SMOTE
x_train_embed.shape
x_train_embed = x_train_embed.reshape(-1, 1000)
y_train
oversample = SMOTE()
X, y = oversample.fit_resample(x_train_embed, y_train)
X = X.reshape(-1, 20, 50)
X.shape, y.shape
from sklearn.model_selection import train_test_split
x1, x2, y1, y2 = train_test_split(X, y, test_size=0.2, random_state=42)
# for i in range(0, 5):
#     print(x_train[i], mapp[y_train[i]])
x1.shape
from keras.models import Sequential 
from keras.layers import LSTM, Dense, Dropout, Bidirectional
model = Sequential()
model.add(Bidirectional(LSTM(units = 512, return_sequences=True), input_shape = (20,50)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=256)))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=20, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics =['accuracy'])
hist = model.fit(x1, y1, validation_split=0.2, shuffle=True, batch_size=64, epochs=25)
model.evaluate(x2, y2)
