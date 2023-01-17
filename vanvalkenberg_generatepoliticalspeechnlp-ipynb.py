####################################################################
#               Lstm based story Writer                            #
####################################################################
# Generates some texts till 15 words                               #
# just give a seed sentence consisting 10 words in seed Variable   #
####################################################################
######################################################################
#                      Approach                                      #
######################################################################
# The text is first converted to a sequence then LSTM is used        #
# To find the next word in the sequence                              #
######################################################################
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
## making imports
import matplotlib.pyplot as plt
import sklearn 
import tensorflow as tf
CorPus = ""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file = open(dirname+'/'+filename, "r")
        CorPus = CorPus + file.read() 
        file.close()

#print (CorPus)



## Plotting wordCloud for this CorPus
from wordcloud import WordCloud, STOPWORDS 
stopwords = set(STOPWORDS) 
wordcloud = WordCloud(width = 800, height = 800,background_color ='grey', stopwords = stopwords,  min_font_size = 10).generate(CorPus)
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.rcParams.update({'font.size': 25})
plt.axis("off") 
plt.title('Word Cloud: DJT Rallies ')
plt.tight_layout(pad = 0) 
  
plt.show() 
## Converting Corpus to list
CorPusList = list(CorPus.lower().split())
print (CorPusList[0: 50])
## Lets convert our Corpus to a Sequence
UniqueWords = set(CorPusList)
MapOfCorPus = {}
InverseMapOfCorpus = {}
i = 0

for mem in UniqueWords:
    MapOfCorPus[mem] = i 
    InverseMapOfCorpus[i] = mem
    i = i + 1

#print ('Corpus map:\n{}'.format(MapOfCorPus))

## Now Using this Map Converting our Corpus to a Sequence
SequenceOfCoprpus = []
for mem in CorPusList:
    SequenceOfCoprpus.append(MapOfCorPus[mem])

#print('Corpus Seuence:\n{}'.format(SequenceOfCoprpus))
XX = np.array(SequenceOfCoprpus)
Xnorm = (XX- XX.mean())/ XX.std()
#print(Xnorm)
SequenceOfCoprpus = Xnorm
## Now since we have  transformed our Corpus to a seuence we might use to train a Rnn 
## using Past 10 values for each prediction
T = 10
D = 1
X = []
Y = []

for t in range(len(SequenceOfCoprpus) - T):
  x = SequenceOfCoprpus[t:t+T]
  X.append(x)
  y = SequenceOfCoprpus[t+T]
  Y.append(y)


X = np.array(X).reshape(-1, T, 1) 
Y = np.array(Y)
N = len(X)

MyModel = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape = (T, D)),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.LSTM(150),
    #tf.keras.layers.Embedding(20000, 20),
    tf.keras.layers.LSTM(150, return_sequences = True,kernel_regularizer='l2'),
    tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Dense(1)
])
MyModel.compile(
  loss='mse',
  optimizer='adam',
)
retVal = MyModel.fit(
  X, Y,
  batch_size=25,
  epochs=30,
)
Seed = "i don't do it for the money i've got enough"
def transformer(Seed):
    SeedList = list(Seed.split())
    retList = []
    
    for mem in SeedList:
        retList.append((MapOfCorPus[mem] - XX.mean()) / XX.std() )
    
    return retList


    
## lets make some predictions
predictions = []
PredVect = np.array(transformer(Seed))
#PredVect = std.transform(PredVect.reshape(-1,1))


while len(predictions) < 15:
    word = MyModel.predict(PredVect.reshape(1, T, 1))[0,0]
    the_word = InverseMapOfCorpus[int (word*XX.std() + XX.mean())]
    #print(the_word)
    
    predictions.append(the_word)
    
    PredVect = np.roll(PredVect, -1)
    PredVect[-1] = word
    
def printPrediction(arr):
    for mem in arr:
        print(mem, end = " ")
printPrediction(predictions)
MyModel.save('NLP.h5')
plt.plot(retVal.history['loss'], label = 'training_loss')
plt.legend()
plt.grid(True)