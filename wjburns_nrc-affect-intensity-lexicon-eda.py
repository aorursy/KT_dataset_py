# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra

# fix random seed for reproducibility

seed = 7

np.random.seed(seed)



from sklearn.metrics import confusion_matrix

import itertools
              

# import matplotlib as mp

import matplotlib.pyplot as plt

#from sklearn import preprocessing

#from sklearn.preprocessing import StandardScaler



# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth

import tensorflow as tf 

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

print(tf.test.gpu_device_name())

sess = tf.Session()



# Keras Imports

from keras.models import Sequential

from keras.layers import Dense, BatchNormalization, Dropout

from keras.layers import LeakyReLU

from keras.callbacks import Callback

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.utils import plot_model
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
# Plot Confusion Matrix Function



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
df = pd.read_csv('../input/NRC-AffectIntensity-Lexicon.csv')

df.count()
df.head()
# CREATE 4 BIT KEY

d = {'WORD':[], 'ANGER':[], 'FEAR':[], 'SADNESS':[], 'JOY':[]}

train = pd.DataFrame(data=d)



train['WORD'] = df['term'] 



train.head()
import math
anger = df[df['emotion'].str.contains("anger")]

anger_key = pd.DataFrame()

anger_key['word'] = anger['term']

anger_key['score'] = anger['score'].apply(math.ceil)

train['ANGER'] = anger['score'].apply(math.ceil)



anger_key = anger_key.reset_index(drop=True)

anger_key.head()
fear = df[df['emotion'].str.contains("fear")]

fear_key = pd.DataFrame()

fear_key['word'] = fear['term']

fear_key['score'] = fear['score'].apply(math.ceil)

train['FEAR'] = fear['score'].apply(math.ceil)



fear_key = fear_key.reset_index(drop=True)

fear_key.head()
sadness = df[df['emotion'].str.contains("sadness")]

sadness_key = pd.DataFrame()

sadness_key['word'] = sadness['term']

sadness_key['score'] = sadness['score'].apply(math.ceil)

train['SADNESS'] = sadness['score'].apply(math.ceil)



sadness_key = sadness_key.reset_index(drop=True)

sadness_key.head()
joy = df[df['emotion'].str.contains("joy")]

joy_key = pd.DataFrame()

joy_key['word'] = joy['term']

joy_key['score'] = joy['score'].apply(math.ceil)

train['JOY'] = joy['score'].apply(math.ceil)



joy_key = joy_key.reset_index(drop=True)

joy_key.head()
train = train.fillna(0)

train.head()
train.head()
# CREATE KEY

Y = train.drop(columns=['WORD'])

Y.head()
# CREATE 16 BIT KEY

#def character_encoder:

X = {'CHR1' :[], 'CHR2' :[], 'CHR3' :[], 'CHR4' :[], 'CHR5' :[],

     'CHR6' :[], 'CHR7' :[], 'CHR8' :[], 'CHR9' :[], 'CHR10':[],

     'CHR11':[], 'CHR12':[], 'CHR13':[], 'CHR14':[], 'CHR15':[],

     'CHR16':[]

    }

#, 'CHR17':[], 'CHR18':[], 'CHR19':[], 'CHR20':[]

#    }

X = pd.DataFrame(data=d)



def split(word): 

    return [char for char in word]  

      

X['CHR1'] = train['WORD'].str.slice(stop=1).fillna(0)

X['CHR2'] = train['WORD'].str.slice(start=1,stop=2).fillna(0)

X['CHR3'] = train['WORD'].str.slice(start=2,stop=3).fillna(0)

X['CHR4'] = train['WORD'].str.slice(start=3,stop=4).fillna(0)

X['CHR5'] = train['WORD'].str.slice(start=4,stop=5).fillna(0)

X['CHR6'] = train['WORD'].str.slice(start=5,stop=6).fillna(0)

X['CHR7'] = train['WORD'].str.slice(start=6,stop=7).fillna(0)

X['CHR8'] = train['WORD'].str.slice(start=7,stop=8).fillna(0)

# X['CHR9'] = train['WORD'].str.slice(start=8,stop=9).fillna(0)

# X['CHR10'] = train['WORD'].str.slice(start=9,stop=10).fillna(0)

# X['CHR11'] = train['WORD'].str.slice(start=10,stop=11).fillna(0)

# X['CHR12'] = train['WORD'].str.slice(start=11,stop=12).fillna(0)

# X['CHR13'] = train['WORD'].str.slice(start=12,stop=13).fillna(0)

# X['CHR14'] = train['WORD'].str.slice(start=13,stop=14).fillna(0)

# X['CHR15'] = train['WORD'].str.slice(start=14,stop=15).fillna(0)

# X['CHR16'] = train['WORD'].str.slice(start=15,stop=16).fillna(0)

# X['CHR17'] = train['WORD'].str.slice(start=16,stop=17).fillna(0)

# X['CHR18'] = train['WORD'].str.slice(start=17,stop=18).fillna(0)

# X['CHR19'] = train['WORD'].str.slice(start=18,stop=19).fillna(0)

# X['CHR20'] = train['WORD'].str.slice(start=19,stop=20).fillna(0)



X = X.drop(columns=['WORD','ANGER','FEAR','SADNESS','JOY'])



X.head()

X = X.replace(r'\s+( +\.)|#',np.nan,regex=True).replace('',np.nan)

X.head()
X = X.fillna(' ')

X.head()
floor = 32

limit = 100



X['CHR1'] = (X['CHR1'].apply(ord) - floor) / limit

X['CHR2'] = (X['CHR2'].apply(ord) - floor) / limit 

X['CHR3'] = (X['CHR3'].apply(ord) - floor) / limit

X['CHR4'] = (X['CHR4'].apply(ord) - floor) / limit

X['CHR5'] = (X['CHR5'].apply(ord) - floor) / limit

X['CHR6'] = (X['CHR6'].apply(ord) - floor) / limit

X['CHR7'] = (X['CHR7'].apply(ord) - floor) / limit

X['CHR8'] = (X['CHR8'].apply(ord) - floor) / limit

# X ['CHR9'] = (X['CHR9'].apply(ord) - floor) / limit

# X['CHR10'] = (X['CHR10'].apply(ord) - floor) / limit

# X['CHR11'] = (X['CHR11'].apply(ord) - floor) / limit

# X['CHR12'] = (X['CHR12'].apply(ord) - floor) / limit

# X['CHR13'] = (X['CHR13'].apply(ord) - floor) / limit

# X['CHR14'] = (X['CHR14'].apply(ord) - floor) / limit

# X['CHR15'] = (X['CHR15'].apply(ord) - floor) / limit

# X['CHR16'] = (X['CHR16'].apply(ord) - floor) / limit

# X['CHR17'] = (X['CHR17'].apply(ord) - floor) / limit

# X['CHR18'] = (X['CHR18'].apply(ord) - floor) / limit

# X['CHR19'] = (X['CHR19'].apply(ord) - floor) / limit

# X['CHR20'] = (X['CHR20'].apply(ord) - floor) / limit
X.head()
Y.head()
# CNN MODEL

word_dim = X.shape[1]

        

model = Sequential()



model.add(Dense(64, input_dim=word_dim, activation='relu'))

# model.add(Dense(64, activation='relu'))

# model.add(Dense(128, activation='relu'))

# model.add(Dense(256, activation='relu'))



# model.add(Dense(512, activation='relu'))





# model.add(Dense(1024, activation='relu'))



model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

# model.add(Dropout(0.1))

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

# model.add(Dropout(0.1))

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

# model.add(Dropout(0.1))



model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(16, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(4, activation='softmax'))

model.summary()





# model.add(Dense(1024))

# model.add(LeakyReLU(alpha=0.1))

# model.add(BatchNormalization())

# model.add(Dropout(0.1))

# 1.4 EXTENDED KERAS MODEL CHECKPOINT CALLBACK

class GetBest(Callback):

    def __init__(self, monitor='val_loss', verbose=0,

                 mode='auto', period=1):

        super(GetBest, self).__init__()

        self.monitor = monitor

        self.verbose = verbose

        self.period = period

        self.best_epochs = 0

        self.epochs_since_last_save = 0



        if mode not in ['auto', 'min', 'max']:

            warnings.warn('GetBest mode %s is unknown, '

                          'fallback to auto mode.' % (mode),

                          RuntimeWarning)

            mode = 'auto'



        if mode == 'min':

            self.monitor_op = np.less

            self.best = np.Inf

        elif mode == 'max':

            self.monitor_op = np.greater

            self.best = -np.Inf

        else:

            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):

                self.monitor_op = np.greater

                self.best = -np.Inf

            else:

                self.monitor_op = np.less

                self.best = np.Inf



    def on_train_begin(self, logs=None):

        self.best_weights = self.model.get_weights()



    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        self.epochs_since_last_save += 1

        if self.epochs_since_last_save >= self.period:

            self.epochs_since_last_save = 0

            current = logs.get(self.monitor)

            if current is None:

                warnings.warn('\033[1;44m「KERAS」- CLEARED: Can pick best model only with %s available, '

                              'skipping. \033[0m' % (self.monitor), RuntimeWarning)

            else:

                if self.monitor_op(current, self.best):

                    if self.verbose > 0:

                        print('\033[1;42m「KERAS」- PASSED: Epoch %05d: %s improved from %0.5f to %0.5f,'

                              ' storing weights. \033[0m' % (epoch + 1, self.monitor, self.best,

                                 current))

                    self.best = current

                    self.best_epochs = epoch + 1

                    self.best_weights = self.model.get_weights()

                else:

                    if self.verbose > 0:

                        print('\033[1;41m「KERAS」- FAILED: Epoch %05d: %s did not improve \033[0m' %

                              (epoch + 1, self.monitor))



    def on_train_end(self, logs=None):

        if self.verbose > 0:

            print('\n\033[1;7m「KERAS」- LEVEL: Using epoch %05d with %s: %0.5f \033[0m' % 

                  (self.best_epochs, self.monitor, self.best))

        self.model.set_weights(self.best_weights)
def make_plot(data, metric="loss"):

    # Data for plotting

    data1 = data[0]



    t = np.arange(1,len(data1)+1,1)

    plt.figure(figsize=(10,5))

    plt.plot(t, data1)



    plt.xlabel('epoch')

    plt.ylabel(metric)

    plt.title('Train' + metric)

    plt.grid()

    plt.legend(['train'], ncol=2, loc='upper right');

    plt.savefig("train_"+metric+".png", dpi=300)

    plt.show()
# Compile Model

model.compile(loss='categorical_crossentropy',

              metrics=['accuracy'],

              optimizer='adam')



callbacks = [EarlyStopping(monitor='acc', patience=100),

             GetBest(monitor='acc', verbose=1, mode='max')]
history = model.fit(X, Y,

                    callbacks=callbacks,

                    epochs=9999,

                    batch_size=256,

                    shuffle=True,

                    #validation_split=0.1,

                    verbose=2)
loss = history.history['loss'];acc = history.history['acc']
make_plot([loss]);make_plot([acc], metric="acc")
numeric_train = pd.concat([X,Y], axis=1)
numeric_train.head()
corr_matrix = numeric_train.corr()

corr_matrix['JOY'].sort_values(ascending = False)
import seaborn as sb

plt.figure(figsize=(20,15))

sb.heatmap(corr_matrix)

plt.title("Correlation Matrix", size = 25)
# 5. evaluate the model

crc_score = model.evaluate(X, Y, verbose=1)

print('CRC Test loss:', crc_score[0])

print('CRC Test accuracy:', crc_score[1])
mp_X = pd.DataFrame(model.predict(X)).apply(round)

model_prediction = pd.concat([train,mp_X], axis = 1 ) 

print('COMPLETE')

model_prediction[0:100]