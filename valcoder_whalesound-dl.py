# ignore warnings
import warnings
warnings.filterwarnings("ignore")
# importing multiple visualization libraries
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
import pylab as pl
import seaborn
# importing libraries to manipulate the data files
import os
from glob import glob
# import a library to read the .aiff format
import aifc
filenames = glob(os.path.join('../input/whaledatatrainonly/whale_data_train_only/whale_data_train_only/','train','*.aiff'))
filenames = sorted(filenames)
from scipy import signal
# read signals and apply the welch filter
feature_dict = {}
fs = 2000
for filename in filenames[:10000]:
    aiff = aifc.open(filename,'r')
    whale_strSig = aiff.readframes(aiff.getnframes())
    whale_array = np.fromstring(whale_strSig, np.short).byteswap()
    # apply welch filter
    feature = 10*np.log10(signal.welch(whale_array, fs=fs, window='hanning', nperseg=256, noverlap=128+64)[1])
    feature_dict[filename.split('/')[-1]] = feature
import pandas as pd
# convert to a pandas dataframe
X = pd.DataFrame(feature_dict).T
from __future__ import division
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
labels = pd.read_csv(os.path.join('../input/whaledatatrainonly/whale_data_train_only/whale_data_train_only/','train.csv'), index_col = 0)
# plotting a filtered signal
ticks = signal.welch(whale_array, fs=fs, window='hanning', nperseg=256, noverlap=128+64)[0]
plt.plot(ticks,X.iloc[3,:]) 
plt.xlabel('Frequency')
plt.title('Upcall Example')
plt.xticks()
# y contains the labels
y = np.array(labels['label'][X.index])[:10000]
from sklearn.model_selection import train_test_split
target_names = ['Upcall', 'NO_Upcall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2018)

# Convert label to onehot
#y_train = keras.utils.to_categorical(y_train, num_classes=2)
#y_test = keras.utils.to_categorical(y_test, num_classes=2)

print(X_train.shape)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
# f1 based metric
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(' — val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
        return
 
metrics = Metrics()
# Build the Neural Network
model = Sequential()

model.add(Conv1D(16, 3, activation='relu', input_shape=(129, 1)))
model.add(Conv1D(16, 3, activation='relu'))
model.add(BatchNormalization())

model.add(Conv1D(32, 3, activation='relu'))
model.add(Conv1D(32, 3, activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling1D(2))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling1D(2))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling1D(2))
model.add(Conv1D(256, 3, activation='relu'))
model.add(Conv1D(256, 3, activation='relu'))
model.add(BatchNormalization())
model.add(GlobalAveragePooling1D())

model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])




model_name = 'deep_1'
top_weights_path = 'model_' + str(model_name) + '.h5'

callbacks_list = [ModelCheckpoint(top_weights_path, monitor = 'val_acc', verbose = 1, save_best_only = True, save_weights_only = True), 
    EarlyStopping(monitor = 'val_acc', patience = 6, verbose = 1),
    ReduceLROnPlateau(monitor = 'val_acc', factor = 0.1, patience = 3, verbose = 1),
    CSVLogger('model_' + str(model_name) + '.log')]
#from keras.utils import plot_model
#plot_model(model,to_file='model_plot.png')
#<img src="img/model_plot.png" width=200/>
%%time
# Fitting the Model (this will take a loooooooot of time if run on CPU)
model.fit(X_train, y_train, batch_size=128, epochs=100, validation_data = [X_test, y_test], callbacks = callbacks_list
    )
model.load_weights(top_weights_path)
loss, acc = model.evaluate(X_test, y_test, batch_size=16)
print('Test accuracy:', acc)
# predict
y_pred = model.predict(X_test)
# precision
sum(y_pred[y_test == 1]>.1)/len(y_test == 1)
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test.astype('int'), y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.plot(fpr, tpr)
plt.title('ROC (AUC = %0.2f)' % ( roc_auc))
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(y_test.astype('int'), y_pred)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
from sklearn.metrics import f1_score
f1_score(y_test.astype('int'), (y_pred>0.3).astype(int))