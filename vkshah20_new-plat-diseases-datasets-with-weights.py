import tensorflow as tf

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import random



from tensorflow.keras import Sequential

from tensorflow.keras.applications.resnet import ResNet50

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout

from tensorflow.keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.initializers import HeNormal

sns.set()
PATH = '/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)'

total = 0

for dirs, dis, files in os.walk(os.path.join(PATH, 'train')):

    dis.sort()

    total += len(files)

print('total image in train datasets', total)



total = 0

for dirs, dis, files in os.walk(os.path.join(PATH, 'valid')):

    dis.sort()

    total += len(files)

print('total image in valid datasets', total)
def getDF(labels, subpath, df):

    for i, (dirs, dist, files) in enumerate(os.walk(os.path.join(PATH, subpath))):

        if i == 0:

            continue

        dist.sort()

        lst = [os.path.join(dirs, f) for f in files]

        ones, zeros = np.ones(len(lst), dtype = np.int8), labels[i] + np.zeros(len(lst), dtype = np.int8)

        dft = pd.DataFrame(list(zip(lst,zeros)), columns = df.columns)

        df = df.append(dft, ignore_index = True)

    df['label'] = df['label'].astype('str')

    return df

        

    

labels = ['bacteria', 'blignt', 'disease', 'healthy', 'rust', 'powdery']

labelsVal = [-1,2,1,4,3,3,5,3,5,4,1,3,1,2,1,3,2,0,3,0,3,1,1,3,3,3,5,2,3,0,1,1,2,1,2,2,2,2,3]



df = pd.DataFrame(columns = ['img','label'])

df = getDF(labelsVal, 'train', df)

df = getDF(labelsVal, 'valid', df)

def splitDataSets(train, val, test, df, verb = False):

    lenght = len(df)

    bincount = np.bincount(df['label'].astype(int))

    minVal = bincount[np.argmin(bincount)]

    testMin = int(test * minVal)

    valMin = int(val * minVal)

    

    testDF = pd.DataFrame(columns = df.columns)

    valDF = pd.DataFrame(columns = df.columns)

    trainDF = pd.DataFrame(columns = df.columns)

    

    for i in range(6):

        tdf = pd.DataFrame(columns = df.columns)

        tdf = tdf.append(df[df['label'] == str(i)], ignore_index = True)

        idx = np.random.choice(len(tdf), testMin + valMin, False)

        

        testDF = testDF.append(tdf.loc[:int(testMin)], ignore_index = True)

        valDF = valDF.append(tdf.loc[int(testMin): len(idx)], ignore_index = True)

        trainDF = trainDF.append(tdf.loc[len(idx):], ignore_index = True)

        

    if verb == True:

        print(trainDF.shape, np.bincount(trainDF['label']))

        print(valDF.shape, np.bincount(valDF['label']))

        print(testDF.shape, np.bincount(testDF['label']))

    

    return trainDF, valDF, testDF

df = df.sample(frac = 1)

df.head(15)

trainDF, valDF, testDF = splitDataSets(0.85, 0.1, 0.05, df)



plt.bar(np.arange(6), np.bincount(trainDF['label']), tick_label  = labels, color = ['r','g','b','y','m','c'])

plt.figure()

plt.bar(np.arange(6), np.bincount(valDF['label']), tick_label  = labels, color = ['r','g','b','y','m','c'])

plt.figure()

plt.bar(np.arange(6), np.bincount(testDF['label']), tick_label  = labels, color = ['r','g','b','y','m','c'])
class_weights = len(trainDF)  / (len(labels) * np.bincount(trainDF['label']))

plt.title('Class weights')

plt.bar(np.arange(6), class_weights, tick_label  = labels, color = ['r','g','b','y','m','c'])



class_weight = {}

for i in range(6):

    class_weight[i] = class_weights[i]
trainGen = ImageDataGenerator(rescale = 1./255,

                          rotation_range = 0.2,

                          width_shift_range = 0.1,

                          height_shift_range = 0.1,

                          zoom_range = 0.25,

                          horizontal_flip = True,

                          vertical_flip = True)

testGen = ImageDataGenerator(rescale = 1./255)



trainFlow = trainGen.flow_from_dataframe(trainDF, x_col = 'img', y_col = 'label', validate_filenames = True)

valFlow = testGen.flow_from_dataframe(valDF, x_col = 'img', y_col = 'label', validate_filenames = True)

testFlow = testGen.flow_from_dataframe(testDF, x_col = 'img', y_col = 'label', validate_filenames = True)
res = ResNet50(include_top = False, input_shape = (256, 256, 3))

flag = False

for l in res.layers:

    if l.name == 'conv5_block1_1_conv':

        flag = True

    l.trainable = flag

he_weights = HeNormal()

model = Sequential()

model.add(res)

model.add(Conv2D(512, (1,1), activation = 'relu', kernel_initializer = he_weights))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(1024, activation = 'relu', kernel_initializer = he_weights))

model.add(Dense(128, activation = 'relu', kernel_initializer = he_weights))

model.add(Dense(6, activation = 'softmax', kernel_initializer = he_weights))

model.summary()

model.compile('adam', 'categorical_crossentropy', ['acc', 'Precision', 'AUC'])
logdir = 'logs'



class My_callback(Callback):

    def on_epoch_end(self, epoch, logs = {}):

        if (logs['acc'] > 0.93):

            self.model.stop_training = True    



esCallback = EarlyStopping(monitor = 'acc', patience = 5)

mcCallback = ModelCheckpoint(filepath = 'modelOn.hdf5', monitor = 'val_loss', save_best_only = True)

lrCallback = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3)

tbCallback = TensorBoard(log_dir = logdir)

csCallback = My_callback()
history = model.fit(trainFlow,

         epochs = 30,

         validation_data = valFlow,

         callbacks = [tbCallback,csCallback, lrCallback, esCallback, mcCallback], 

         class_weight = class_weight

)
plt.figure()

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
plt.figure()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
plt.figure()

plt.plot(history.history['auc'])

plt.plot(history.history['val_auc'])

plt.title('model AUC')

plt.ylabel('AUC')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
plt.figure()

plt.plot(history.history['precision'])

plt.plot(history.history['val_precision'])

plt.title('model Precision')

plt.ylabel('Precision')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()