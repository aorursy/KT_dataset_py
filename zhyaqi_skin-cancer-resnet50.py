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
import pandas as pd 



df = pd.read_csv('../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')

df.head()
from os.path import isfile

from PIL import Image as pil_image

df['num_images'] = df.groupby('lesion_id')["image_id"].transform("count")



classes = df['dx'].unique()

labeldict = {}

for num, name in enumerate(classes):

    labeldict[name] = num

df['dx_id'] = df['dx'].map(lambda x: labeldict[x])





def expand_path(p):

    if isfile('../input/skin-cancer-mnist-ham10000/ham10000_images_part_1/' + p + '.jpg'): return '../input/skin-cancer-mnist-ham10000/ham10000_images_part_1/' + p + '.jpg'

    if isfile('../input/skin-cancer-mnist-ham10000/ham10000_images_part_2/' + p + '.jpg'): return '../input/skin-cancer-mnist-ham10000/ham10000_images_part_2/' + p + '.jpg'

    return p 

df['image_path'] = df['image_id']

df['image_path'] = df['image_path'].apply(expand_path)





df['images'] = df['image_path'].map(lambda x: np.asarray(pil_image.open(x).resize((150,112))))

df.head()

from sklearn.model_selection import train_test_split



df_single = df[df['num_images'] == 1]

trainset1, testset = train_test_split(df_single, test_size=0.2,random_state = 80)

trainset2, validationset = train_test_split(trainset1, test_size=0.2,random_state = 600)

trainset3 = df[df['num_images'] != 1]

frames = [trainset2, trainset3]

trainset = pd.concat(frames)

def prepareimages(images):

    # images is a list of images

    images = np.asarray(images).astype(np.float64)

    images = images[:, :, :, ::-1]

    m0 = np.mean(images[:, :, :, 0])

    m1 = np.mean(images[:, :, :, 1])

    m2 = np.mean(images[:, :, :, 2])

    images[:, :, :, 0] -= m0

    images[:, :, :, 1] -= m1

    images[:, :, :, 2] -= m2

    return images

trainimages = prepareimages(list(trainset['images']))

testimages = prepareimages(list(testset['images']))

validationimages = prepareimages(list(validationset['images']))

trainlabels = np.asarray(trainset['dx_id'])

testlabels = np.asarray(testset['dx_id'])

validationlabels = np.asarray(validationset['dx_id'])
import matplotlib.pyplot as plt



plt.hist(trainlabels,bins = 7,density = True)

plt.show()



plt.hist(validationlabels,bins = 7,density= True)

plt.show()



plt.hist(testlabels,bins = 7,density= True)

plt.show()
from keras.preprocessing.image import ImageDataGenerator



trainimages = trainimages.reshape(trainimages.shape[0], *(112, 150, 3))



data_gen = ImageDataGenerator(

        rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally

        height_shift_range=0.1)  # randomly shift images vertically

#x = imageLoader(trainset,batch_size)

data_gen.fit(trainimages)

#from keras.applications.resnet50 import ResNet50

from tensorflow.python.keras.applications import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Dropout

from keras import regularizers

import numpy as np



model = Sequential()

num_labels = 7



base_model = ResNet50(include_top=False, input_shape=(112, 150, 3),pooling = 'avg', weights = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

model = Sequential()

model.add(base_model)

model.add(Dropout(0.5))

model.add(Dense(128, activation="relu",kernel_regularizer=regularizers.l2(0.02)))

model.add(Dropout(0.5))

model.add(Dense(num_labels, activation = 'softmax',kernel_regularizer=regularizers.l2(0.02)))



for layer in base_model.layers:

    layer.trainable = False



for layer in base_model.layers[-22:]:

    layer.trainable = True



#model.add(ResNet50(include_top = False, pooling = 'max', weights = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'))



model.summary()
from tensorflow.python.keras.optimizers import Adam

optimizer = Adam (lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)

model.compile(optimizer = optimizer , loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
# Fit the model

import keras

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

class CustomModelCheckPoint(keras.callbacks.Callback):

    def __init__(self,**kargs):

        super(CustomModelCheckPoint,self).__init__(**kargs)

        self.epoch_accuracy = {} # loss at given epoch

        self.epoch_loss = {} # accuracy at given epoch

        def on_epoch_begin(self,epoch, logs={}):

            # Things done on beginning of epoch. 

            return



        def on_epoch_end(self, epoch, logs={}):

            # things done on end of the epoch

            self.epoch_accuracy[epoch] = logs.get("acc")

            self.epoch_loss[epoch] = logs.get("loss")

            self.model.save_weights("../output/resnet50/name-of-model-%d.h5" %epoch)

            

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = 4)

cb_checkpointer = ModelCheckpoint(filepath = '../working/best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')

            

epochs = 30 

batch_size = 20

trainhistory = model.fit_generator(data_gen.flow(trainimages,trainlabels, batch_size=batch_size),

                              epochs = epochs, validation_data = (validationimages,validationlabels),

                              verbose = 1, steps_per_epoch=trainimages.shape[0] // batch_size,

                                       callbacks=[cb_checkpointer, cb_early_stopper])
import matplotlib.pyplot as plt

acc = trainhistory.history['acc']

val_acc = trainhistory.history['val_acc']

loss = trainhistory.history['loss']

val_loss = trainhistory.history['val_loss']

epochs = range(1, len(acc) + 1)



plt.plot(epochs, loss, '', label='Training loss')

plt.plot(epochs, val_loss, '', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.figure()



plt.plot(epochs, acc, '', label='Training accuracy')

plt.plot(epochs, val_acc, '', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()
model.load_weights("../working/best.hdf5")

test_loss, test_acc = model.evaluate(testimages, testlabels, verbose=1)

print("test_accuracy = %f  ;  test_loss = %f" % (test_acc, test_loss))
from sklearn.metrics import confusion_matrix

train_pred = model.predict(trainimages)

train_pred_classes = np.argmax(train_pred,axis = 1)

test_pred = model.predict(testimages)

# Convert predictions classes to one hot vectors 

test_pred_classes = np.argmax(test_pred,axis = 1) 



confusionmatrix = confusion_matrix(testlabels, test_pred_classes)

confusionmatrix
from sklearn.metrics import classification_report

labels = labeldict.keys()

# Generate a classification report

trainreport = classification_report(trainlabels, train_pred_classes, target_names=list(labels))

testreport = classification_report(testlabels, test_pred_classes, target_names=list(labels))



print(trainreport)

print(testreport)