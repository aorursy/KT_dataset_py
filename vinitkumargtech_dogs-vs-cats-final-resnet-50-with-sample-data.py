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
%matplotlib inline
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFile 
import numpy as np
import os 
import cv2
from tqdm import tqdm_notebook
from random import shuffle
import pandas as pd
import random
from tqdm import tqdm
import seaborn as sns
import math
import keras
from keras import applications
from keras import optimizers
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import Callback,ModelCheckpoint
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.models import load_model
from keras.preprocessing.image import load_img
from sklearn.metrics import classification_report
batch_size = 32
epochs =20
num_classes = 2
num_t_samples = 20000
num_v_samples = 5000
path='../input/dogs-vs-cats-redux-kernels-edition/'
dir= '../input/dogscats/data/data/'
train_data_path = path+'train/'
test_data_path =  path+'test/'
train_data_dir = dir+'train/'
validation_data_dir=dir+'validation/'
test_dir='../input/dogscatstest/test1/test1/'
sample_test_dir='../input/sample-testdata/sample test data/sample test data'
resnet50_weight_path='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
#img_size=197
img_size=224
#Process training data to make it ready for fitting.
train_datagen = ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_size, img_size),
                                                    batch_size=batch_size, 
                                                    shuffle=True,seed=42,
                                                    class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(validation_data_dir,
                                                       target_size=(img_size, img_size),
                                                        batch_size=batch_size,
                                                       shuffle=True,seed=42,
                                                       class_mode='categorical') 
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(img_size, img_size),
                                                  batch_size=batch_size,
                                                  class_mode=None,shuffle=False,seed=42)
filename=test_generator.filenames
sample_test_generator=test_datagen.flow_from_directory(sample_test_dir,
                                                  target_size=(img_size, img_size),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',shuffle=False,seed=42)
print ('Creating model...')
resnet50_model = Sequential()
resnet50_model.add(ResNet50(include_top=False, pooling='max', weights=resnet50_weight_path))
resnet50_model.add(Dense(num_classes, activation='softmax'))
resnet50_model.layers[0].trainable = True
print ('Summary of the model...')
resnet50_model.summary()
print ('Compiling model...')
#resnet50_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
resnet50_model.compile(optimizer=SGD(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
#resnet50_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print ('Model is ready to be fit with training data.')
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
history = LossHistory()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=0, mode='auto')
checkpoint = ModelCheckpoint('resnet50_model_Final.h5',monitor='val_loss', verbose=1, save_best_only=True,mode='auto')
callbacks_list = [checkpoint,history,early_stopping]
fitted_resnet50_model=resnet50_model.fit_generator(train_generator,
    steps_per_epoch=math.ceil(train_generator.samples/train_generator.batch_size),
    epochs=epochs,validation_data=validation_generator,
    validation_steps=math.ceil(validation_generator.samples/validation_generator.batch_size),callbacks=callbacks_list,verbose=1)
acc = fitted_resnet50_model.history['acc']
val_acc = fitted_resnet50_model.history['val_acc']
loss = fitted_resnet50_model.history['loss']
val_loss =fitted_resnet50_model.history['val_loss']
epochs = range(len(acc)) 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#Get the sample test data filenames from the generator
fnames = sample_test_generator.filenames
# Get the ground truth from generator
ground_truth = sample_test_generator.classes
# Get the label to class mapping from the generator
label2index = sample_test_generator.class_indices
# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())
# Get the predictions from the model using the generator
predictions = resnet50_model.predict_generator(sample_test_generator,
steps=math.ceil(sample_test_generator.samples/sample_test_generator.batch_size),verbose=1)
predicted_classes = np.argmax(predictions,axis=1)
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),sample_test_generator.samples))
# Display missclassified images
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]
    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(fnames[errors[i]].split('/')[0],pred_label,
        predictions[errors[i]][pred_class])
    original = load_img('{}/{}'.format(sample_test_dir,fnames[errors[i]]))
    plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()
#get the predictions for the test data
#predicted_classes = model.predict_classes(Xt)
#get the indices to be plotted
#y_true = np.argmax(yt,axis=1)
correct = np.nonzero(predicted_classes==ground_truth)[0]
incorrect = np.nonzero(predicted_classes!=ground_truth)[0]
target_names = ["Class {}:".format(i) for i in range(num_classes)]
print(classification_report(ground_truth, predicted_classes, target_names=target_names))
test_generator.reset()
pred=resnet50_model.predict_generator(test_generator,steps=math.ceil(test_generator.samples/test_generator.batch_size),verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
preds= [labels[k] for k in predicted_class_indices]
new_preds=[]
for i in range(len(preds)):
    if preds[i]=='dogs':
        new_preds.append('dog')
    else:
        new_preds.append('cat')
def display_testdata(testdata,filenames):
    f, ax = plt.subplots(5,5, figsize=(15,15))
    i=0
    for a,b in zip(testdata,filenames):
        pred_label=a
        fname=b
        title = 'Prediction :{}'.format(pred_label)   
        original = load_img('{}/{}'.format(test_dir,fname))
        ax[i//5,i%5].axis('off')
        ax[i//5,i%5].set_title(title)
        ax[i//5,i%5].imshow(original)
        i=i+1
    plt.show()
display_testdata(new_preds[1800:1825],filename[1800:1825])
labels=[]
file_index=[]
for a,b in tqdm(zip(predicted_class_indices,filename)):
    pred_label=a
    fname=b[5:]
    fname=fname.split('.')[0]
    labels.append(pred_label)
    file_index.append(fname)
results=pd.DataFrame({"id":file_index,
                      "label":labels})
results.to_csv("submission.csv",index=False)