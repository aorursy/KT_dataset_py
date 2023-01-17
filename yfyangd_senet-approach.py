# Import Tensorflow & Keras
from tensorflow import keras
from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Import DataSets
from keras.datasets import cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# dict
dictionary={1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck",0:"airplane"}
# visualizing training samples
plt.figure(figsize=(15,5))
for i in range(40):  
    plt.subplot(4, 10, i+1)
    plt.imshow(train_images[i].reshape((32, 32, 3)),cmap=plt.cm.hsv)
    plt.title(dictionary[train_labels[i][0]])
    plt.axis('off')
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()
# Normalize pixel values to be between 0 and 1
train_images.astype('float32');test_images.astype('float32')
train_images, test_images = train_images / 255.0, test_images / 255.0
# Encoding
train_labels = to_categorical(np.array(train_labels[:, 0]))
test_labels = to_categorical(np.array(test_labels[:, 0]))
def squeeze_excite_block(input, ratio=16):
    
    init = input
    filters = init._keras_shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = multiply([init, se])
    
    return x
def resnet_block(input, filters, k=1, strides=(1, 1)):
    init = input
    channel_axis = -1 

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)

    if strides != (1, 1) or init._keras_shape[channel_axis] != filters * k:
        init = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, init])
    return m
def create_se_resnet(classes, img_input, include_top, initial_conv_filters, filters,
                      depth, width, weight_decay, pooling):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    N = list(depth)

    x = Conv2D(initial_conv_filters, (7, 7), padding='same', use_bias=False, strides=(2, 2),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    for i in range(N[0]):
            x = resnet_block(x, filters[0], width)

    for k in range(1, len(N)):
            x = resnet_block(x, filters[k], width, strides=(2, 2))

    for i in range(N[k] - 1):
            x = resnet_block(x, filters[k], width)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)


    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                  activation='softmax')(x)
    
    return x
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)
initial_conv_filters=64
depth=[2, 2, 2, 2]
filters=[64, 128, 256, 512]
width=1
weight_decay=1e-4
include_top=True
weights=None
input_tensor=None
pooling=None
classes=10

img_input = Input(shape=input_shape)
   
x = create_se_resnet(classes, img_input, include_top, initial_conv_filters,
                          filters, depth, width, weight_decay, pooling)

model = Model(img_input, x, name='resnext')
print('model created')
    
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_images, train_labels, epochs=20, validation_split=0.25)
plt.style.use('seaborn')
plt.figure(figsize = (16,8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training Result',fontsize=20)
plt.ylabel('Loss',fontsize=16)
plt.xlabel('Epoch',fontsize=16)
plt.legend(['accuracy','Validation_accuracy'], loc='lower right',fontsize=16)
plt.show()
score = model.evaluate(test_images, test_labels, verbose=0)
print('accuracy: ',score[1])
print('loss: ',score[0])
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
model.compile(optimizer='Nadam',loss='categorical_crossentropy',metrics=['accuracy'])
history_Nadam = model.fit(train_images, train_labels, epochs=20, validation_split=0.25,callbacks=[earlyStopping])
plt.style.use('seaborn')
plt.figure(figsize = (16,8))
plt.plot(history_Nadam.history['accuracy'])
plt.plot(history_Nadam.history['val_accuracy'])
plt.title('Training Result',fontsize=20)
plt.ylabel('Loss',fontsize=16)
plt.xlabel('Epoch',fontsize=16)
plt.legend(['accuracy','Validation_accuracy'], loc='lower right',fontsize=16)
plt.show()
score = model.evaluate(test_images, test_labels, verbose=0)
print('accuracy: ',score[1])
print('loss: ',score[0])
import math
from keras.callbacks import Callback
from keras import backend as K

class CosineAnnealingScheduler(Callback):

    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
callbacks = [CosineAnnealingScheduler(T_max=100, eta_max=1e-2, eta_min=1e-4)]
model.compile(optimizer='Nadam',loss='categorical_crossentropy',metrics=['accuracy'])
history_Nadam_cosLR = model.fit(train_images, train_labels, epochs=20, validation_split=0.25,callbacks=callbacks)
plt.style.use('seaborn')
plt.figure(figsize = (16,8))
plt.plot(history_Nadam_cosLR.history['accuracy'])
plt.plot(history_Nadam_cosLR.history['val_accuracy'])
plt.title('Training Result',fontsize=20)
plt.ylabel('Loss',fontsize=16)
plt.xlabel('Epoch',fontsize=16)
plt.legend(['accuracy','Validation_accuracy'], loc='lower right',fontsize=16)
plt.show()
score = model.evaluate(test_images, test_labels, verbose=0)
print('accuracy: ',score[1])
print('loss: ',score[0])
import matplotlib.pyplot as plot
import math
label_desc = [ 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]
def show_feature_label_prediction( features
                                 , labels
                                 , predictions
                                 , indexList
                                 ) :
    num = len(indexList)
    plot.gcf().set_size_inches( 2*5, (2+0.4)*math.ceil(num/5) )
    loc = 0
    for i in indexList :
        loc += 1
        subp = plot.subplot( math.ceil(num/5), 5, loc )
        subp.imshow( features[i], cmap='binary' )
        if( len(predictions) > 0 ) :
            title = 'ai = ' + label_desc[ predictions[i] ]
            title += (' (o)' if predictions[i]==labels[i] else ' (x)')
            title += '\nlabel = ' + label_desc[ labels[i] ]
        else :
            title = 'label = ' + label_desc[ labels[i] ]
        subp.set_title( title, fontsize=12 )
        subp.set_xticks( [] )
        subp.set_yticks( [] )
    plot.show()
(train_images2, train_labels2), (test_images2, test_labels2) = cifar10.load_data()
predict = model.predict(test_images)
predict=np.argmax(predict,axis=1)
test_label_onearr = test_labels2.reshape(len(test_labels2))
show_feature_label_prediction(test_images, test_label_onearr, predict, range(0, 10) )
checkList = pd.DataFrame( {'label':test_label_onearr,'prediction':predict})
show_feature_label_prediction(test_images, test_label_onearr, predict, checkList.index[checkList.prediction != checkList.label][0:20])