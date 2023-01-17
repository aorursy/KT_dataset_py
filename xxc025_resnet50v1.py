from __future__ import print_function

import keras

import keras.layers as layers

from keras.models import Model

from keras.optimizers import Adam

import numpy as np

from matplotlib import pylab as plt

from keras.callbacks import LearningRateScheduler

from keras.callbacks import ReduceLROnPlateau
(x_train,y_train),(x_test,y_test)=keras.datasets.cifar100.load_data()

print(y_train[0])

input_shape=x_train.shape[1:]

print(input_shape)

x_train=x_train.astype('float32')/255

x_test=x_test.astype('float32')/255

x_train_mean=np.mean(x_train,axis=0)

x_train=x_train-x_train_mean

x_test=x_test-x_train_mean

y_train=keras.utils.to_categorical(y_train,100)

y_test=keras.utils.to_categorical(y_test,100)

print(x_train.shape)

print(y_train.shape)

print(y_train[0])

# plt.imshow(x_train[0])

# plt.axis('off')
#Implementation of the popular ResNet50 the following architecture:

# CONV2D -> BATCHNORM -> RELU -> MAXPOOL ->|| CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3

# -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2|| -> AVGPOOL -> TOPLAYER

def conv_block(layer_input,f_size,filters,stage,block,s=2):

    conv_name_base='res'+str(stage)+block+'_branch'

    bn_name_base='bn'+str(stage)+block+'_branch'

  

    F1,F2,F3=filters

    x=layers.Conv2D(F1,kernel_size=(1,1),strides=(s,s),padding='valid',

                  name=conv_name_base+'2a',

                  kernel_initializer='he_normal')(layer_input)

    x=layers.BatchNormalization(name=bn_name_base+'2a')(x)

    x=layers.Activation('relu')(x)

  

    x=layers.Conv2D(F2,kernel_size=(f_size,f_size),strides=(1,1),padding='same',

                  name=conv_name_base+'2b',

                  kernel_initializer='he_normal')(x)

    x=layers.BatchNormalization(name=bn_name_base+'2b')(x)

    x=layers.Activation('relu')(x)

  

    x=layers.Conv2D(F3,kernel_size=(1,1),strides=(1,1),padding='valid',

                  name=conv_name_base+'2c',

                  kernel_initializer='he_normal')(x)

    x=layers.BatchNormalization(name=bn_name_base+'2c')(x)

  

    y=layers.Conv2D(F3,kernel_size=(1,1),strides=(s,s),padding='valid',

                  name=conv_name_base+'1',

                  kernel_initializer='he_normal')(layer_input)

    y=layers.BatchNormalization(name=bn_name_base+'1')(y)

  

    x=layers.add([x,y])

    x=layers.Activation('relu')(x)

  

    return x
def identity_block(layer_input,f_size,filters,stage,block):

    conv_name_base='res'+str(stage)+block+'_branch'

    bn_name_base='bn'+str(stage)+block+'_branch'

    F1,F2,F3=filters

    x=layers.Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding='valid',

                  name=conv_name_base+'2a',

                  kernel_initializer='he_normal'

                  )(layer_input)

    x=layers.BatchNormalization(name=bn_name_base+'2a')(x)

    x=layers.Activation('relu')(x)

  

    x=layers.Conv2D(filters=F2,kernel_size=(f_size,f_size),strides=(1,1),padding='same',

                  name=conv_name_base+'2b',

                  kernel_initializer='he_normal')(x)

    x=layers.BatchNormalization(name=bn_name_base+'2b')(x)

    x=layers.Activation('relu')(x)

  

    x=layers.Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding='valid',

                  name=conv_name_base+'2c',

                  kernel_initializer='he_normal')(x)

    x=layers.BatchNormalization(name=bn_name_base+'2c')(x)

  

    x=layers.add([x,layer_input])

    x=layers.Activation('relu')(x)

  

    return x
def resnet50(input_shape,num_classes=100):

    inputs=layers.Input(shape=input_shape)

    x=layers.ZeroPadding2D((3,3))(inputs)

  

    x=layers.Conv2D(64,kernel_size=(7,7),strides=(2,2),padding='valid',name='conv1',kernel_initializer='he_normal')(x)

    x=layers.BatchNormalization(name='bn_conv1')(x)

    x=layers.Activation('relu')(x)

    x=layers.MaxPooling2D((3,3),strides=(2,2))(x)

  

    x=conv_block(x,f_size=3,filters=[64,64,256],stage=2,block='a',s=1)

    x=identity_block(x,3,filters=[64,64,256],stage=2,block='b')

    x=identity_block(x,3,filters=[64,64,256],stage=2,block='c')

  

    x=conv_block(x,f_size=3,filters=[128,128,512],stage=3,block='a',s=2)

    x=identity_block(x,3,filters=[128,128,512],stage=3,block='b')

    x=identity_block(x,3,filters=[128,128,512],stage=3,block='c')

    x=identity_block(x,3,filters=[128,128,512],stage=3,block='d')

  

    x=conv_block(x,f_size=3,filters=[256,256,1024],stage=4,block='a',s=2)

    x=identity_block(x,3,filters=[256,256,1024],stage=4,block='b')

    x=identity_block(x,3,filters=[256,256,1024],stage=4,block='c')

    x=identity_block(x,3,filters=[256,256,1024],stage=4,block='d')

    x=identity_block(x,3,filters=[256,256,1024],stage=4,block='e')

    x=identity_block(x,3,filters=[256,256,1024],stage=4,block='f')

  

    x=conv_block(x,f_size=3,filters=[512,512,2048],stage=5,block='a',s=2)

    x=identity_block(x,3,filters=[512,512,2048],stage=5,block='b')

    x=identity_block(x,3,filters=[512,512,2048],stage=5,block='c')

  

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    outputs= layers.Dense(num_classes, activation='softmax', name='fc100')(x)



#   x=layers.AveragePooling2D(name='avg_pool')(x)

#   x=layers.Flatten()(x)

#   outputs=layers.Dense(num_classes,

#                  activation='softmax',

#                  name='fc'+str(num_classes),

#                  kernel_initializer='he_normal')(x)

  

    model=Model(inputs=inputs,outputs=outputs)

  

    return model
# a=resnet50(input_shape)

# a.summary()
def lr_schedule(epoch):

    lr=1e-3

    if epoch>180:

        lr*=0.5e-3

    elif epoch>160:

        lr*=1e-3

    elif epoch>120:

        lr*=1e-2

    elif epoch>80:

        lr*=1e-1

    print('LearningRate:',lr)

    return lr
model=resnet50(input_shape=input_shape)

model.compile(loss='categorical_crossentropy',

              optimizer=Adam(lr=lr_schedule(0)),

              metrics=['accuracy'])



lr_scheduler=LearningRateScheduler(lr_schedule)

lr_reducer=ReduceLROnPlateau(factor=np.sqrt(0.1),

                             cooldown=0,

                             patience=5,

                             min_lr=0.5e-6)

callbacks=[lr_scheduler,lr_reducer]

a=model.fit(x_train,y_train,

            batch_size=128,

            epochs=200,

            validation_data=(x_test,y_test),

            shuffle=True,

            callbacks=callbacks)

score=model.evaluate(x_test,y_test,verbose=1)

print('Test loss:',score[0])

print('Test Accuracy:',score[1])