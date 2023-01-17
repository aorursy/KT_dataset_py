#最开始需要import的包

from keras.models import Model

from keras.layers import Dense,Dropout,GlobalAveragePooling2D,BatchNormalization,Activation

from keras.applications import VGG16

#filename = ['i2','i4','i5','io','ip','p5','p11','p23','p26','pl5','pl30','pl40','pl50','pl60','pl80','pn','pne','po','w57']

#模型输出对应的类别
#check your path

import os

print(os.listdir('../input/task2-3'))
#模型只需调用一次即可

def create_model():

    benchmark=VGG16(weights='imagenet',input_shape=(32,32,3),include_top=False)

    x=benchmark.output

    #print(x.shape)

    x=GlobalAveragePooling2D()(x)

    #x=Dropout(0.1)(x)    #0.5太高了

    x=Dense(512)(x)

    x=BatchNormalization()(x)         #没有BN可能出现loss不变（2.69）的bug

    x=Activation('relu')(x)

    #x = Dropout(0.5)(x)

    x=Dense(128)(x)

    x=BatchNormalization()(x)         #减少节点数，增加深度

    x=Activation('relu')(x)

    #x=Dense(64)(x)

    #x=BatchNormalization()(x)         #减少节点数，增加深度

    #x=Activation('relu')(x)

    x = Dense(19)(x)

    x=Activation('softmax')(x)

    model=Model(inputs=benchmark.input,outputs=x)

    return model

model_vgg16=get_model_VGG16(h,w)

model_vgg16.load_weights('../input/task2-3/model-VGG16.h5')
#输出为filename中各类别的score

def pred(img):

    return model_vgg16.predict(img)