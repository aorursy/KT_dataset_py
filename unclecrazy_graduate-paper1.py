import os

import random as rn

import numpy as np

from sklearn import metrics

import matplotlib.pyplot as plt

%matplotlib inline

import tensorflow as tf

from keras import backend as K

from keras.applications.vgg16 import VGG16

from keras.applications.inception_v3 import InceptionV3

from keras.models import Model,Sequential

from keras.layers import GlobalAveragePooling2D, Convolution2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

from keras.preprocessing import image

from keras.optimizers import SGD
# 添加一些相关的随机参数的设置，保证模型的复现

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(1)

rn.seed(1)

# 强制tensorflow使用单线程，多线程是结果不可浮现的一个潜在来源

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

tf.set_random_seed(1)

sess = tf.Session(graph=tf.get_default_graph(),config=session_conf)

K.set_session(sess)
import keras as Ka

print("version: ",Ka.__version__)
from tensorflow.python.client import device_lib

# 列出所有的本地机器设备

local_device_protos = device_lib.list_local_devices()

# 只打印GPU设备

[print(x) for x in local_device_protos if x.device_type == 'GPU']
def sigmoid(x):

    return (1.0/(1.+np.exp(-x)))

def dsigmoid(x):

    tmp = np.exp(-x)

    return (tmp/((tmp+1)*(tmp+1)))



x = np.linspace(-10,10,500)

y = sigmoid(x)

y1 = dsigmoid(x)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)

plt.grid()

plt.xlabel("x")

plt.ylabel("sigmoid(x)")

plt.plot(x,y)

plt.subplot(1,2,2)

plt.grid()

plt.xlabel("x")

plt.ylabel("(sigmoid(x))'")

plt.plot(x,y1)
data_path = "/kaggle/input/cell-image1/cell_images1/cell_images1/"

# print(os.listdir(data_path))

train_path = data_path + "train/"

validation_path = data_path + "test/"

target_names = ["Parasitized","Uninfected"]
Parasitized = data_path + "train/Parasitized/"

Uninfected = data_path + "train/Uninfected/"

Parasitized_img_name = os.listdir(Parasitized)[:5]

Uninfected_img_name = os.listdir(Uninfected)[:5]



Parasitized_img_list = []

for img_name in Parasitized_img_name:

    img_path = Parasitized + img_name

    img = image.load_img(path=img_path, target_size= (50,50))

    Parasitized_img_list.append(img)



Uninfected_img_list = []

for img_name in Uninfected_img_name:

    img_path = Uninfected + img_name

    img = image.load_img(path=img_path, target_size= (50,50))

    Uninfected_img_list.append(img)

    

plt.figure(figsize=(20,8))



for i in range(5):

    plt.subplot(2,5,i+1)

    plt.tight_layout()

    plt.imshow(Parasitized_img_list[i])

    plt.axis('off')

for i in range(5):

    plt.subplot(2,5,6+i)

    plt.tight_layout()

    plt.imshow(Uninfected_img_list[i])

    plt.axis('off')
datagen = image.ImageDataGenerator(rotation_range=20,

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



img = image.load_img(Parasitized + Parasitized_img_name[0])

x = image.img_to_array(img)

x = x.reshape((1,) + x.shape)

i = 0

plt.figure(figsize=(12,8))

for batch in datagen.flow(x,batch_size=1,save_to_dir='/kaggle/input',save_prefix='cell',save_format='jpeg'):

    i += 1

    a = batch.reshape(batch.shape[1:])

    plt.subplot(2,3,i)

    plt.axis('off')

    plt.imshow(a)

    if(i >= 6):

        break
# 学习曲线函数

def plot_learning_curve(history):

    acc = history["acc"]

    val_acc = history["val_acc"]

    loss = history["loss"]

    val_loss = history["val_loss"]

    x = [i for i in range(0,len(acc))]

    plt.figure(figsize=(11,5))

    plt.subplot(1,2,2)

    plt.plot(x,acc,'-*')

    plt.plot(x,val_acc,'--')

    plt.grid()

    plt.ylim(0,1)

    plt.legend(["acc", "val_acc"])

    plt.xlabel("Epoch")

    plt.xticks(np.linspace(start=0,stop=len(x),num=5))

    plt.ylabel("Acc")

    

    plt.subplot(1,2,1)

    plt.plot(x,loss,'-*')

    plt.plot(x,val_loss,'--')

    plt.grid()

#     plt.ylim(0,1)

    plt.legend(["loss","val_loss"])

    plt.xlabel("Epoch")

    plt.xticks(np.linspace(start=0,stop=len(x),num=5))

    plt.ylabel("Loss")



# 混淆矩阵函数

def plot_cmatrix(y,y_pred):

    confmat = metrics.confusion_matrix(y_true=y, y_pred=y_pred)

    fig, ax = plt.subplots(figsize=(4,4))

    ax.matshow(confmat,cmap=plt.cm.BuGn_r,alpha=0.5)



    for i in range(confmat.shape[0]):

        for j in range(confmat.shape[1]):

            ax.text(x=j, y=i, s=confmat[i, j], va="center", ha="center")



    plt.xlabel("predicted label")

    plt.ylabel("true label")

    plt.xticks(range(2),target_names)

    plt.yticks(range(2),target_names)

    

def show_result(model,validation_datagen,input_shape):

    plot_learning_curve(model.history.history)

    test_generator = validation_datagen.flow_from_directory(shuffle=False,

        directory = validation_path,

        target_size=input_shape[:-1],

        batch_size=16,

        class_mode='categorical')

    test_generator.batch_index = 0

    y_pred = model.predict_generator(test_generator,steps=len(test_generator))

    y_pred = y_pred.argmax(axis=1)

    y = test_generator.labels

    print("acc: ",100*(1-np.abs(y-y_pred).mean()))

    print(metrics.classification_report(y_true=y,y_pred=y_pred,target_names=target_names))

    plot_cmatrix(y=y,y_pred=y_pred)
# 权重路径

weights_path = "../input/keras-pretrained-models/"

os.listdir(weights_path)
model = Sequential()

model.add(Convolution2D(filters=32,kernel_size=(3,3),input_shape=(150, 150, 3)))#padding=valid

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Convolution2D(filters=64,kernel_size=(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Convolution2D(filters=96,kernel_size=(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(2))

model.add(Activation('softmax'))
# categorical_crossentropy 多类对数损失

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
np.random.seed(1)

rn.seed(1)

# 强制tensorflow使用单线程，多线程是结果不可浮现的一个潜在来源

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

tf.set_random_seed(1)

sess = tf.Session(graph=tf.get_default_graph(),config=session_conf)

K.set_session(sess)



my_input_shape = (150,150,3)

# 设置输入图片augmentation方法，这里对于validation data只设置rescale,即图片矩阵除以255，对train data 进行翻转、放大、剪切

train_datagen = image.ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



test_datagen = image.ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        directory = train_path,  

        target_size= my_input_shape[:-1],  # all images will be resized to 150x150

        batch_size=16,

        shuffle= True, seed= 1,

        class_mode='categorical') #  "categorical"会返回2D的one-hot编码标签,



validation_generator = test_datagen.flow_from_directory(

        directory = validation_path,

        target_size= my_input_shape[:-1],

        batch_size=16,

        shuffle= True,seed = 1,

        class_mode='categorical')
model.fit_generator(train_generator,steps_per_epoch=200,epochs=20,

                    validation_steps=len(validation_generator),validation_data=validation_generator)
show_result(model=model,validation_datagen=test_datagen,input_shape=my_input_shape)
model.history.history
def Inception_V3(fine_tune):

    inceptionV3_weight = weights_path + "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

    base_model = InceptionV3(weights=inceptionV3_weight,include_top=False,pooling='avg')

    # base_model.load_weights(by_name=True,filepath=inceptionV3_weight)

    x = base_model.output

    x = Dense(256, activation='relu')(x)

    x = Dense(64,activation='relu')(x)

    predictions = Dense(2,activation='softmax')(x)

    model = Model(inputs= base_model.input,outputs=predictions)

    if not fine_tune:

        for layer in base_model.layers:

            layer.trainable = False

    return model
np.random.seed(1)

rn.seed(1)

# 强制tensorflow使用单线程，多线程是结果不可浮现的一个潜在来源

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

tf.set_random_seed(1)

sess = tf.Session(graph=tf.get_default_graph(),config=session_conf)

K.set_session(sess)



inceptionV3_input_shape = (299,299,3)

train_datagen = image.ImageDataGenerator(rescale=1./255,

                    shear_range=0.2,

                    zoom_range=0.2,

                    horizontal_flip=True)

validation_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(

                directory=train_path,

                target_size=inceptionV3_input_shape[:-1],

                color_mode= 'rgb',class_mode='categorical',

                classes= target_names, 

                batch_size= 16,

                shuffle= True, seed= 1)

validation_generator = validation_datagen.flow_from_directory(

                directory=validation_path,target_size=inceptionV3_input_shape[:-1],

                color_mode= 'rgb', class_mode='categorical',

                classes= target_names,            

                batch_size= 16,

                shuffle= True, seed= 1)
model_base_IV3 = Inception_V3(fine_tune=False)
model_base_IV3.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
model_base_IV3.fit_generator(generator=train_generator,steps_per_epoch=200,epochs=20,

                             validation_steps=len(validation_generator),validation_data=validation_generator)
show_result(model_base_IV3,validation_datagen=validation_datagen,input_shape=inceptionV3_input_shape)
model_base_IV3.history.history
np.random.seed(1)

rn.seed(1)

# 强制tensorflow使用单线程，多线程是结果不可浮现的一个潜在来源

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

tf.set_random_seed(1)

sess = tf.Session(graph=tf.get_default_graph(),config=session_conf)

K.set_session(sess)



inceptionV3_input_shape = (299,299,3)

train_datagen = image.ImageDataGenerator(rescale=1./255,

                    shear_range=0.2,

                    zoom_range=0.2,

                    horizontal_flip=True)

validation_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(

                directory=train_path,

                target_size=inceptionV3_input_shape[:-1],

                color_mode= 'rgb',class_mode='categorical',

                classes= target_names, 

                batch_size= 16,

                shuffle= True, seed= 1)

validation_generator = validation_datagen.flow_from_directory(

                directory=validation_path,target_size=inceptionV3_input_shape[:-1],

                color_mode= 'rgb', class_mode='categorical',

                classes= target_names,            

                batch_size= 16,

                shuffle= True, seed= 1)
model_base_IV3_FT = Inception_V3(fine_tune=True)
model_base_IV3_FT.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
model_base_IV3_FT.fit_generator(generator=train_generator,steps_per_epoch=200,epochs=20,

                           validation_steps=len(validation_generator),validation_data=validation_generator)
show_result(model_base_IV3_FT,validation_datagen=validation_datagen,input_shape=inceptionV3_input_shape)
model_base_IV3_FT.history.history
x = [i for i in range(20)]

y1 = model_base_IV3.history.history['val_acc']

y2 = model_base_IV3_FT.history.history['val_acc']

plt.plot(x,y1,'*-')

plt.plot(x,y2,'--')

plt.grid()

plt.legend(["InceptionV3-transfer","InceptionV3-transfer-ft"])

plt.xlabel("Epoch")

plt.ylabel("Acc")
VGG16_input_shape = (224, 224, 3)

def VGG16_model(fine_tune):

    weights = weights_path + "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

    base_model = VGG16(weights=weights, include_top=False,input_shape=VGG16_input_shape)

#     base_model.load_weights(weights,by_name=True)

    x = base_model.get_layer('block5_pool').output

    x = Flatten()(x)

    x = Dense(256,activation='relu')(x)

    x = Dense(64,activation='relu')(x)

    predictions = Dense(2,activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    if not fine_tune:

        print(fine_tune)

        for layer in base_model.layers[:15]:

            layer.trainable = False

    return model
np.random.seed(1)

rn.seed(1)

# 强制tensorflow使用单线程，多线程是结果不可浮现的一个潜在来源

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

tf.set_random_seed(1)

sess = tf.Session(graph=tf.get_default_graph(),config=session_conf)

K.set_session(sess)



train_datagen = image.ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)

validation_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(shuffle=True,seed=1,

        directory = train_path,  

        target_size=VGG16_input_shape[:-1],  # all images will be resized to 150x150

        batch_size=16,

        class_mode='categorical') #  "categorical"会返回2D的one-hot编码标签,

validation_generator = validation_datagen.flow_from_directory(shuffle=True,seed=1,

        directory = validation_path,

        target_size=VGG16_input_shape[:-1],

        batch_size=16,

        class_mode='categorical')
model_base_VGG16 = VGG16_model(fine_tune=False)
model_base_VGG16.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model_base_VGG16.fit_generator(generator=train_generator,steps_per_epoch=200,epochs=20,

                               validation_data=validation_generator,validation_steps=len(validation_generator))
show_result(model=model_base_VGG16,validation_datagen=validation_datagen,input_shape=VGG16_input_shape)
model_base_VGG16.history.history
np.random.seed(1)

rn.seed(1)

# 强制tensorflow使用单线程，多线程是结果不可浮现的一个潜在来源

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(1)

sess = tf.Session(graph=tf.get_default_graph(),config=session_conf)

K.set_session(sess)



train_datagen = image.ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)

validation_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(shuffle=True,seed=1,

        directory = train_path,  

        target_size=VGG16_input_shape[:-1],  # all images will be resized to 150x150

        batch_size=16,

        class_mode='categorical') #  "categorical"会返回2D的one-hot编码标签,

validation_generator = validation_datagen.flow_from_directory(shuffle=True,seed=1,

        directory = validation_path,

        target_size=VGG16_input_shape[:-1],

        batch_size=16,

        class_mode='categorical')
model_base_VGG16_FT = VGG16_model(fine_tune=True)
model_base_VGG16_FT.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model_base_VGG16_FT.fit_generator(generator=train_generator,steps_per_epoch=200,epochs=20,

                               validation_data=validation_generator,validation_steps=len(validation_generator))
show_result(model=model_base_VGG16_FT,validation_datagen=validation_datagen,input_shape=VGG16_input_shape)
model_base_VGG16_FT.history.history
x = [i for i in range(20)]

y1 = model_base_VGG16.history.history['val_acc']

y2 = model_base_VGG16_FT.history.history['val_acc']

plt.plot(x,y1,'*-')

plt.plot(x,y2,'--')

plt.grid()

plt.legend(["VGG16-transfer","VGG16-transfer-ft"])

plt.xlabel("Epoch")

plt.ylabel("Acc")