import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import glob

import random

from matplotlib.patches import Rectangle

from lxml import etree

%matplotlib inline
import os

os.listdir('../input/images/images')
os.listdir('../input/label/label')
image_path = glob.glob('../input/images/images/*/*.jpg')

len(image_path)
image_path[:3]
xmls_path = glob.glob('../input/label/label/*.xml')

len(xmls_path)
xmls_path[:3]
#xml_name extraction

xmls_train = [p.split('/')[-1].split('.')[0] for p in xmls_path]

xmls_train[:3]
#img_name extraction

imgs_train = [img for img in image_path if (img.split('/')[-1].split)('.jpg')[0] in xmls_train]

imgs_train[:3]
len(imgs_train),len(xmls_path)
#check the image to label sorts

xmls_path.sort(key=lambda x:x.split('/')[-1].split('.xml')[0])

imgs_train.sort(key=lambda x:x.split('/')[-1].split('.jpg')[0])

xmls_path[:3],imgs_train[:3]
#labels names

names = [x.split("/")[-2] for x in imgs_train]

names[:3]
names = pd.DataFrame(names,columns=['Types'])

names
#onehot for mutiple classes

from sklearn.preprocessing import LabelBinarizer



Class = names['Types'].unique()

Class_dict = dict(zip(Class, range(1,len(Class)+1)))

names['str'] = names['Types'].apply(lambda x: Class_dict[x])

lb = LabelBinarizer()

lb.fit(list(Class_dict.values()))

transformed_labels = lb.transform(names['str'])

y_bin_labels = []  



for i in range(transformed_labels.shape[1]):

    y_bin_labels.append('str' + str(i))

    names['str' + str(i)] = transformed_labels[:, i]
Class_dict
names.drop('str',axis=1,inplace=True)

names.drop('Types',axis=1,inplace=True)

names.head()
#analysis rectangular box value in xmls

def to_labels(path):

    xml = open('{}'.format(path)).read()                         #read xml in path 

    sel = etree.HTML(xml)                     

    width = int(sel.xpath('//size/width/text()')[0])     #extract the width/height

    height = int(sel.xpath('//size/height/text()')[0])    #extract the x,y value

    xmin = int(sel.xpath('//bndbox/xmin/text()')[0])

    xmax = int(sel.xpath('//bndbox/xmax/text()')[0])

    ymin = int(sel.xpath('//bndbox/ymin/text()')[0])

    ymax = int(sel.xpath('//bndbox/ymax/text()')[0])

    return [xmin/width, ymin/height, xmax/width, ymax/height]   #return the four relative points 
#set value to labels

labels = [to_labels(path) for path in xmls_path]

labels[:3]
#set four labels as outputs

out1,out2,out3,out4 = list(zip(*labels))        

#convert to np.array

out1 = np.array(out1)

out2 = np.array(out2)

out3 = np.array(out3)

out4 = np.array(out4)

label = np.array(names.values)
#label to tf.data

label_datasets = tf.data.Dataset.from_tensor_slices((out1,out2,out3,out4,label))

label_datasets
#def load_image function

def load_image(path):

    image = tf.io.read_file(path)                           

    image = tf.image.decode_jpeg(image,3)               

    image = tf.image.resize(image,[224,224])               

    image = tf.cast(image/127.5-1,tf.float32)                 

    return image      
#build dataset

dataset = tf.data.Dataset.from_tensor_slices(imgs_train)

dataset = dataset.map(load_image)
dataset_label = tf.data.Dataset.zip((dataset,label_datasets))
#batch constant

BATCH_SIZE = 16

AUTO = tf.data.experimental.AUTOTUNE
#batch extraction and shuffle

dataset_label = dataset_label.repeat().shuffle(500).batch(BATCH_SIZE)

dataset_label = dataset_label.prefetch(AUTO)
#Split dataset

test_count = int(len(imgs_train)*0.2)

train_count = len(imgs_train) - test_count

test_count,train_count
train_dataset = dataset_label.skip(test_count)

test_dataset = dataset_label.take(test_count)
train_dataset
species_dict = {v:k for k,v in Class_dict.items()}
#check from train_data

for img, label in train_dataset.take(1):

    plt.imshow(keras.preprocessing.image.array_to_img(img[0]))     

    out1,out2,out3,out4,out5 = label                            

    xmin,ymin,xmax,ymax = out1[0].numpy()*224,out2[0].numpy()*224,out3[0].numpy()*224,out4[0].numpy()*224

    rect = Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),fill=False,color='r')  

    ax = plt.gca()                      

    ax.axes.add_patch(rect)   

    pred_imglist = []

    pred_imglist.append(species_dict[np.argmax(out5[0])+1])

    plt.title(pred_imglist)

    plt.show()
#Convolution based

conv = keras.applications.xception.Xception(weights='imagenet',

                                            include_top=False,

                                            input_shape=(224,224,3),

                                            pooling='avg')
#open trainable

conv.trainable = True
#define Conv + FC structure

inputs = keras.Input(shape=(224,224,3))

x = conv(inputs)

x1 = keras.layers.Dense(1024,activation='relu')(x)

x1 = keras.layers.Dense(512,activation='relu')(x1)





out1 = keras.layers.Dense(1,name='out1')(x1)

out2 = keras.layers.Dense(1,name='out2')(x1)

out3 = keras.layers.Dense(1,name='out3')(x1)

out4 = keras.layers.Dense(1,name='out4')(x1)



x2 = keras.layers.Dense(1024,activation='relu')(x)

x2 = keras.layers.Dropout(0.5)(x2)

x2 = keras.layers.Dense(512,activation='relu')(x2)

out_class = keras.layers.Dense(10,activation='softmax',name='out_item')(x2)



out = [out1,out2,out3,out4,out_class]



model = keras.models.Model(inputs=inputs,outputs=out)

model.summary()
#model compille

model.compile(keras.optimizers.Adam(0.0003),

              loss={'out1':'mse',

                    'out2':'mse',

                    'out3':'mse',

                    'out4':'mse',

                    'out_item':'categorical_crossentropy'},

              metrics=['mae','acc'])
#learning_rate reduce module

lr_reduce = keras.callbacks.ReduceLROnPlateau('val_loss', patience=6, factor=0.5, min_lr=1e-6)
history = model.fit(train_dataset,

                   steps_per_epoch=train_count//BATCH_SIZE,

                   epochs=200,

                   validation_data=test_dataset,

                   validation_steps=test_count//BATCH_SIZE)
#training visualization

def plot_history(history):                

    hist = pd.DataFrame(history.history)           

    hist['epoch']=history.epoch

    

    plt.figure()                                     

    plt.xlabel('Epoch')

    plt.ylabel('MSE')               

    plt.plot(hist['epoch'],hist['loss'],

            label='Train Loss')

    plt.plot(hist['epoch'],hist['val_loss'],

            label='Val Loss')                           

    plt.legend()

    

    plt.figure()                                      

    plt.xlabel('Epoch')

    plt.ylabel('Val_MAE')               

    plt.plot(hist['epoch'],hist['val_out1_mae'],

            label='Out1_mae')

    plt.plot(hist['epoch'],hist['val_out2_mae'],

            label='Out2_mae')

    plt.plot(hist['epoch'],hist['val_out3_mae'],

            label='Out3_mae')

    plt.plot(hist['epoch'],hist['val_out4_mae'],

            label='Out4_mae')

    plt.legend()      

    

    plt.figure()                                      

    plt.xlabel('Epoch')

    plt.ylabel('Val_Item_Acc')               

    plt.plot(hist['epoch'],hist['val_out_item_acc'],

            label='Out5_acc')

    

    plt.show()

    

plot_history(history)      
mae = model.evaluate(test_dataset)
print('out1_mae in test:{}'.format(mae[6]))

print('out2_mae in test:{}'.format(mae[8]))

print('out3_mae in test:{}'.format(mae[10]))

print('out4_mae in test:{}'.format(mae[12]))

print('class_label in test:{}'.format(mae[15]))
model.save("class_location.h5")
species_dict = {v:k for k,v in Class_dict.items()}
species_dict
plt.figure(figsize=(10,24))

for img,_ in train_dataset.take(1):

    out1,out2,out3,out4,label = model.predict(img)

    for i in range(3):

        plt.subplot(3,1,i+1)            

        plt.imshow(keras.preprocessing.image.array_to_img(img[i]))    

        pred_imglist = []

        pred_imglist.append(species_dict[np.argmax(out5[i])+1])

        plt.title(pred_imglist)

        xmin,ymin,xmax,ymax = out1[i]*224,out2[i]*224,out3[i]*224,out4[i]*224

        rect = Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),fill=False,color='r') 

        ax = plt.gca()                   

        ax.axes.add_patch(rect)        