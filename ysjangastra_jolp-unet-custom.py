import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import cv2

import os

%matplotlib inline



from scipy.stats import norm

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.callbacks import EarlyStopping

from keras.layers import *

from keras.models import *

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras import regularizers,optimizers

from keras.callbacks import LearningRateScheduler

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
base_path = '../input/davis480dataset/DAVIS480/'

input_output_shape =(480, 854, 3)
train_file_path = os.listdir('../input/davis480dataset/DAVIS480/ImageSets/480p')[0]

test_file_path  = os.listdir('../input/davis480dataset/DAVIS480/ImageSets/480p')[1]

val_file_path  = os.listdir('../input/davis480dataset/DAVIS480/ImageSets/480p')[2]
train_file_path = os.path.join(base_path,'ImageSets/480p',train_file_path)

train_base_path = os.path.join(base_path,'ImageSets/480p')

train_anno_path = os.path.join(base_path,'annotaion/480p')

                               

test_file_path = os.path.join(base_path,'ImageSets/480p',test_file_path)

test_base_path = os.path.join(base_path,'ImageSets/480p')

test_anno_path = os.path.join(base_path,'annotaion/480p')



val_file_path = os.path.join(base_path,'ImageSets/480p',val_file_path)

val_base_path = os.path.join(base_path,'ImageSets/480p')

val_anno_path = os.path.join(base_path,'annotaion/480p')
train_file = open(train_file_path,"r")

train_path_list = train_file.read().split('\n')
test_file = open(test_file_path,"r")

test_path_list = test_file.read().split('\n')
val_file = open(val_file_path,"r")

val_path_list = val_file.read().split('\n')
train_info_arr =[]

train_file_path_list = open(os.path.join(base_path,"ImageSets/480p/train.txt"), 'r')



train_file_raw_list = train_file_path_list.read().split('\n')
annotation_path = '../input/davis480dataset/DAVIS480/annotaion/480p'

annotation_lables_list = os.listdir(annotation_path)



img_path = '../input/davis480dataset/DAVIS480/jpegimg/480p'

img_lables_list = os.listdir(annotation_path)
val_label = {}

val_data = {}



train_label = {}

train_data = {}



test_label = {}

test_data = {}
for one in val_path_list:

    chk=0

    tmp_label = ''

    for two in one.split(' /'):

        #print(two)

        

        for three in two.split('/'):

           # print(three,tmp_label,"png" in three)

            if three == "480p":

                chk+=1

            elif chk==1:

                if three not in (val_label):

                    val_label[three] = []

                    val_data[three]=[]

                tmp_label=three

                chk+=1

            if "jpg" in three:

                val_data[tmp_label].append(os.path.join(img_path,tmp_label,three))

            if "png" in three:

                #print('sipal',tmp_label,three,val_label[tmp_label])

                val_label[tmp_label].append(os.path.join(annotation_path,tmp_label,three))
for one in train_path_list:

    chk=0

    tmp_label = ''

    for two in one.split(' /'):

        #print(two)

        

        for three in two.split('/'):

           # print(three,tmp_label,"png" in three)

            if three == "480p":

                chk+=1

            elif chk==1:

                if three not in (train_label):

                    train_label[three] = []

                    train_data[three]=[]

                tmp_label=three

                chk+=1

            if "jpg" in three:

                train_data[tmp_label].append(os.path.join(img_path,tmp_label,three).strip())

            if "png" in three:

                #print('sipal',tmp_label,three,val_label[tmp_label])

                train_label[tmp_label].append(os.path.join(annotation_path,tmp_label,three).strip())
train_base_path
for one in test_path_list:

    chk=0

    tmp_label = ''

    for two in one.split(' /'):

        #print(two)

        

        for three in two.split('/'):

           # print(three,tmp_label,"png" in three)

            if three == "480p":

                chk+=1

            elif chk==1:

                if three not in (test_label):

                    test_label[three] = []

                    test_data[three]=[]

                tmp_label=three

                chk+=1

            if "jpg" in three:

                test_data[tmp_label].append(os.path.join(img_path,tmp_label,three).strip())

            if "png" in three:

                #print('sipal',tmp_label,three,val_label[tmp_label])

                test_label[tmp_label].append(os.path.join(annotation_path,tmp_label,three).strip())
x_train= []

y_train  = []

x_val =[]

y_val = []

x_test = []

y_test = []
for key in train_data:

    for x_trp in train_data[key]:

        x_train.append(cv2.imread(x_trp))

x_train = np.asarray(x_train)



for key in train_label:

    for x_trp in train_label[key]:

        y_train.append(cv2.imread(x_trp))

y_train = np.asarray(y_train)
for key in val_data:

    for x_valp in val_data[key]:

        x_val.append(cv2.imread(x_valp))

x_val = np.asarray(x_val)



for key in val_label:

    for x_valp in val_label[key]:

        y_val.append(cv2.imread(x_valp.strip()))

y_val = np.asarray(y_val)
"""

for key in test_data:

    for x_tstp in test_data[key]:

        x_test.append(cv2.imread(x_tstp))

x_test = np.asarray(x_test)



for key in test_label:

    for x_tstp in test_label[key]:

        y_test.append(cv2.imread(x_tstp))

y_test = np.asarray(y_test)

"""
print("x_val",x_val.shape,",",y_val.shape)

print("train",x_train.shape,",",y_train.shape)

#print("test",x_test.shape,",",y_test.shape)
def CustomUnet():

   

    input_ = Input(shape=input_output_shape)

    conv1 =  Conv2D(64,(3,3),activation='relu')(input_)

    conv1 = Conv2D(64,(3,3),activation='relu')(conv1)

    crop_conv1 = Cropping2D(cropping=((106,106),(109,109)))(conv1)

    print("conv1 shape",conv1.shape,"crop_conv1",crop_conv1.shape)

    conv1_maxpool = MaxPooling2D(2,2)(conv1)

    conv2 = (Conv2D(128,(3,3),activation='relu')(conv1_maxpool))

    conv2 = (Conv2D(128,(3,3),activation='relu')(conv2))

    crop_conv2 = Cropping2D(cropping=((49,49),(50,51)))(conv2)

    print("conv2 shape",conv2.shape,"crop_conv2",crop_conv2.shape)

    conv2_maxpool = MaxPooling2D(2,2)(conv2)

    conv3 = (Conv2D(256,(3,3),activation='relu')(conv2_maxpool))

    conv3 =(Conv2D(256,(3,3),activation='relu')(conv3))

    crop_conv3 = Cropping2D(cropping=((20,21),(21,21)))(conv3)

    print("conv3 shape",conv3.shape,"crop_conv3",crop_conv3.shape)

    conv3_maxpool = MaxPooling2D(2,2)(conv3)

    conv4 = (Conv2D(512,(3,3),activation='relu')(conv3_maxpool))

    conv4 =(Conv2D(512,(3,3),activation='relu')(conv4))

    crop_conv4 = Cropping2D(cropping=((6,6),(6,7)))(conv4)

    conv4_maxpool = conv3_maxpool = MaxPooling2D(2,2)(conv4)

    print("conv4 shape",conv4.shape,"crop_conv4",crop_conv4.shape)

    conv5 = (Conv2D(1024,(3,3),activation='relu')(conv4_maxpool))

    conv5 =(Conv2D(1024,(3,3),activation='relu')(conv5))

    conv5 =(Conv2D(512,(3,3),activation='relu')(conv5))

    print("conv5 shape",conv5.shape)

    upconv5 = UpSampling2D()(conv5)

    upconv5 = Concatenate(axis=-1)([crop_conv4,upconv5])

    print("upconv5 shape",upconv5.shape)

    upconv5 = Conv2D(512,(3,3))(upconv5)

    upconv5 = Conv2D(256,(3,3))(upconv5)

    upconv6 = UpSampling2D()(upconv5)

    upconv6 = Concatenate(axis=-1)([crop_conv3,upconv6])

    print("upconv6 shape",upconv6.shape)

    upconv6 = Conv2D(256,(3,3))(upconv6)

    upconv6 = Conv2D(128,(3,3))(upconv6)

    upconv7 = UpSampling2D()(upconv6)

    upconv7 = Concatenate(axis=-1)([crop_conv2,upconv7])

    print("upconv7 shape",upconv7.shape)

    upconv7 = Conv2D(128,(3,3))(upconv7)

    upconv7 = Conv2D(64,(3,3))(upconv7)

    upconv8 = UpSampling2D()(upconv7)

    upconv8 = Concatenate(axis=-1)([crop_conv1,upconv8])

    print("upconv8 shape",upconv8.shape)

    upconv8 = Conv2D(64,(3,3))(upconv8)

    upconv8 = Conv2D(64,(3,3))(upconv8)

    upconv8 = Conv2D(3,(1,1))(upconv8)

    

    model = Model(input_,upconv8)

    return model
Dvs_Unet_base = CustomUnet()
Dvs_Unet_base.summary()
patience=2

callbacks1 = [

    #EarlyStopping(monitor='val_loss', patience=patient, mode='min', verbose=1),

    ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = patience / 2, min_lr=0.00001, verbose=1, mode='min'),

    #ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),

    ]
optimizer = optimizers.RMSprop(lr=0.00001)

Dvs_Unet_base.compile(optimizer=optimizer,loss="binary_crossentropy")

Dvs_Unet_base.fit(x_train,y_train,epochs=50,batch_size=5,validation_data=(x_val[:1000],y_val[:1000]),callbacks=callbacks1)
Dvs_Unet_base.save('./Dvs_Unet_base_prototype.h5')
res = Dvs_Unet_base.predict(x_val[999:1000])
res =res.astype(np.uint8)
res
plt.imshow(res[0])
plt.imshow(x_val[999])
plt.imshow(y_val[999])