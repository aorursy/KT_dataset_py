import numpy as np

import pandas as pd 

import keras

from keras.models import Model

from keras.layers import *

from keras.layers import Input, Dense,Dropout

import keras.layers as layers

from glob import glob

from keras import optimizers

import os

import keras

from keras.models import Model

from keras.layers import *

from keras.layers import Input, Dense,Dropout

import keras.layers as layers

from keras.models import Model

from keras.layers import Dropout

from keras.layers import merge, Convolution2D,ZeroPadding2D,UpSampling2D

from keras.layers import Conv2D,Concatenate,Flatten,Reshape,Add

import cv2

import re,random

import scipy
def conv2d_bn(x,filters,num_row,num_col,padding='same',strides=(1, 1), name=None):

    if name is not None:

        bn_name = name + '_bn'

        conv_name = name + '_conv'

    else:

        bn_name = None

        conv_name = None

    if keras.backend.image_data_format() == 'channels_first':

        bn_axis = 1

    else:

        bn_axis = 3

    x = keras.layers.Conv2D(

        filters, (num_row, num_col),

        strides=strides,

        padding=padding,

        use_bias=False,

        name=conv_name)(x)

    x = keras.layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    x = keras.layers.Activation('relu', name=name)(x)

    return x



def model_keras():

    image_shape = (161,577,3)

    if keras.backend.image_data_format() == 'channels_first':

        channel_axis = 1

    else:

        channel_axis = 3

    img_input = keras.layers.Input(shape=image_shape)

    #image_shape = (621,189)

    #process block with convs

    x = conv2d_bn(img_input, 32, 3, 3, strides=(1, 1), padding='valid')

    x = conv2d_bn(x, 32, 3, 3, padding='valid')

    x = conv2d_bn(x, 64, 3, 3)

    x = keras.layers.MaxPooling2D((3, 3), strides=(1, 1))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')

    x = conv2d_bn(x, 192, 3, 3, padding='valid')

    x = keras.layers.MaxPooling2D((3, 3), strides=(1, 1))(x)

    # 609x177x192

    #mixed0: 609x177x256

    #inception+resnet block1 

    branch1x1 = conv2d_bn(x, 64, 1, 1)



    branch5x5 = conv2d_bn(x, 48, 1, 1)

    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)



    branch3x3dbl = conv2d_bn(x, 64, 1, 1)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)



    branch_pool = keras.layers.AveragePooling2D((3, 3),

                                          strides=(1, 1),

                                          padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)

    x = keras.layers.concatenate(

        [branch1x1, branch5x5, branch3x3dbl, branch_pool],

        axis=channel_axis,

        name='mixed0')



    # mixed 1: 609x177x 288 BLOCK2

    branch1x1 = conv2d_bn(x, 64, 1, 1)



    branch5x5 = conv2d_bn(x, 48, 1, 1)

    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)



    branch3x3dbl = conv2d_bn(x, 64, 1, 1)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)



    branch_pool = keras.layers.AveragePooling2D((3, 3),

                                          strides=(1, 1),

                                          padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    x = keras.layers.concatenate(

        [branch1x1, branch5x5, branch3x3dbl, branch_pool],

        axis=channel_axis,

        name='mixed1')

    

    # mixed 2: 609x177x 288 BLOCK3

    branch1x1 = conv2d_bn(x, 64, 1, 1)



    branch5x5 = conv2d_bn(x, 48, 1, 1)

    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)



    branch3x3dbl = conv2d_bn(x, 64, 1, 1)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)



    branch_pool = keras.layers.AveragePooling2D((3, 3),

                                          strides=(1, 1),

                                          padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    x = keras.layers.concatenate(

        [branch1x1, branch5x5, branch3x3dbl, branch_pool],

        axis=channel_axis,

        name='mixed2')



    # mixed 3: 304 x 88 x 768 BLOCK4

    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')



    branch3x3dbl = conv2d_bn(x, 64, 1, 1)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch3x3dbl = conv2d_bn(

        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')



    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = layers.concatenate(

        [branch3x3, branch3x3dbl, branch_pool],

        axis=channel_axis,

        name='mixed3')



    # mixed 4: 304 x 88 x 768 BLOCK5

    branch1x1 = conv2d_bn(x, 192, 1, 1)



    branch7x7 = conv2d_bn(x, 128, 1, 1)

    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)

    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)



    branch7x7dbl = conv2d_bn(x, 128, 1, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)



    branch_pool = layers.AveragePooling2D((3, 3),

                                          strides=(1, 1),

                                          padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    x = layers.concatenate(

        [branch1x1, branch7x7, branch7x7dbl, branch_pool],

        axis=channel_axis,

        name='mixed4')

    

    # mixed 5, 6: 304 x 88 x 768 BLOCK6

    branch1x1 = conv2d_bn(x, 192, 1, 1)



    branch7x7 = conv2d_bn(x, 160, 1, 1)

    branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)

    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)



    branch7x7dbl = conv2d_bn(x, 160, 1, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)



    branch_pool = layers.AveragePooling2D(

        (3, 3), strides=(1, 1), padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    x = layers.concatenate(

        [branch1x1, branch7x7, branch7x7dbl, branch_pool],

        axis=channel_axis,

        name='mixed' + str(5 + 0))



    # mixed 7: 304 x 88 x 768 BLOCK 7

    branch1x1 = conv2d_bn(x, 192, 1, 1)



    branch7x7 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)

    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)



    branch7x7dbl = conv2d_bn(x, 192, 1, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)



    branch_pool = layers.AveragePooling2D((3, 3),

                                          strides=(1, 1),

                                          padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    x = layers.concatenate(

        [branch1x1, branch7x7, branch7x7dbl, branch_pool],

        axis=channel_axis,

        name='mixed7')

    # mixed 8: 151 x 43 x 1280

    branch3x3 = conv2d_bn(x, 192, 1, 1)

    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,

                          strides=(2, 2), padding='valid')



    branch7x7x3 = conv2d_bn(x, 192, 1, 1)

    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)

    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)

    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')



    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = layers.concatenate(

        [branch3x3, branch7x7x3, branch_pool],

        axis=channel_axis,

        name='mixed8')

    # mixed 9: 151 x 43 x 2048

    for i in range(2):

        branch1x1 = conv2d_bn(x, 320, 1, 1)



        branch3x3 = conv2d_bn(x, 384, 1, 1)

        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)

        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)

        branch3x3 = layers.concatenate(

            [branch3x3_1, branch3x3_2],

            axis=channel_axis,

            name='mixed9_' + str(i))



        branch3x3dbl = conv2d_bn(x, 448, 1, 1)

        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)

        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)

        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)

        branch3x3dbl = layers.concatenate(

            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)



        branch_pool = layers.AveragePooling2D(

            (3, 3), strides=(1, 1), padding='same')(x)

        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        x = layers.concatenate(

            [branch1x1, branch3x3, branch3x3dbl, branch_pool],

            axis=channel_axis,

            name='mixed' + str(9 + i))

    

    #     x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    if keras.backend.image_data_format() == 'channels_first':

        bn_axis = 1

    else:

        bn_axis = 3

    x=layers.Conv2DTranspose(1024,kernel_size=(4,4),strides=(2,2))(x)

    x = keras.layers.BatchNormalization(axis=bn_axis, scale=False, name='1')(x)

    x=layers.Conv2DTranspose(512,kernel_size=(4,4),strides=(2,2))(x)

    x = keras.layers.BatchNormalization(axis=bn_axis, scale=False, name='2')(x)

    x=layers.Conv2DTranspose(256,kernel_size=(8,8),strides=(1,1))(x)

#     x=layers.Conv2DTranspose(2048,kernel_size=(,128),strides=(1,1))(x)

#     x=layers.Conv2D(size=(2, 2))(x)

#     x = conv2d_bn(x, 128, 1, 1)(x)

    x=Dense(128, activation='relu')(x)

    x = keras.layers.BatchNormalization(axis=bn_axis, scale=False, name='3')(x)

    x=Dropout(0.5)(x)

    x=Dense(32, activation='relu')(x)

    x = keras.layers.BatchNormalization(axis=bn_axis, scale=False, name='4')(x)

    x=Dropout(0.5)(x)

    x=Dense(8, activation='relu')(x)

    x = keras.layers.BatchNormalization(axis=bn_axis, scale=False, name='5')(x)

    x=Dropout(0.5)(x)

    x=Dense(2, activation='softmax')(x)

    model = Model(img_input, x, name='SYnet')



    return model

model1= model_keras()

model1.summary()
# test_y=[]

# test_x=[]

# def get_batches_fn(data_folder,image_shape,batch_size=32):

#     X=[]

#     Y=[]

#     image_shape=(161,577,3)

#     image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))

#     label_paths = {

#         re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path

#         for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

#     background_color = np.array([255, 0, 0])

# #     random.shuffle(image_paths)

#     i=0

    

#     for image_file in image_paths:

#         i+=1

#         gt_image_file = label_paths[os.path.basename(image_file)]

#         image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

#         gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)



#         gt_bg = np.all(gt_image == background_color, axis=2)

#         gt_bg = gt_bg.reshape(*gt_bg.shape, 1)

#         gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

#         if i>200:

#             test_x.append(image)

#             test_y.append(gt_image)

#         else:

#             X.append(image)

#             Y.append(gt_image)

#     return X, Y



# data_dir="../input/roadlane-detection-evaluation-2013/data_road/data_road/training/"

# image_shape=(161,577,3)

# X, Y = get_batches_fn(data_dir,image_shape)

# X=np.array(X)

# Y=np.array(Y)

# model1 = model_keras()

# # model1.summary()

# opt_adam=optimizers.adam(clipnorm=1)

# model1.compile(loss='binary_crossentropy',

#               optimizer=opt_adam,

#               metrics=['accuracy'])

# print(X.shape)

# print(Y.shape)

# import requests

# headers={"Authorization":"Bearer ya29.Il-zBwYbzP18KM2kRuKEgghEn9y2mA1PZftcxPmjw9zlQbFfi-x-jvPIEdt8Xe9QORi9gVKYN0sY-E32EOHhZGVkj7RpvOnPlOS7-bPFm_JigsEAilXpa_XIkvpbQe7GZg"}

# r=requests.get('https://www.googleapis.com/drive/v3/files/1DNpGJpAor6qmXCy9C0yDIn2mbHyZeoqq?alt=media',headers=headers)

# open('model.h5', 'wb').write(r.content)
# model1.load_weights('./model.h5')
# ind=10

# xhat=np.expand_dims(test_x[ind],axis=0)

# y_pred=model1.predict(xhat,batch_size=1)
# y=y_pred[0]

# c = np.zeros((161,577,3))

# w=np.argmax(y,axis=-1)

# print('Pred')

# for i in range(len(w)):

#     for j in range(len(w[i])):

#         c[i][j][w[i][j]]=255
# import matplotlib.pyplot as plt

# import matplotlib.image as mpimg

# plt.imshow(c)
plt.imshow(test_x[ind])
# b=np.zeros((161,577,3))

# b[:,:,:-1]=test_y[ind]

# plt.imshow(b)


# def mean_iou(y_pred,y_true):

#     res=0

#     t=0

#     for i in range(2):

#         inter=0

#         union=0

#         for j in range(len(y_pred)):

#             for k in range(len(y_pred[0])):

#                 if y_pred[j][k][i]>0 and y_true[j][k][i]>0:

#                     inter+=1

#                 if y_pred[j][k][i]>0 or y_true[j][k][i]>0:

#                     union+=1

#         if union>0 and inter>0:

#             t+=1

#             res=res+(inter)/union

#     print(res/t)
# mean_iou(c,b)
# from moviepy.editor import VideoFileClip
# def run_inference_on_video(video_path, model1):

#     def run_inference_on_image(image):

#         image = scipy.misc.imresize(image, image_shape)

#         xhat=np.expand_dims(image,axis=0)

#         y_pred=model1.predict(xhat,batch_size=1)

#         y=y_pred[0]

#         c = np.zeros((161,577,3))

#         w=np.argmax(y,axis=-1)

        

#         for i in range(len(w)):

#             for j in range(len(w[i])):

#                 c[i][j][w[i][j]]=1

        

# #         c=np.expand_dims(c,axis=0)

#         im_softmax=c[:,:,1]

# #         im_softmax = im_softmax[:, 1].reshape(image_shape[0], image_shape[1])

#         #(im_softmax >= 0)

#         segmentation = (im_softmax >= 0.5).reshape(image_shape[0], image_shape[1], 1)

#         mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))

#         mask = scipy.misc.toimage(mask, mode="RGBA")

#         street_im = scipy.misc.toimage(image)

#         street_im.paste(mask, box=None, mask=mask)



#         return np.array(street_im)



#     print('Running inference on video...')

#     clip = VideoFileClip(video_path)

#     new_clip = clip.fl_image(run_inference_on_image)

#     print('Inference complete.')



#     # write to file

#     new_clip.write_videofile('output_video.mp4')

#     print('Video saved!')
run_inference_on_video('./driving.mp4',model1)
# !wget https://raw.githubusercontent.com/informramiz/Road-Semantic-Segmentation/master/data/driving.mp4
# !pip install moviepy
# c[:,:,1].shape
# bearer="ya29.Il-zB_99PLYc9QtuJxIAxH1ee7LO5i9gezNRBRTUMRasThHVtpiaqKmN6QPQYKufdq-SgFV_X3R9b_ebkEJN_KcdkHZsh8hKo7kdiqdHpWqoObmnYk7QJkQ5w8HMuWc_dA"

# import json

# import requests



# folder_id='1yLGW3ClBQtDWmnXK6M2itNsnLnfAAszk'

# headers = {"Authorization": "Bearer "+bearer}



# #Folder id is https://drive.google.com/drive/folders/<folder_id>

# # Bearer token has to be generated. Follow this link - https://help.talend.com/reader/Ovc10QFckCdvYbzxTECexA/EoAKa_oFqZFXH0aE0wNbHQ



# para = {

#     "name": "output_video.mp4",

#     "parents": [folder_id]

# }

# files = {

#     'data': ('metadata', json.dumps(para), 'application/json; charset=UTF-8'),

#     'file': open("./output_video.mp4", "rb")

# }

# r = requests.post(

#     "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",

#     headers=headers,

#     files=files

# )

# print(r.text)



!ls