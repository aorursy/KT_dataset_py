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



import matplotlib.pyplot as plt

import cv2



import keras

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers.core import Dense, Dropout, Activation, Flatten

from sklearn.model_selection import train_test_split

from PIL import Image

import glob

from keras.preprocessing.image import load_img, img_to_array

from keras.initializers import TruncatedNormal, Constant

from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization

from keras.optimizers import SGD
# まずはデータを読み込む。画像のリストが用意されている。

df_train=pd.read_csv("../input/train.csv",skiprows=[1,]) 

df_test=pd.read_csv("../input/test.csv",skiprows=[1,])



print(df_train)

print(df_test)
# データの内訳を見てみる

# 軽度のヤツよりちょっとひどくなったヤツの方が多いらしい

# 自覚症状が出て気づくんだろうな

CLASSES={0:"No DR",1:"Mild",2:"Moderate",3:"Severe",4:"Proliferative DR"}



df_train['diagnosis'].value_counts().plot(kind='bar');

plt.title('Samples Per Class');
# データを1つ見てみる

imgPath=f"../input/train_images/cd54d022e37d.png"

img=cv2.imread(imgPath)

print(img.shape)

plt.imshow(img)

plt.show()
# なんてこった！目玉が蒼井、いや、青い。

# どうやらRGB記述になっているらしい

print(img.shape)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

plt.imshow(img)

plt.show()
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(img.shape)

plt.imshow(img,cmap='gray')

plt.show()

print(img)
# 画像の上下左右端を得る

def getEyeArea(image):

    print(image.shape)

    left = 0

    right = image.shape[1] - 1

    top = 0

    bottom = image.shape[0] - 1

    # 左端を求める

    for i in range(0, image.shape[1], 1):

        for j in range(0, image.shape[0], 1):

            if image[j][i] > 10:

                left = i

                break

        else:

             continue

        break

    

    #print(left)

    

    # 右端を求める

    for i in range(image.shape[1]-1, 0, -1):

        for j in range(0, image.shape[0], 1):

            if image[j][i] > 10:

                right = i

                break

        else:

             continue

        break

    

    #print(right)

    

    # 上端を求める

    for j in range(0, image.shape[0], 1):

        for i in range(0, image.shape[1], 1):

            if image[j][i] > 10:

                top = j

                break

        else:

             continue

        break

    

    #print(top)

        

    # 下端を求める

    for j in range(image.shape[0]-1, 0, -1):

        for i in range(0, image.shape[1], 1):

            if image[j][i] > 10:

                bottom = j

                break

        else:

             continue

        break

    

    #print(bottom)

    

    return left, right, top, bottom

    

left, right, top, bottom = getEyeArea(img)

img_cut = img[top:bottom,left:right]

print(img_cut.shape)

plt.imshow(img_cut,cmap='gray')

plt.show()
resize_num_test = 150

img_resize = cv2.resize(img_cut, dsize=(resize_num_test, resize_num_test))

print(img_resize.shape)

plt.imshow(img_resize,cmap='gray')

plt.show()
from keras.preprocessing.image import load_img, img_to_array, array_to_img

from keras.utils.np_utils import to_categorical

import csv



image_list = []

label_list = []



resize_num = 224



take = 0



#print(df_train.values)

for i in range(0, len(df_train), 1):

    #print(df_train.values[i,0])

    imgPath="../input/train_images/" + str(df_train.values[i,0]) + ".png" 

    print(imgPath)

    img = cv2.imread(str(imgPath))

    #plt.imshow(img)

    #plt.show()

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    left, right, top, bottom = getEyeArea(img)

    img_cut = img[top:bottom,left:right]

    img_resize = cv2.resize(img_cut, dsize=(resize_num, resize_num))

    img_reshape = np.reshape(img_resize, (resize_num, resize_num, 1))

    image_list.append(img_reshape)

    label_list.append(str(df_train.values[i,1]))

    take = take + 1

    print(take)

    

images = np.array(image_list)

labels = to_categorical(label_list)



# imageの画素値をint型からfloat型にする

images = images.astype('float32')

# 画素値を[0～255]⇒[0～1]とする

images = images / 255.0



print(labels)
print(labels)
# GrayScaleのときに1、COLORのときに3にする

COLOR_CHANNEL = 1



# 入力画像サイズ(画像サイズは正方形とする)

INPUT_IMAGE_SIZE = 224



# 訓練時のバッチサイズとエポック数

BATCH_SIZE = 32

EPOCH_NUM = 100



# CLASS数を取得する

CLASS_NUM = len(CLASSES)

print("クラス数 : " + str(CLASS_NUM))





def conv2d(filters, kernel_size, strides=1, bias_init=1, **kwargs):

    trunc = TruncatedNormal(mean=0.0, stddev=0.01)

    cnst = Constant(value=bias_init)

    return Conv2D(

        filters,

        kernel_size,

        strides=strides,

        padding='same',

        activation='relu',

        kernel_initializer=trunc,

        bias_initializer=cnst,

        **kwargs

    )



def dense(units, **kwargs):

    trunc = TruncatedNormal(mean=0.0, stddev=0.01)

    cnst = Constant(value=1)

    return Dense(

        units,

        activation='tanh',

        kernel_initializer=trunc,

        bias_initializer=cnst,

        **kwargs

    )



def AlexNet():

    model = Sequential()



    # 第1畳み込み層

    model.add(conv2d(96, 11, strides=(4,4), bias_init=0, input_shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, COLOR_CHANNEL)))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(BatchNormalization())



    # 第2畳み込み層

    model.add(conv2d(256, 5, bias_init=1))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(BatchNormalization())



    # 第3~5畳み込み層

    model.add(conv2d(384, 3, bias_init=0))

    model.add(conv2d(384, 3, bias_init=1))

    model.add(conv2d(256, 3, bias_init=1))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(BatchNormalization())



    # 全結合層

    model.add(Flatten())

    model.add(dense(4096))

    model.add(Dropout(0.5))

    model.add(dense(4096))

    model.add(Dropout(0.5))



    # 出力層

    model.add(Dense(CLASS_NUM, activation='softmax'))



    model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    return model



# コンパイル

model = AlexNet()

model.fit(images, labels, nb_epoch=100, batch_size=100, validation_split=0.1)
image_list_test = []

label_list_test = []

results = []



path = "../submission.csv"



results.append("id_code,diagnosis")



take = 0



#print(df_train.values)

for i in range(0, len(df_test), 1):

    #print(df_train.values[i,0])

    print(i)

    imgPath="../input/test_images/" + str(df_test.values[i,0]) + ".png" 

    print(imgPath)

    img = cv2.imread(str(imgPath))

    #plt.imshow(img)

    #plt.show()

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    left, right, top, bottom = getEyeArea(img)

    img_cut = img[top:bottom,left:right]

    img_resize = cv2.resize(img_cut, dsize=(resize_num, resize_num))

    img_reshape = np.reshape(img_resize, (resize_num, resize_num, 1))

    target = np.array(img_reshape, dtype=np.float32)

    target = target / 255

    target = target[None, ...]

    #print(target_array.shape)

    predict = model.predict(target, batch_size=1, verbose=0)

    score = np.max(predict)

    pred_label = np.argmax(predict)

    image_list_test.append(str(df_test.values[i,0]))

    label_list_test.append(str(pred_label))

    result = str(df_test.values[i,0]) + "," + str(pred_label)

    results.append(result)



print(results)

    

StackingSubmission = pd.DataFrame({ 'id_code': image_list_test,'diagnosis': label_list_test })

StackingSubmission.to_csv("../submission.csv", index=False)
os.remove("../input/submission.csv")
# 結果のCSVを確認したい。

df_result=pd.read_csv("../submission.csv",skiprows=[1,])



print(df_result)