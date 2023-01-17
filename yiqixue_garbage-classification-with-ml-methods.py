# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import random
import pathlib
import glob
import math

all_images = pathlib.Path("/kaggle/input/garbage-classification/Garbage classification/Garbage classification/")
all_images = list(all_images.glob('*/*.jpg'))
all_images = [str(path) for path in all_images]                  
random.shuffle(all_images) #shuffle the images

N=len(all_images)
N_category=[393,491,400,584,472,127]

frac=4/5   #fraction of data for training
N1 = math.floor(N*frac)
N2 = N-N1
train_images = all_images[:N1]
test_images = all_images[N1+1:]
data_dir=pathlib.Path("/kaggle/input/garbage-classification/Garbage classification/Garbage classification/")
label_names = sorted(item.name for item in data_dir.glob('*/') if item.is_dir()) #sorted label names
label_to_index = dict((name, index) for index,name in enumerate(label_names))  #assign 0~5 to the labels

print(label_to_index)
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_images]
train_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in train_images]
test_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in test_images]
#PIL-Python Imaging Library
import PIL.Image  #resize images
def GreyPicture(image,width):
    img = PIL.Image.open(image).convert('LA') #convert to greyscale
    basewidth = width
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    return img.resize((basewidth,hsize), PIL.Image.ANTIALIAS) 
    
def RGBPicture(image,width):
    img = PIL.Image.open(image)
    basewidth = width
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    return img.resize((basewidth,hsize), PIL.Image.ANTIALIAS) 
    

#Sample
img_1=RGBPicture(train_images[0],150)
display(img_1)
print(img_1.size,img_1.mode) 

img_2=GreyPicture(train_images[1],150)
display(img_2)
print(img_2.size,img_2.mode)

#images to single matrix
Grey_train_matrix=[]
for img in train_images:
    matrix= np.asarray(GreyPicture(img,width=150))
    image_arr = np.ndarray.flatten(matrix)
    Grey_train_matrix.append(image_arr)
Grey_test_matrix=[]
for img in test_images:
    matrix= np.asarray(GreyPicture(img,width=150))
    image_arr = np.ndarray.flatten(matrix)
    Grey_test_matrix.append(image_arr)
RGB_train_matrix=[]
for img in train_images:
    matrix= np.asarray(RGBPicture(img,width=150))
    image_arr = np.ndarray.flatten(matrix)
    RGB_train_matrix.append(image_arr)
RGB_test_matrix=[]
for img in test_images:
    matrix= np.asarray(RGBPicture(img,width=150))
    image_arr = np.ndarray.flatten(matrix)
    RGB_test_matrix.append(image_arr)
#Naive Bayes Model(generative)

#sklearn MultinomialNB
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

#train accuracy
clf.fit(np.asarray(RGB_train_matrix), train_image_labels)
res_rgb=clf.predict(RGB_train_matrix)
clf.fit(np.asarray(Grey_train_matrix), train_image_labels)
res_grey=clf.predict(Grey_train_matrix)
size=len(train_image_labels)
count_rgb = 0
count_grey = 0
for i in range(size):
    if res_rgb[i] == train_image_labels[i]:
        count_rgb += 1
    if res_grey[i] == train_image_labels[i]:
        count_grey += 1
print("RGB train accuracy is {}".format(count_rgb/size)) 
print("Grey train accuracy is {}".format(count_grey/size)) 
#test accuracy
clf.fit(np.asarray(RGB_train_matrix), train_image_labels)
res_rgb=clf.predict(RGB_test_matrix)
clf.fit(np.asarray(Grey_train_matrix), train_image_labels)
res_grey=clf.predict(Grey_test_matrix)
size=len(test_image_labels)
count_rgb = 0
count_grey = 0
for i in range(size):
    if res_rgb[i] == test_image_labels[i]:
        count_rgb += 1
    if res_grey[i] == test_image_labels[i]:
        count_grey += 1
print("RGB test accuracy is {}".format(count_rgb/size)) 
print("Grey test accuracy is {}".format(count_grey/size)) 
#Logistic Regression(discrimitive)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=100,multi_class='multinomial').fit(RGB_train_matrix, train_image_labels)
res_rgb=clf.predict(RGB_test_matrix)
acc_1=clf.score(RGB_train_matrix,train_image_labels)
acc_2=clf.score(RGB_test_matrix,test_image_labels)
clf = LogisticRegression(max_iter=100,multi_class='multinomial').fit(Grey_train_matrix, train_image_labels)
res_grey=clf.predict(Grey_test_matrix)

#accuracy
acc_3=clf.score(Grey_train_matrix,train_image_labels)
acc_4=clf.score(Grey_test_matrix,test_image_labels)
print("RGB train accuracy is {}".format(acc_1)) 
print("RGB test accuracy is {}".format(acc_2)) 
print("Grey train accuracy is {}".format(acc_3)) 
print("Grey test accuracy is {}".format(acc_4)) 
#Multilayer Perceptron Neural Network
import tensorflow as tf
#standardization
RGB_train_matrix = np.asarray(RGB_train_matrix)/255
#encoder
train_labels = np.zeros((N1,6));
for i in range(N1):
    train_labels[i,train_image_labels[i]] = 1;
multilayer_perceptron = tf.keras.Sequential([tf.keras.layers.Dense(64, input_shape = (50400,), activation = 'relu'),
                             tf.keras.layers.Dense(64,  activation = 'relu'),
                             tf.keras.layers.Dense(32,  activation = 'relu'),
                            tf.keras.layers.Dense(6, activation = 'softmax')])
multilayer_perceptron.summary()
#optm = tf.keras.optimizers.SGD(lr=0.001)
multilayer_perceptron.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
multilayer_perceptron.fit(RGB_train_matrix, train_labels, batch_size = 100, shuffle = True, epochs = 100)
#prediction for test set
pred = multilayer_perceptron.predict(np.asarray(RGB_test_matrix))
prediction = np.asarray(tf.argmax(pred, 1));
#accuracy
count_correct = 0;
for i in range(N2-1):
    if prediction[i] == test_image_labels[i]:
        count_correct += 1;
acc = count_correct/N2
print("Accuracy is {}".format(acc))
RGB_train_tensor=[]
for img in train_images:
    matrix= np.asarray(RGBPicture(img,width=150))
    RGB_train_tensor.append(matrix)
RGB_train_tensor = np.asarray(RGB_train_tensor)/255
RGB_test_tensor=[]
for img in test_images:
    matrix= np.asarray(RGBPicture(img,width=150))
    RGB_test_tensor.append(matrix)
RGB_test_tensor = np.asarray(RGB_test_tensor)/255
#CNN
CNN = tf.keras.Sequential()
CNN.add(tf.keras.layers.Conv2D(32,input_shape = ([112, 150, 3]),kernel_size = 5,activation='relu'))
CNN.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2))) 
CNN.add(tf.keras.layers.Flatten())
CNN.add(tf.keras.layers.Dense(32, activation='relu'))
CNN.add(tf.keras.layers.Dense(6, activation='softmax'))
CNN.summary()
CNN.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
CNN.fit(RGB_train_tensor, train_labels, batch_size = 100, shuffle = True, epochs = 20, validation_split=0.1)
pred = CNN.predict(RGB_test_tensor)
prediction = np.asarray(tf.argmax(pred, 1));
#accuracy
count_correct = 0;
for i in range(N2-1):
    if prediction[i] == test_image_labels[i]:
        count_correct += 1;
acc = count_correct/N2
print("Accuracy is {}".format(acc))