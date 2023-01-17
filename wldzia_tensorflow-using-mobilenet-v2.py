# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob as gb

import os

import cv2
class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']

class_names_label = {class_name:i for i, class_name in enumerate(class_names)}



n_classes = 6
class_names[2]
print(class_names_label)
def load_data():

    datasets = ['seg_train/seg_train', 'seg_test/seg_test']

    size = (150, 150)

    output = [] 

    for dataset in datasets:

        directory = "/kaggle/input/intel-image-classification/" + dataset

        images = []

        labels = []

        for folder in os.listdir(directory):

            curr_label = class_names_label[folder]

            for file in os.listdir(directory + "/" + folder):

                img_path = directory + "/" + folder + "/" + file

                curr_img = cv2.imread(img_path)

                curr_img = cv2.resize(curr_img, size)

                images.append(curr_img)

                labels.append(curr_label)

        images = np.array(images, dtype='float32')

        labels = np.array(labels, dtype='int32')



        images = images / 255



        output.append((images, labels))



    return output
#%tensorflow_version 2.x  # this line is not required unless you are in a notebook

import tensorflow as tf



from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt

keras = tf.keras
(train_images, train_labels),(test_images, test_labels) = load_data()
train_images.shape
# Let's look at a one image

IMG_INDEX = 8710  # change this to look at other images



plt.imshow(train_images[IMG_INDEX] ,cmap=plt.cm.binary)

plt.xlabel(class_names[train_labels[IMG_INDEX]])

plt.show()
IMG_SIZE = 150
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)



# Create the base model from the pre-trained model MobileNet V2

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,

                                               include_top=False,

                                               weights='imagenet')
#base_model.summary()
base_model.trainable = False # freezing the trainable hyperparemetes of the existing model. pre-trained model MobileNet V2
#base_model.summary()
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(6)
model = tf.keras.Sequential([

  base_model,

  global_average_layer,

  prediction_layer

])
#model.summary()
base_learning_rate = 0.0001

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

            metrics=['accuracy'])



history = model.fit(train_images, train_labels, epochs=10, 

                    validation_data=(test_images, test_labels))
def plot_accuracy_loss(history):

    """

        Plot the accuracy and the loss during the training of the nn.

    """

    fig = plt.figure(figsize=(10,5))



    # Plot accuracy

    plt.subplot(221)

    plt.plot(history.history['accuracy'],'bo--', label = "acc")

    plt.plot(history.history['val_accuracy'], 'ro--', label = "val_acc")

    plt.title("train_acc vs val_acc")

    plt.ylabel("accuracy")

    plt.xlabel("epochs")

    plt.legend()



    # Plot loss function

    plt.subplot(222)

    plt.plot(history.history['loss'],'bo--', label = "loss")

    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")

    plt.title("train_loss vs val_loss")

    plt.ylabel("loss")

    plt.xlabel("epochs")



    plt.legend()

    plt.show()
plot_accuracy_loss(history)
test_loss = model.evaluate(test_images, test_labels)
predpath = '../input/intel-image-classification/seg_pred/'
pred = []

files = gb.glob(pathname= str(predpath + 'seg_pred/*.jpg'))

for file in files: 

    image = cv2.imread(file)

    image_array = cv2.resize(image,(IMG_SIZE,IMG_SIZE))

    pred.append(list(image_array))
images = np.array(pred, dtype='float32')

images = images / 255
len((images))
plt.figure(figsize=(20,20))

for n , i in enumerate(list(np.random.randint(0,len(images),36))) : 

    plt.subplot(6,6,n+1)

    plt.imshow(images[i])    

    plt.axis('off')

preds = model.predict(images) # contains the final probiblity distribution of the output classes for each of the image. 7301 images and 6 classes distribution.
len(preds)
plt.figure(figsize=(20,20))

for n , i in enumerate(list(np.random.randint(0,len(images),36))) : 

    plt.subplot(6,6,n+1)

    plt.imshow(images[i])    

    plt.axis('off')

    x =np.argmax(preds[i]) # takes the maximum of of the 6 probabilites. 

    plt.title((class_names[x]))