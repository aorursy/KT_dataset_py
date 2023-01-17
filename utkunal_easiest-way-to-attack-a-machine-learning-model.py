from __future__ import absolute_import, division, print_function, unicode_literals



# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras



# Helper libraries

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.cbook import flatten



fashion_mnist = keras.datasets.fashion_mnist



(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



#Loading the dataset returns four NumPy arrays:

#The train_images and train_labels arrays are the training set—the data the model uses to learn.

#The model is tested against the test set, the test_images, and test_labels arrays.



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#The following shows there are 60,000 images in the training set, with each image represented as 28 x 28 pixels:

train_images.shape

len(train_labels)

train_labels

#There are 10,000 images in the test set. Again, each image is represented as 28 x 28 pixels:

test_images.shape

len(test_labels)



#The data must be preprocessed before training the network. If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:



train_images = train_images / 255.0



test_images = test_images / 255.0



model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])

    

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



def plot_image(i, predictions_array, true_label, img):

  predictions_array, true_label, img = predictions_array, true_label[i], img[i]

  plt.grid(False)

  plt.xticks([])

  plt.yticks([])



  plt.imshow(img, cmap=plt.cm.binary)



  predicted_label = np.argmax(predictions_array)

  if predicted_label == true_label:

    color = 'blue'

  else:

    color = 'red'



  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],

                                100*np.max(predictions_array),

                                class_names[true_label]),

                                color=color)



def plot_value_array(i, predictions_array, true_label):

  predictions_array, true_label = predictions_array, true_label[i]

  plt.grid(False)

  plt.xticks(range(10))

  plt.yticks([])

  thisplot = plt.bar(range(10), predictions_array, color="#777777")

  plt.ylim([0, 1])

  predicted_label = np.argmax(predictions_array)



  thisplot[predicted_label].set_color('red')

  thisplot[true_label].set_color('blue')
a_img=[]

for imgnum in range(0,10000):

    a_img.append(list(flatten(train_images[imgnum])))



for img in range(0,len(a_img)):

    for pix in range(0,len(a_img[9999])):

        if a_img[img][pix]>0.7:

           a_img[img][pix]=0

           

    a_img[img]=np.array(a_img[img]).reshape(-1, 28)



a_labels=[]

a_labels.extend(train_labels[0:10000])



train_images1=np.concatenate((train_images, a_img))

train_labels1=np.concatenate((train_labels, a_labels))



#deformed test images



b_img=[]

for imgnum in range(0,1):

    b_img.append(list(flatten(test_images[imgnum])))



for img in range(0,len(b_img)):

    for pix in range(0,len(b_img[0])):

        if b_img[img][pix]>0.7:

           b_img[img][pix]=0

           

    b_img[img]=np.array(b_img[img]).reshape(-1, 28)



b_labels=[]

b_labels.extend(test_labels[0:1])



test_images1=np.concatenate((b_img, test_images))

test_labels1=np.concatenate((b_labels, test_labels))
model.fit(train_images1, train_labels1, epochs=10)



predictions = model.predict(test_images1)



num_rows = 5

num_cols = 3

num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):

  plt.subplot(num_rows, 2*num_cols, 2*i+1)

  plot_image(i, predictions[i], test_labels1, test_images1)

  plt.subplot(num_rows, 2*num_cols, 2*i+2)

  plot_value_array(i, predictions[i], test_labels1)

plt.tight_layout()

plt.show()



test_loss, test_acc = model.evaluate(test_images1,  test_labels1, verbose=2)



print('\nTest accuracy:', test_acc)