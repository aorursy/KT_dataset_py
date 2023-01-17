# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf 

from tensorflow import keras 

import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#print(tf.__version__)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels) ,(test_images,test_labels) = fashion_mnist.load_data()

class_names = ['Tshirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'] 
len(class_names)
train_images.shape



len(train_labels)
train_labels
test_images.shape

test_labels

len(test_labels)
#preprocess data 

plt.figure()

plt.imshow(train_images[0])

plt.colorbar()

plt.grid(False)

plt.show()

plt.figure(figsize= (10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i],cmap=plt.cm.binary)

    plt.xlabel(class_names[train_labels[i]])

plt.show
model = keras.Sequential([

        keras.layers.Flatten(input_shape = (28,28)),

        keras.layers.Dense(128,activation = 'relu'),

        keras.layers.Dense(10)

    

])
model.compile(optimizer = 'adam',

              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),

              metrics = ['accuracy'] 

              

             )
model.fit(train_images,train_labels,epochs = 10)
test_loss, test_acc = model.evaluate(test_images ,test_labels ,verbose = 2)

print(test_loss)

print(test_acc)
#make predictions 

probability_model = tf.keras.Sequential([model,

                                       tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

predictions[0]
np.argmax(predictions[0])
test_labels[0]
def plot_image(i,predictions_array,true_label,img_passed):

    predictions_array, true_label, img = predictions_array, true_label[i], img_passed[i]

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)

    

    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label :

        color = 'blue'

    else :

        color = 'red'

        

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],

                                100*np.max(predictions_array),

                                class_names[true_label]),

                                color=color)



    

def plot_value_array(i,predictions_array,true_label):

        prediction_array ,true_label = predictions_array ,true_label[i]

        plt.grid(False)

        plt.xticks(range(10))

        plt.yticks([])

        thisplot = plt.bar(range(10),predictions_array,color = '#777777')

        plt.ylim([0,1])

        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')

        thisplot[true_label].set_color('blue')

i = 0 

plt.figure(figsize = (6,3))

plt.subplot(1,2,1)

plot_image(i,predictions[i],test_labels,test_images)

plt.subplot(1,2,2)

plot_value_array(i,predictions[i],test_labels)

plt.show()
# Plot the first X test images, their predicted labels, and the true labels.

# Color correct predictions in blue and incorrect predictions in red.

num_rows = 5

num_cols = 3

num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):

  plt.subplot(num_rows, 2*num_cols, 2*i+1)

  plot_image(i, predictions[i], test_labels, test_images)

  plt.subplot(num_rows, 2*num_cols, 2*i+2)

  plot_value_array(i, predictions[i], test_labels)

plt.tight_layout()

plt.show()