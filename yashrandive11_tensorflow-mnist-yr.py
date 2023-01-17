# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install tensorflow_datasets
import tensorflow as tf

import numpy as np



import tensorflow_datasets as tfds
mnist_dataset, mnist_info = tfds.load(name = 'mnist',  with_info = True, as_supervised = True)
mnist_train, mnist_test = mnist_dataset['train'],mnist_dataset['test']



# Fetching number of 10% of Train data for Validation dataset as TensorFlow Datrasets do not provide a readymade validation Sample 

num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples

num_validation_samples = tf.cast(num_validation_samples, tf.int64)



# Assigning new variable for test dataset

num_test_samples = mnist_info.splits['test'].num_examples

num_test_samples= tf.cast(num_test_samples, tf.int64)



#Function Scaling the training and Validation data

def scale(image, label):

    image = tf.cast(image, tf.float32)

    image /= 255. #This is where the image value is scaled

    

    return image,label



scaled_train_and_validation_data = mnist_train.map(scale)



test_data = mnist_test.map(scale)



#Shuffling the Train and Validation Dataset



BUFFER_SIZE = 10000

shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)



validation_data = shuffled_train_and_validation_data.take(num_validation_samples)



train_data = shuffled_train_and_validation_data.skip(num_validation_samples)



#Preprocessing data for Batching and Batching



BATCH_SIZE = 100



train_data = train_data.batch(BATCH_SIZE)

validation_data = validation_data.batch(num_validation_samples)

test_data = test_data.batch(num_test_samples)



#TO give Validation data same SHAPE and PROPERTIES as train data and test data

validation_inputs, validation_targets = next(iter(validation_data))
input_size = 784

output_size = 10

hidden_layer_size =  200



model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape = (28,28,1)),

    tf.keras.layers.Dense(hidden_layer_size, activation = 'relu' ),

    tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),

    tf.keras.layers.Dense(output_size, activation = 'softmax')

    

])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
NUM_EPOCHS = 10



model.fit(train_data, epochs = NUM_EPOCHS, validation_data = (validation_inputs, validation_targets),validation_steps = 1, verbose = 2)
test_loss, test_accuracy = model.evaluate(test_data)
print( 'Test Accuracy is ', repr(test_accuracy *100), '% and Test Loss = ', repr(test_loss))