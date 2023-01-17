

import numpy as np 

import pandas as pd 



from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense



        

root_dir = '../input/horse2zebra1/horse2zebra'  



IMAGE_WIDTH = 256

IMAGE_HEIGHT = 256

INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)



BATCH_SIZE = 20

EPOCHS = 10



training_data_generator = ImageDataGenerator(rescale=1./255,

                                             shear_range=0.1,

                                             zoom_range=0.1,

                                             horizontal_flip=True,

                                             validation_split = 0.2)

test_data_generator = ImageDataGenerator(rescale=1./255)



training_generator = training_data_generator.flow_from_directory(

    root_dir,

    classes = ['trainA','trainB'],

    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training")



validation_generator = training_data_generator.flow_from_directory(

    root_dir,

    classes = ['trainA','trainB'],

    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation")



test_generator = test_data_generator.flow_from_directory(

    root_dir,

    classes = ['testA','testB'],

    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

    batch_size=BATCH_SIZE,

    class_mode="categorical")







model = Sequential([

    Conv2D(4, (5, 5), input_shape=INPUT_SHAPE, activation='relu'),

    MaxPooling2D(),

    Conv2D(4, (11, 11), activation='relu'),

    Conv2D(4, (5, 5), activation='relu'),

    MaxPooling2D(),

    Flatten(),

    Dense(64, activation='relu'),

    Dense(12, activation='relu'),

    Dense(2, activation='softmax')

])





model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])











model.fit_generator(

    training_generator,

    steps_per_epoch = training_generator.samples // BATCH_SIZE,

    validation_data = validation_generator, 

    validation_steps = validation_generator.samples // BATCH_SIZE,

    epochs=EPOCHS,

    verbose=1)
model.evaluate_generator(test_generator)
import matplotlib.image as mpimg





test_generator.reset()

probabilities = model.predict_generator(test_generator)

for index, probability in enumerate(probabilities[:10]):

    image_path = root_dir + "/" + test_generator.filenames[test_generator.index_array[index]]

    image = mpimg.imread(image_path)

    plt.imshow(image)

    if probability[0] > 0.5:

        plt.title("%.2f" % (probability[0]*100) + "% horse")

    else:

        plt.title("%.2f" % ((1-probability[0])*100) + "% zebra")

    plt.show()
test_generator.reset()

Y_pred = model.predict_generator(test_generator)

print(test_generator.index_array)

print(test_generator.classes)

classes = test_generator.classes[test_generator.index_array]

print(classes)

y_pred = np.argmax(Y_pred, axis=-1)

print(y_pred)

sum(y_pred==classes)/262
from sklearn.metrics import confusion_matrix

confusion_matrix(test_generator.classes[test_generator.index_array],y_pred)