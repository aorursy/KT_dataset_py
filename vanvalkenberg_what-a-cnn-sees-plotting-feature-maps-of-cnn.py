#################################################################################################################

# For this Notebook is took help from the following resources:                                                  #

# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/#

# https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16                                        #

# https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/preprocess_input                       #

# https://www.tensorflow.org/api_docs/python/tf/keras/Model                                                     #

#################################################################################################################





#################################################################################################################

# This notebooks written to help you visualize what a Cnn is learning at each Convulusion layer                 #

#################################################################################################################

## Importing All the usefull libraries

import tensorflow as tf2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras import Model
###################################################

# We will be using VGG 16 model                   #

###################################################



MyModel = tf2.keras.applications.VGG16(

    include_top=True, weights='imagenet', input_tensor=None, input_shape=None,

    pooling=None, classes=1000, classifier_activation='softmax'

)



MyModel.summary()
### lets Load our test images

import cv2

Cat = cv2.imread('/kaggle/input/dogs-cats-images/dog vs cat/dataset/test_set/cats/cat.4949.jpg')

Dog = cv2.imread('/kaggle/input/dogs-cats-images/dog vs cat/dataset/test_set/dogs/dog.4816.jpg')



Dog = cv2.resize(Dog,(224, 224))

Cat = cv2.resize(Cat,(224, 224))

plt.imshow(Cat)
plt.imshow(Dog)
### lets Define a Function that can show Features learned by CNN's nth convolusion layer

def ShowMeWhatYouLearnt(Image, layer, MyModel):

    img = img_to_array(Image)

    img = np.expand_dims(img, 0)

    ### preprocessing for img for vgg16

    img = tf2.keras.applications.vgg16.preprocess_input(img)

    

    ## Now lets define a model which will help us

    ## see what vgg16 sees 

    inputs = MyModel.inputs

    outputs = MyModel.layers[layer].output

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()

    

    ## let make predictions to see what the Cnn sees

    featureMaps = model.predict(img)

    

    ## Plotting Features

    for maps in featureMaps:

        plt.figure(figsize=(20,20))

        pltNum = 1

        

        for a in range(8):

            for b in range(8):

                plt.subplot(8, 8, pltNum)

                plt.imshow(maps[: ,: ,pltNum - 1], cmap='gray')

                pltNum += 1

        

        plt.show()

        

    

    

    

    

    
ShowMeWhatYouLearnt(Cat, 1, MyModel)
ShowMeWhatYouLearnt(Dog, 1, MyModel)
ShowMeWhatYouLearnt(Cat, 5, MyModel)
ShowMeWhatYouLearnt(Cat, 9, MyModel)
#############################################################################

#  Commentary:                                                              #

# A Cnn breaks images into it's component features, hence it's good idea    #

# to increase feature maps at every convolusion layer                       #

#############################################################################
ShowMeWhatYouLearnt(Dog, 5, MyModel)