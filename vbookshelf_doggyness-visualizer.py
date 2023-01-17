import pandas as pd

import numpy as np

import os



import cv2



from keras.preprocessing.image import load_img, img_to_array

from keras.applications.vgg16 import preprocess_input, decode_predictions

import numpy as np

from keras import backend as K



import matplotlib.pyplot as plt

%matplotlib inline
os.listdir('../input')
from keras.applications.vgg16 import VGG16



model = VGG16(weights='imagenet')



model.summary()
path = '../input/images/Images/n02106662-German_shepherd/n02106662_22394.jpg'

image = plt.imread(path)



plt.imshow(image)

plt.show()
from keras.preprocessing.image import load_img, img_to_array

from keras.applications.vgg16 import preprocess_input, decode_predictions

from keras import backend as K



# set the path to the image

img_path = '../input/images/Images/n02106662-German_shepherd/n02106662_22394.jpg'



# Load the image and resize to 224x224

img = load_img(img_path, target_size=(224, 224))

x = img_to_array(img)



# Expand dims so the image has shape: (1, 224, 224, 3)

x = np.expand_dims(x, axis=0)



# Pre-process the image using the same pre-processing that was applied

# to the original imagenet images that were used to train VGG16.

x = preprocess_input(x)



# Make a prediction

preds = model.predict(x)



# Decode the prediction vector into a human readable format.

decode_predictions(preds, top=3)[0]
# VGG16 predicts on 1000 classes.

# Here we want the index of the class that has the highest prediction probability i.e. German_sheperd

imagenet_class = np.argmax(preds[0])



imagenet_class
# Get the German_sheperd entry from the prediction vector.

dog_output = model.output[:, imagenet_class]



# Get the output feature map from the last conv layer.

last_conv_layer = model.get_layer('block5_conv3')



# Gradient of the “German_sheperd” class with regard to the output feature map of block5_conv3.

grads = K.gradients(dog_output, last_conv_layer.output)[0] 



pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input],

                     [pooled_grads, last_conv_layer.output[0]])



# Values of these two quantities, as Numpy arrays, given the sample image.

pooled_grads_value, conv_layer_output_value = iterate([x])



for i in range(512):

    # Multiplies each channel in the feature-map array by 

    # “how important this channel is" with regard to the “German_sheperd” class.

    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]



# The channel-wise mean of the resulting feature map is the heatmap of the class activation.

heatmap = np.mean(conv_layer_output_value, axis=-1)



# For visulaization we normalize the heatmap between 0 and 1.

heatmap = np.maximum(heatmap, 0)

heatmap /= np.max(heatmap)



plt.matshow(heatmap)
# Use cv2 to read the image.

img = cv2.imread(img_path)



# Resize the heatmap to be the same size as the original image.

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))



# Convert the heatmap to RGB.

heatmap = np.uint8(255 * heatmap)



# Apply the heatmap to the original image.

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)



# 0.4 is the heatmap intensity factor.

superimposed_img = heatmap * 0.4 + img



# Save the image.

cv2.imwrite('dog_heatmap.jpg', superimposed_img)



# Display the superimposed image

image = plt.imread('dog_heatmap.jpg')

plt.imshow(image)



plt.show()