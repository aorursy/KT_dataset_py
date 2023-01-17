import tensorflow
from tensorflow import keras

from tensorflow.keras.applications.vgg19 import VGG19
model=VGG19(weights='imagenet')

model.save('VGG19.h5')