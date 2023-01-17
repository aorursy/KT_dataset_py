from IPython.display import Image, display
display(Image('../input/testcatdog/test/test/dog.4001.jpg'))
display(Image('../input/testcatdog/test/test/cat.4001.jpg'))
display(Image('../input/cat-and-dog/test_set/test_set/cats/cat.4453.jpg'))
display(Image('../input/cat-and-dog/training_set/training_set/dogs/dog.423.jpg'))

#Check paths
import os.path
from os import walk
#for i in os.walk('../input/testcatdog/test/') :
#    print(i)
    
for i in os.walk('../input/catanddogdlweight/') :
    print(i)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
# identical to the previous one
classifier = load_model('../input/catanddogdlweight/cat_dog_dl_weight_2epoch.h5')

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('../input/cat-and-dog/training_set/training_set/',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('../input/cat-and-dog/test_set/test_set/',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 2000)

#save weights
classifier.save("cat_dog_dl_weight_7epoch.h5")
#prediction

# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image

#input a test image
test_img_path = '../input/testcatdog/test/test/dog.4001.jpg'

test_image = image.load_img(test_img_path, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
display(Image(test_img_path))
print("Prediction:",prediction)