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
import os

print(os.listdir("../input/chest-xray-pneumonia/chest_xray"))


from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers.core import Flatten, Dense, Dropout

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import SGD

from keras import optimizers

model= Sequential()





model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))

model.add(Dense(units = 1, activation = 'sigmoid'))

#optim=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                  horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                 horizontal_flip = True)



training_set = train_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/chest_xray/train',

                                                 target_size = (64, 64),

                                                 batch_size = 1000,

                                                 class_mode = 'binary')



test_set = test_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/chest_xray/test',

                                            target_size = (64, 64),

                                            batch_size = 600,

                                            class_mode = 'binary')



history=model.fit_generator(training_set,

                         steps_per_epoch = 40,

                         epochs = 3,

                         validation_data = test_set,

                         validation_steps = 10)

from keras.utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
import cv2



img = cv2.imread('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person577_virus_1121.jpeg')

img = cv2.resize(img,(64,64))

img = np.reshape(img,[1,64,64,3])



classes = model.predict_classes(img)



print(classes)
import cv2



img = cv2.imread('../input/chest-xray-pneumonia/chest_xray/test/NORMAL/NORMAL2-IM-0343-0001.jpeg')

img = cv2.resize(img,(64,64))

img = np.reshape(img,[1,64,64,3])



classes = model.predict_classes(img)



print(classes)
model.save_weights('model3.h5')
import sklearn

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

Y_pred = model.predict(test_set)

print(Y_pred)

p=np.round(Y_pred)

#print(p)

print(test_set.classes)

y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')

cm =sklearn.metrics.confusion_matrix(test_set.classes,p)

print(cm)

print('Classification Report')

print(sklearn.metrics.classification_report(test_set.classes,p))

print(p)
from mlxtend.plotting import plot_confusion_matrix

cm_plot_label = ['Normal','Pneumonia']

plot_confusion_matrix(cm,cm_plot_label)
score  = model.evaluate(test_set)

print(score[1])