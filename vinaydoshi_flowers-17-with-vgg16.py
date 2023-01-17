# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator, img_to_array

from keras.optimizers import SGD, RMSprop

from keras.applications import VGG16

from keras.layers import Input

from keras.models import Model





import numpy as np

#import argparse

import os

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Part of imutils library. Cannot insert custom library for GPU, so adding code for function

import os



image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")





def list_images(basePath, contains=None):

    # return the set of files that are valid

    return list_files(basePath, validExts=image_types, contains=contains)





def list_files(basePath, validExts=None, contains=None):

    # loop over the directory structure

    for (rootDir, dirNames, filenames) in os.walk(basePath):

        # loop over the filenames in the current directory

        for filename in filenames:

            # if the contains string is not none and the filename does not contain

            # the supplied string, then ignore the file

            if contains is not None and filename.find(contains) == -1:

                continue



            # determine the file extension of the current file

            ext = filename[filename.rfind("."):].lower()



            # check to see if the file is an image and should be processed

            if validExts is None or ext.endswith(validExts):

                # construct the path to the image and yield it

                imagePath = os.path.join(rootDir, filename)

                yield imagePath



def resize(image, width=None, height=None, inter=cv2.INTER_AREA):

    # initialize the dimensions of the image to be resized and

    # grab the image size

    dim = None

    (h, w) = image.shape[:2]



    # if both the width and height are None, then return the

    # original image

    if width is None and height is None:

        return image



    # check to see if the width is None

    if width is None:

        # calculate the ratio of the height and construct the

        # dimensions

        r = height / float(h)

        dim = (int(w * r), height)



    # otherwise, the height is None

    else:

        # calculate the ratio of the width and construct the

        # dimensions

        r = width / float(w)

        dim = (width, int(h * r))



    # resize the image

    resized = cv2.resize(image, dim, interpolation=inter)



    # return the resized image

    return resized
class SimpleDatasetLoader:

    def __init__(self, preprocessors = None):

        # store the image preprocessor

        self.preprocessors = preprocessors

        

        # if the preprocessors are None, initialize them as an empty list

        if self.preprocessors is None:

            self.preprocessors = []

    

    def load(self, imagePaths, verbose=-1):

        # initialize the list of features and labels 

        data = []

        labels = []

        # loop over the input images 

        for i, imagePath in enumerate(imagePaths):

            # load the image and extract the class label assuming 

            # that our path has the following format:

            # /path/to/dataset/{class}/{image}.jpg

            image = cv2.imread(imagePath)

            label = imagePath.split(os.path.sep)[-2]

            

            if self.preprocessors is not None:

                # loop over the preprocessors and apply each to the image 

                for p in self.preprocessors:

                    image = p.preprocess(image)

            data.append(image)

            labels.append(label)

            

            # show an update every ‘verbose‘ images

            if verbose > 0 and i > 0 and (i+1) % verbose == 0:

                print(f'[INFO] processed {((i+1)/(len(imagePaths)))*100}%')

        # return a tuple of the data and labels                

        return (np.array(data), np.array(labels))
class imageToArrayPreprocessor:

    def __init__(self, dataFormat=None):

        self.dataFormat = dataFormat

    

    def preprocess(self, image):

        return img_to_array(image, data_format = self.dataFormat)
class AspectAwarePreprocessor:

    

    def __init__(self, width, height, inter = cv2.INTER_AREA):

        self.width = width 

        self.height= height

        self.inter = inter 

        

    def preprocess(self, image):

        (h,w) = image.shape[:2]

        dH, dW = 0, 0

        

        if w < h:

            image = resize(image, width = self.width, inter = self.inter)

            dH = (image.shape[0] - self.height)//2

            

        else:

            image = resize(image, height = self.height, inter = self.inter)

            dW = (image.shape[1] - self.width)//2      

            

        (h,w) = image.shape[:2]

        #print('new',image.shape)

        image = image[dH:h-dH, dW:w-dW]

        

        return cv2.resize(image, (self.width, self.height), interpolation = self.inter)
from keras.layers.core import Dropout, Flatten, Dense



class FC_Replacement_Layer:

    def build(baseModel, classes, D):

        headModel = baseModel.output

        headModel = Flatten(name='flatten')(headModel)

        headModel = Dense(D, activation='relu')(headModel)

        headModel = Dropout(0.5)(headModel)

        headModel = Dense(classes, activation='softmax')(headModel)

        return headModel
imagePaths = list(list_images('../input/flowers17/17flowers/17flowers/jpg/'))

print(f'No. of images = {len(imagePaths)}')

classNames = [i.split(os.path.sep)[-2] for i in imagePaths]

print(f'Classes = {len(np.unique(classNames))}')

classNames = [str(x) for x in np.unique(classNames)]
aap = AspectAwarePreprocessor(224,224)

iap = imageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[aap,iap])

(data, labels) = sdl.load(imagePaths, verbose=500)

data = data.astype('float')/255.0
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.25, random_state=42, stratify=labels)
X_train.shape, X_test.shape,len(classNames)
lb = LabelBinarizer()

y_train = lb.fit_transform(y_train)

y_test = lb.transform(y_test)
baseModel = VGG16(weights= 'imagenet', include_top=False, input_tensor = Input(shape=(224,224,3)))

headModel = FC_Replacement_Layer.build(baseModel, len(classNames), 256)

model = Model(inputs = baseModel.input, outputs=headModel)
for layer in baseModel.layers:

    layer.trainable = False
optimizer = RMSprop(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])

model.summary()
aug = ImageDataGenerator(rotation_range=60, width_shift_range=0.3, height_shift_range=0.3,

                         shear_range=0.4, zoom_range=0.3, horizontal_flip=True, vertical_flip=True, fill_mode='nearest'

                        )
model.fit_generator(aug.flow(X_train, y_train,batch_size=32), validation_data=(X_test,y_test), epochs=25,

                    steps_per_epoch=len(X_train)//32, verbose=1)
preds = model.predict(X_test, batch_size=32)

print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names= classNames))
for layer in baseModel.layers[7:]:

    layer.trainable = True
for i,l in enumerate(baseModel.layers):

    print(i,l.__class__.__name__)
opt = SGD(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])

model.summary()
model.fit_generator(aug.flow(X_train, y_train,batch_size=32), validation_data=(X_test,y_test), epochs=100,

                    steps_per_epoch=len(X_train)//32, verbose=1)
preds = model.predict(X_test, batch_size=32)

print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names= classNames))
a=3