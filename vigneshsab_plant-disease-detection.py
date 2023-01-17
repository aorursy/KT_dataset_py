# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pickle

import cv2

from os import listdir

from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation, Flatten, Dropout, Dense

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing import image

from keras.preprocessing.image import img_to_array, array_to_img

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
EPOCHS = 25

INIT_LR = 1e-3

BS = 32

default_image_size = tuple((256, 256))

image_size = 0

directory_root = '../input/plantdisease/'

width=256

height=256

depth=3
def convert_image_to_array(image_dir):

    try:

        image = cv2.imread(image_dir)

        if image is not None :

            image = cv2.resize(image, default_image_size)   

            return img_to_array(image)

        else :

            return np.array([])

    except Exception as e:

        print(f"Error : {e}")

        return None
image_list, label_list = [], []

try:

    print("[INFO] Loading images ...")

    root_dir = listdir(directory_root)

    for directory in root_dir :

        # remove .DS_Store from list

        if directory == ".DS_Store" :

            root_dir.remove(directory)



    for plant_folder in root_dir :

        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")

        

        for disease_folder in plant_disease_folder_list :

            # remove .DS_Store from list

            if disease_folder == ".DS_Store" :

                plant_disease_folder_list.remove(disease_folder)



        for plant_disease_folder in plant_disease_folder_list:

            print(f"[INFO] Processing {plant_disease_folder} ...")

            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")

                

            for single_plant_disease_image in plant_disease_image_list :

                if single_plant_disease_image == ".DS_Store" :

                    plant_disease_image_list.remove(single_plant_disease_image)



            for image in plant_disease_image_list[:200]:

                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"

                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:

                    image_list.append(convert_image_to_array(image_directory))

                    label_list.append(plant_disease_folder)

    print("[INFO] Image loading completed")  

except Exception as e:

    print(f"Error : {e}")
image_size = len(image_list)
label_binarizer = LabelBinarizer()

image_labels = label_binarizer.fit_transform(label_list)

pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))

n_classes = len(label_binarizer.classes_)
print(label_binarizer.classes_)
label_binarizer.classes_[13]
np_image_list = np.array(image_list, dtype=np.float16) / 225.0
print("[INFO] Spliting data to train, test")

x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 
aug = ImageDataGenerator(

    rotation_range=25, width_shift_range=0.1,

    height_shift_range=0.1, shear_range=0.2, 

    zoom_range=0.2,horizontal_flip=True, 

    fill_mode="nearest")
model = Sequential()

inputShape = (height, width, depth)

chanDim = -1

if K.image_data_format() == "channels_first":

    inputShape = (depth, height, width)

    chanDim = 1

model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(n_classes))

model.add(Activation("softmax"))
model.summary()
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# distribution

model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the network

print("[INFO] training network...")
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(x_train)
history = model.fit_generator(

    aug.flow(x_train, y_train, batch_size=BS),

    validation_data=(x_test, y_test),

    steps_per_epoch=len(x_train) // BS,

    epochs=EPOCHS, verbose=1

    )
print("[INFO] Calculating model accuracy")

scores = model.evaluate(x_test, y_test)

print(f"Test Accuracy: {scores[1]*100}")
%matplotlib inline
def mention_output(x):

    plt.imshow(array_to_img(x))

    return label_binarizer.classes_[np.argmax(model.predict(np.expand_dims(x, axis = 0)))]
mention_output(x_test[56])
from keras.models import model_from_json

from keras.models import load_model







import cv2
img3 = cv2.imread('C:\Vignesh\Projects\leaf disease\Test Images sample\pot_EB.JPG')

#img3 = cv2.resize(img3,(224,224))

#img3 = np.reshape(img3,[1,224,224,3])
# serialize weights to HDF5

model.save("model_num.h5")
import numpy as np

import keras 

import cv2

import matplotlib.pyplot as plt

%matplotlib inline


model.save("model_weights.h5")


model = load_model('model_weights.h5')


model.summary()


from keras import models

from keras import layers



from keras import optimizers

from keras.preprocessing import image

from IPython.display import Image, display

from learntools.deep_learning.decode_predictions import decode_predictions

import numpy as np

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.preprocessing.image import load_img, img_to_array


model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
img = cv2.imread('C:\Vignesh\Projects\leaf disease\Test Images sample\fffee500-8469-4c0f-a17d-d95c5516b446___Matt.S_CG 6210.JPG')

img = np.full((100,80,3), 12, dtype = np.uint8)

img = cv2.resize(img,(256,256))

img = np.reshape(img,[1,256,256,3])

#plt.imshow(img)

#plt.show()

classes = model.predict_classes(img)



print (classes)

"""

model.load_weights("model_weights.h5")

img = cv2.imread('C:\Vignesh\Projects\leaf disease\Test Images sample\fffee500-8469-4c0f-a17d-d95c5516b446___Matt.S_CG 6210.JPG')

test_img = cv2.resize(img,(256,256),interpolation = cv2.INTER_AREA)

img_class = model.predict_classes(test_img)

prediction = img_class[0]

classname = img_class[0]

print("Class: ",classname)

def create_model():

  if K.image_data_format() == "channels_first":

    inputShape = (depth, height, width)

    chanDim = 1

    model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))

    model.add(Activation("relu"))

    model.add(BatchNormalization(axis=chanDim))

    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))

    model.add(Activation("relu"))

    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3, 3), padding="same"))

    model.add(Activation("relu"))

    model.add(BatchNormalization(axis=chanDim))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))

    model.add(Activation("relu"))

    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3, 3), padding="same"))

    model.add(Activation("relu"))

    model.add(BatchNormalization(axis=chanDim))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(1024))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(n_classes))

    model.add(Activation("softmax"))

    

    return model



img = cv2.imread('C:\Vignesh\Projects\leaf disease\Test Images sample\fffee500-8469-4c0f-a17d-d95c5516b446___Matt.S_CG 6210.JPG')

model = create_model()

model=load_model('model_num.h5')

model.predict(img, steps=None)



model=create_model()

model = load_model('model_num.h5')

img = cv2.imread('C:\Vignesh\Projects\leaf disease\Test Images sample\fffee500-8469-4c0f-a17d-d95c5516b446___Matt.S_CG 6210.JPG')

img = np.full((100,80,3), 12, dtype = np.uint8)

x = cv2.resize(img,(1,256,256,3))

#x = np.reshape(img,[1,256,256,3])

#img_path = 'C:\Vignesh\Projects\leaf disease\Test Images sample\pot_EB.JPG'

#img = image.load_img(img_path, target_size=(224, 224))

#x = image.img_to_array(img)

#x = np.expand_dims(x, axis=0)

x = preprocess_input(x)



preds = model.predict(x)

# decode the results into a list of tuples (class, description, probability)

# (one such list for each sample in the batch)

print('Predicted:', decode_predictions(preds, top=3)[0])

"""
reverse_mapping= ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',

 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',

 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',

 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',

 'Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot',

 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',

 'Tomato_healthy']