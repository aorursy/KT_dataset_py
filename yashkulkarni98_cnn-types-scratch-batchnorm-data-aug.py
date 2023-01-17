import os

from tensorflow import keras

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import numpy as np
kaggleDir = '/kaggle/input/state-farm-distracted-driver-detection/'

train_img_dir = 'train/'

test_img_dir = 'test/'
df_train = pd.read_csv(kaggleDir + 'driver_imgs_list.csv', low_memory=True)

print('Number of Samples in trainset : {}'.format(df_train.shape[0]))

print('Number Of districted Classes : {}'.format(len((df_train.classname).unique())))
df_train.head()
class_freq_count = df_train.classname.value_counts()



class_freq_count.plot(kind='bar', label='index')

plt.title('Sample Per Class');

plt.show()
CLASSES = {"c0": "safe driving", "c1": "texting - right", "c2": "talking on the phone - right", "c3": "texting - left",

           "c4": "talking on the phone - left", "c5": "operating the radio", "c6": "drinking", "c7": "reaching behind",

           "c8": "hair and makeup", "c9": " talking to passenger"}

plt.pie(class_freq_count, autopct='%1.1f%%', shadow=True, labels=CLASSES.values())

plt.title('Sample % per class');

plt.show()
dataset = pd.read_csv('../input/state-farm-distracted-driver-detection/driver_imgs_list.csv')

dataset.head(5)
# Plot figure size

plt.figure(figsize = (4,4))

# Count the number of images per category

sns.countplot(x = 'classname', data = dataset)

# Change the Axis names

plt.ylabel('Count')

plt.title('Categories Distribution')

# Show plot

plt.show()
dataset['class_type'] = dataset['classname'].str.extract('(\d)',expand=False).astype(np.float)

plt.figure(figsize = (10,10))

dataset.hist('class_type', alpha=0.5, layout=(1,1), bins=10)

plt.title('Class distribution')

plt.show()
from matplotlib import image

from matplotlib import pyplot



#load image as pixel array

data=image.imread(r"../input/state-farm-distracted-driver-detection/imgs/train/c1/img_448.jpg")

#summarize shape of the pixel array

print(data.dtype)

print(data.shape)

#display the arrays of pixels as an image

pyplot.imshow(data)

pyplot.show()
import cv2
from matplotlib import image

from matplotlib import pyplot



#load image as pixel array

data=cv2.imread(r"../input/state-farm-distracted-driver-detection/imgs/train/c1/img_448.jpg",cv2.IMREAD_GRAYSCALE)

#summarize shape of the pixel array

data=cv2.resize(data,(240,240))

print(data.dtype)

print(data.shape)

#display the arrays of pixels as an image

pyplot.imshow(data)

pyplot.show()
from matplotlib import image

from matplotlib import pyplot



#load image as pixel array

data=cv2.imread(r"../input/state-farm-distracted-driver-detection/imgs/train/c2/img_100029.jpg",cv2.IMREAD_GRAYSCALE)

#summarize shape of the pixel array

data=cv2.resize(data,(240,240))

print(data.dtype)

print(data.shape)

#display the arrays of pixels as an image

pyplot.imshow(data)

pyplot.show()
from matplotlib import image

from matplotlib import pyplot



#load image as pixel array

data=cv2.imread(r"../input/state-farm-distracted-driver-detection/imgs/train/c3/img_100041.jpg",cv2.IMREAD_GRAYSCALE)

#summarize shape of the pixel array

data=cv2.resize(data,(240,240))

print(data.dtype)

print(data.shape)

#display the arrays of pixels as an image

pyplot.imshow(data)

pyplot.show()
from matplotlib import image

from matplotlib import pyplot



#load image as pixel array

data=cv2.imread(r"../input/state-farm-distracted-driver-detection/imgs/train/c5/img_100027.jpg",cv2.IMREAD_GRAYSCALE)

#summarize shape of the pixel array

data=cv2.resize(data,(240,240))

print(data.dtype)

print(data.shape)

#display the arrays of pixels as an image

pyplot.imshow(data)

pyplot.show()
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
files=os.listdir(r"../input/state-farm-distracted-driver-detection/imgs/train")

print("Number of files in train:",len(files))
c0=os.listdir(r"../input/state-farm-distracted-driver-detection/imgs/train/c0")

print("Number of images in C0 class:",len(c0))
c1=os.listdir(r"../input/state-farm-distracted-driver-detection/imgs/train/c1")

print("Number of images in C1 class:",len(c1))
test_im=os.listdir(r"../input/state-farm-distracted-driver-detection/imgs/test")

print("Number of images in Test:",len(test_im))
from matplotlib import image

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



activity_map = {'c0': 'Safe driving', 

                'c1': 'Texting - right', 

                'c2': 'Talking on the phone - right', 

                'c3': 'Texting - left', 

                'c4': 'Talking on the phone - left', 

                'c5': 'Operating the radio', 

                'c6': 'Drinking', 

                'c7': 'Reaching behind', 

                'c8': 'Hair and makeup', 

                'c9': 'Talking to passenger'}





plt.figure(figsize = (12, 20))

image_count = 1

BASE_URL = r'../input/state-farm-distracted-driver-detection/imgs/train/'

for directory in os.listdir(BASE_URL):

    if directory[0] != '.':

        for i, file in enumerate(os.listdir(BASE_URL + directory)):

            if i == 1:

                break

            else:

                fig = plt.subplot(5, 2, image_count)

                image_count += 1

                image = mpimg.imread(BASE_URL + directory + '/' + file)

                plt.imshow(image)

                plt.title(activity_map[directory])
train_directory = r"../input/state-farm-distracted-driver-detection/imgs/train"

test_directory =r"../input/state-farm-distracted-driver-detection/imgs/test"

classes = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
img_size1 = 240

img_size2 = 240
import os

import cv2

#TRAINING DATA



training_data = []

i = 0

def create_training_data():

    for category in classes:

        path = os.path.join(train_directory,category)

        class_num = classes.index(category)

        

        for img in os.listdir(path):

            img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

            new_img = cv2.resize(img_array,(img_size2,img_size1))

            training_data.append([new_img,class_num])
testing_data = []

i = 0

def create_testing_data():        

    for img in os.listdir(test_directory):

        img_array = cv2.imread(os.path.join(test_directory,img),cv2.IMREAD_GRAYSCALE)

        new_img = cv2.resize(img_array,(img_size2,img_size1))

        testing_data.append([img,new_img])
create_training_data()
create_testing_data()
import random

random.shuffle(training_data)
x = []

y = []

for features, label in training_data:

    x.append(features)

    y.append(label)
import numpy as np

X = np.array(x).reshape(-1,img_size2,img_size1,1)

X[0].shape
y=np.array(y).reshape(-1,1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=50)
x_test.shape
y_test.shape
from keras.utils import np_utils

Y_train = np_utils.to_categorical(y_train,num_classes=10)

Y_test = np_utils.to_categorical(y_test,num_classes=10)
Y_test.shape
Y_train.shape
x_train.shape
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Flatten, Activation

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

model = keras.models.Sequential()
from keras.callbacks import ModelCheckpoint

models_dir = "saved_models"

if not os.path.exists(models_dir):

    os.makedirs(models_dir)
def create_model_v1():

    # Vanilla CNN model

    model = Sequential()



    model.add(Conv2D(filters = 64, kernel_size = 3, padding='same', activation = 'relu', input_shape=(240,240,1)))

    model.add(MaxPooling2D(pool_size = 2))



    model.add(Conv2D(filters = 128, padding='same', kernel_size = 3, activation = 'relu'))

    model.add(MaxPooling2D(pool_size = 2))



    model.add(Conv2D(filters = 256, padding='same', kernel_size = 3, activation = 'relu'))

    model.add(MaxPooling2D(pool_size = 2))



    model.add(Conv2D(filters = 512, padding='same', kernel_size = 3, activation = 'relu'))

    model.add(MaxPooling2D(pool_size = 2))



    model.add(Flatten())



    model.add(Dense(500, activation = 'relu'))

    model.add(Dense(10, activation = 'softmax'))

    

    return model
model_v1 = create_model_v1()



# More details about the layers

model_v1.summary()



# Compiling the model

model_v1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# Training the Vanilla Model 

history_v1 = model_v1.fit(x_train, Y_train, epochs=10,batch_size=40,validation_data=(x_test , Y_test))
import matplotlib.pyplot as plt

def plot_train_history(history):

    # Summarize history for accuracy

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('Model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()



    # Summarize history for loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
plot_train_history(history_v1)
score = model_v1.evaluate(x_test, Y_test, verbose=1)

print('Score of V1 model: ', score)
preds = model_v1.predict(np.array(testing_data[100][1]).reshape(-1,img_size2,img_size1,1))

test_data = np.array(testing_data[100][1]).reshape(-1,img_size2,img_size1,1)



preds = model_v1.predict(test_data)

preds
import matplotlib.pyplot as plt



print('Predicted: {}'.format(np.argmax(preds)))



new_img = cv2.resize(testing_data[100][1],(img_size2,img_size1))

plt.imshow(new_img,cmap='gray')

plt.show()
preds = model_v1.predict(np.array(testing_data[200][1]).reshape(-1,img_size2,img_size1,1))

test_data = np.array(testing_data[200][1]).reshape(-1,img_size2,img_size1,1)



preds = model_v1.predict(test_data)

preds
import matplotlib.pyplot as plt



print('Predicted: {}'.format(np.argmax(preds)))



new_img = cv2.resize(testing_data[200][1],(img_size2,img_size1))

plt.imshow(new_img,cmap='gray')

plt.show()
preds = model_v1.predict(np.array(testing_data[300][1]).reshape(-1,img_size2,img_size1,1))

test_data = np.array(testing_data[300][1]).reshape(-1,img_size2,img_size1,1)



preds = model_v1.predict(test_data)

preds
import matplotlib.pyplot as plt



print('Predicted : {}'.format(np.argmax(preds)))



new_img = cv2.resize(testing_data[300][1],(img_size2,img_size1))

plt.imshow(new_img,cmap='gray')

plt.show()
preds = model_v1.predict(np.array(testing_data[400][1]).reshape(-1,img_size2,img_size1,1))

test_data = np.array(testing_data[400][1]).reshape(-1,img_size2,img_size1,1)



preds = model_v1.predict(test_data)

preds
import matplotlib.pyplot as plt



print('Predicted: {}'.format(np.argmax(preds)))



new_img = cv2.resize(testing_data[400][1],(img_size2,img_size1))

plt.imshow(new_img,cmap='gray')

plt.show()
preds = model_v1.predict(np.array(testing_data[500][1]).reshape(-1,img_size2,img_size1,1))

test_data = np.array(testing_data[500][1]).reshape(-1,img_size2,img_size1,1)



preds = model_v1.predict(test_data)

preds
import matplotlib.pyplot as plt



print('Predicted: {}'.format(np.argmax(preds)))



new_img = cv2.resize(testing_data[500][1],(img_size2,img_size1))

plt.imshow(new_img,cmap='gray')

plt.show()
def create_model_v2():

    # Optimised Vanilla CNN model

    model = Sequential()



    ## CNN 1

    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(240,240,1)))

    model.add(BatchNormalization())

    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))

    model.add(BatchNormalization(axis = 3))

    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

    model.add(Dropout(0.3))



    ## CNN 2

    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))

    model.add(BatchNormalization(axis = 3))

    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

    model.add(Dropout(0.3))



    ## CNN 3

    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))

    model.add(BatchNormalization(axis = 3))

    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

    model.add(Dropout(0.5))



    ## Output

    model.add(Flatten())

    model.add(Dense(512,activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(128,activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(10,activation='softmax'))



    return model
model_v2 = create_model_v2()



# More details about the layers

model_v2.summary()



# Compiling the model

model_v2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint_cb=keras.callbacks.ModelCheckpoint("saved_models/cnn_vanilla.hdf5",save_best_only=True)

history_v2 = model_v2.fit(x_train, Y_train, epochs=10,batch_size=40,validation_data=(x_test , Y_test),callbacks=[checkpoint_cb])
!rm -f saved_models/cnn_vanilla.hdf5
import matplotlib.pyplot as plt

def plot_train_history(history):

    # Summarize history for accuracy

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('Model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()



    # Summarize history for loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
plot_train_history(history_v2)
score = model_v2.evaluate(x_test, Y_test, verbose=1)

print('Score: ', score)
preds = model_v2.predict(np.array(testing_data[101][1]).reshape(-1,img_size2,img_size1,1))

test_data = np.array(testing_data[101][1]).reshape(-1,img_size2,img_size1,1)



preds = model_v2.predict(test_data)

preds
import matplotlib.pyplot as plt



print('Predicted: {}'.format(np.argmax(preds)))



new_img = cv2.resize(testing_data[101][1],(img_size2,img_size1))

plt.imshow(new_img,cmap='gray')

plt.show()
preds = model_v2.predict(np.array(testing_data[201][1]).reshape(-1,img_size2,img_size1,1))

test_data = np.array(testing_data[201][1]).reshape(-1,img_size2,img_size1,1)



preds = model_v2.predict(test_data)

preds
import matplotlib.pyplot as plt



print('Predicted: {}'.format(np.argmax(preds)))



new_img = cv2.resize(testing_data[201][1],(img_size2,img_size1))

plt.imshow(new_img,cmap='gray')

plt.show()
preds = model_v2.predict(np.array(testing_data[301][1]).reshape(-1,img_size2,img_size1,1))

test_data = np.array(testing_data[301][1]).reshape(-1,img_size2,img_size1,1)



preds = model_v2.predict(test_data)

preds
import matplotlib.pyplot as plt



print('Predicted: {}'.format(np.argmax(preds)))



new_img = cv2.resize(testing_data[301][1],(img_size2,img_size1))

plt.imshow(new_img,cmap='gray')

plt.show()
preds = model_v2.predict(np.array(testing_data[401][1]).reshape(-1,img_size2,img_size1,1))

test_data = np.array(testing_data[401][1]).reshape(-1,img_size2,img_size1,1)



preds = model_v2.predict(test_data)

preds
import matplotlib.pyplot as plt



print('Predicted: {}'.format(np.argmax(preds)))



new_img = cv2.resize(testing_data[401][1],(img_size2,img_size1))

plt.imshow(new_img,cmap='gray')

plt.show()
preds = model_v2.predict(np.array(testing_data[501][1]).reshape(-1,img_size2,img_size1,1))

test_data = np.array(testing_data[501][1]).reshape(-1,img_size2,img_size1,1)



preds = model_v2.predict(test_data)

preds
import matplotlib.pyplot as plt



print('Predicted: {}'.format(np.argmax(preds)))



new_img = cv2.resize(testing_data[501][1],(img_size2,img_size1))

plt.imshow(new_img,cmap='gray')

plt.show()
def create_model_v3():

    model = Sequential()

    

    model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', input_shape = (240, 240, 3), data_format = 'channels_last'))

    model.add(MaxPooling2D(pool_size = (2, 2)))

    

    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))

    model.add(MaxPooling2D(pool_size = (2, 2)))

    

    model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))

    model.add(MaxPooling2D(pool_size = (2, 2)))

    

    model.add(Flatten())

    model.add(Dense(units = 1024, activation = 'relu'))

    

    model.add(Dense(units = 256, activation = 'relu'))

    model.add(Dense(units = 10, activation = 'sigmoid'))





    

    return model
model_v3 = create_model_v3()



# More details about the layers

model_v3.summary()



# Compiling the model

model_v3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1.0/255, 

                                   shear_range = 0.2, 

                                   zoom_range = 0.2, 

                                   horizontal_flip = True, 

                                   validation_split = 0.2)



training_set = train_datagen.flow_from_directory('../input/state-farm-distracted-driver-detection/imgs/train', 

                                                 target_size = (240, 240), 

                                                 batch_size = 32,

                                                 subset = 'training')



validation_set = train_datagen.flow_from_directory('../input/state-farm-distracted-driver-detection/imgs/train', 

                                                   target_size = (240, 240), 

                                                   batch_size = 32,

                                                   subset = 'validation')

import os

os.chdir(r'/kaggle/working')
checkpoint_cb=keras.callbacks.ModelCheckpoint("./cnn_vanilla.hdf5",save_best_only=True)

history = model_v3.fit_generator(training_set,

                         steps_per_epoch = 17943/32,

                         epochs = 10,

                         validation_data = validation_set,

                         validation_steps = 4481/32,

                         callbacks=[checkpoint_cb])
model_v3.save_weights('./cnn_vanilla.hdf5', overwrite=True)
model_v3.save('./cnn_vanilla.hdf5')
model=keras.models.load_model("cnn_vanilla.hdf5")
from keras.models import load_model

loaded_model = load_model('saved_models/cnn_vanilla.hdf5')
import matplotlib.pyplot as plt

def plot_train_history(history):

    # Summarize history for accuracy

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('Model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()



    # Summarize history for loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
plot_train_history(history)
from keras.models import load_model

import os

import numpy as np

from keras.preprocessing import image

import cv2

import matplotlib.pyplot as plt



model = load_model('cnn_vanilla.hdf5')



test_data_dir = r'../input/state-farm-distracted-driver-detection/imgs/test/'





class_labels = [

    "normal driving",

    "texting - right",

    "talking on the phone - right",

    "texting - left",

    "talking on the phone - left",

    "operating the radio",

    "drinking",

    "reaching behind",

    "hair and makeup",

    "talking to passenger"

]



file_names = np.random.choice(os.listdir(test_data_dir),550)



img_arrays = []



for file_name in file_names:

    img = image.load_img(os.path.join(test_data_dir, file_name), target_size=(240, 240))

    img_array = image.img_to_array(img)

    img_arrays.append(img_array)



img_arrays = np.array(img_arrays)

img_arrays = img_arrays.astype('float32') / 255

#predictions = model.predict(np.array(img_arrays).reshape(-1,240,240,1))

predictions = model.predict(img_arrays)





label_indexes = np.argmax(predictions, axis=1)

probabilities = np.max(predictions, axis=1)



for (file_name, label_index, probability) in zip(file_names, label_indexes, probabilities):

    if probability < 0.50:

        continue



    label_with_probability = "{}: {:.2f}%".format(class_labels[label_index], probability * 100)



    import cv2



    image = cv2.imread(os.path.join(test_data_dir, file_name))

    #image = cv2.resize(image, (240,240))



    cv2.putText(image, label_with_probability.upper(), (210, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)



    c=cv2.imwrite("annotated-results/" + file_name, image)

    

    plt.imshow(image)

    plt.show()