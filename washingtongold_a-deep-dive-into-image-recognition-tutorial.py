import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #commenting this to save space

        #print(os.path.join(dirname, filename))

        pass
import matplotlib.pyplot as plt

image = plt.imread('/kaggle/input/natural-images/natural_images/fruit/fruit_0618.jpg')
image.shape
plt.imshow(image); #adding ; helps suppress printed material
import PIL

from PIL import Image

import numpy as np



#read image from directory using Image

img = Image.open('/kaggle/input/natural-images/natural_images/fruit/fruit_0618.jpg')



#reshape the image to (128, 128). This maintains the aspect ratio (images may be squished/stretched)

img = img.resize((128,128), Image.ANTIALIAS)



#we can convert to an array directly using np.array(PIL image)

img = np.array(img)



#display the image with matplotlib

plt.imshow(img);
X, y = [], []
import numpy as np #import numpy

from tqdm import tqdm #import tqdm for progress bar



#collect names of directories we will be pulling from

#this is necessary because there are other duplicate subdirectories we do not want to go over twice

dirs = ['/kaggle/input/natural-images/natural_images/fruit',

        '/kaggle/input/natural-images/natural_images/flower',

        '/kaggle/input/natural-images/natural_images/person',

        '/kaggle/input/natural-images/natural_images/car',

        '/kaggle/input/natural-images/natural_images/motorbike',

        '/kaggle/input/natural-images/natural_images/airplane',

        '/kaggle/input/natural-images/natural_images/dog',

        '/kaggle/input/natural-images/natural_images/cat']



#we will switch the y label when we finish with a directory

current_y_label = 0

for dirname, _, filenames in os.walk('/kaggle/input'):

    

    #if the directory name is not valid, begin the next iteration

    if dirname not in dirs: continue

        

    #for each file (image.png) in the filenames

    for filename in tqdm(filenames): #use tqdm around an iterator to display progress bar

        

        #combine paths to form complete directory

        directory = os.path.join(dirname, filename)

        

        #read image from directory using Image

        img = Image.open(directory)



        #reshape the image to (128, 128). This maintains the aspect ratio (images may be squished/stretched)

        img = img.resize((128,128), Image.ANTIALIAS)



        #we can convert to an array directly using np.array(PIL image)

        img = np.array(img)

        

        #append the array to X (after dividing it by 255)

        X.append(img/255)

        

        #append the y label as the y label

        y.append(current_y_label)

        

    #we're finished with the directory. time to change the y label

    current_y_label += 1
from keras.utils import to_categorical

X, y = np.array(X), to_categorical(np.array(y))
X.shape
y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
shapes = {'X train': X_train.shape, 'X test': X_test.shape,

          'y train': y_train.shape, 'y test': y_test.shape}

for key in shapes:

    print(f"{key}: {shapes[key]}")
X_train.shape
y_train.shape
import keras

from keras.models import Sequential #our model

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense #import layers
model = Sequential() #initiate model

model.add(Conv2D(64, kernel_size=(5,5), input_shape=(128,128,3)))

model.add(MaxPooling2D(pool_size=(5,5)))

model.add(Conv2D(64, kernel_size=(3,3)))

model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(8, activation='softmax'))
model.summary()
model.compile(metrics=['accuracy'], 

              loss='categorical_crossentropy',

              optimizer='adam')
history = model.fit(X_train, y_train, epochs=15)
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(10, 4))

sns.set_style('whitegrid')

sns.lineplot(x=range(1,16), y=history.history['accuracy'], marker='o', color='#EA4335', label='Accuracy')

sns.lineplot(x=range(1,16), y=history.history['loss'], marker='X', color='#4285F4', label='Loss')

plt.title("Performance with Standard CNN")

plt.xlabel('Epochs');

plt.ylabel('Accuracy | Loss');
model.evaluate(X_test, y_test)
from keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(

                        featurewise_center=False,

                        featurewise_std_normalization=False,

                        rotation_range=10,

                        width_shift_range=0.1,

                        height_shift_range=0.1,

                        zoom_range=.25,

                        horizontal_flip=True)
#this will yield an error but we'll still get to see an example augmented image

plt.imshow(data_generator.flow(X,y)[0][0][0]);

plt.show()

plt.imshow(data_generator.flow(X,y)[0][0][1]);

plt.show()

plt.imshow(data_generator.flow(X,y)[0][0][2]);

plt.show()
model1 = Sequential() #initiate model

model1.add(Conv2D(64, kernel_size=(5,5), input_shape=(128,128,3)))

model1.add(MaxPooling2D(pool_size=(5,5)))

model1.add(Conv2D(64, kernel_size=(3,3)))

model1.add(MaxPooling2D(pool_size=(3,3)))

model1.add(Flatten())

model1.add(Dense(128, activation='relu'))

model1.add(Dense(8, activation='softmax'))

model1.compile(metrics=['accuracy'], 

              loss='categorical_crossentropy',

              optimizer='adam')
history1 = model1.fit_generator(data_generator.flow(X_train, y_train), epochs=25)
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(10, 4))

sns.set_style('whitegrid')

sns.lineplot(x=range(1,26), y=history1.history['accuracy'], marker='o', color='#EA4335', label='Accuracy')

sns.lineplot(x=range(1,26), y=history1.history['loss'], marker='X', color='#4285F4', label='Loss')

plt.title("Performance with Augmented Data")

plt.xlabel('Epochs');

plt.ylabel('Accuracy | Loss');
model1.evaluate(X_test, y_test)