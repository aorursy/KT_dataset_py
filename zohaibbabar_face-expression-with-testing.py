import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf



import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers import Dense, Activation, Dropout, Flatten



from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# get the data

filname = '../input/fer2018/fer20131.csv'

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

names=['emotion','pixels','usage']

df=pd.read_csv('../input/fer2018/fer20131.csv',names=names, na_filter=False)

im=df['pixels']

df.head(10)
def getData(filname):

    # images are 48x48

    # N = 35887

    Y = []

    X = []

    first = True

    for line in open(filname):

        if first:

            first = False

        else:

            row = line.split(',')

            Y.append(int(row[0]))

            X.append([int(p) for p in row[1].split()])



    X, Y = np.array(X) / 255.0, np.array(Y)

    return X, Y
X, Y = getData(filname)

num_class = len(set(Y))

print(num_class)
# keras with tensorflow backend

N, D = X.shape

X = X.reshape(N, 48, 48, 1)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)

y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)
def save_data(X_test, y_test, fname=''):

    """

    The function stores loaded data into numpy form for further processing

    """

    np.save( 'X_test' + fname, X_test)

    np.save( 'y_test' + fname, y_test)

save_data(X_test, y_test,"_privatetest6_100pct")

X_fname = 'X_test_privatetest6_100pct.npy'

y_fname = 'y_test_privatetest6_100pct.npy'

X = np.load(X_fname)

y = np.load(y_fname)

print ('Private test set')

y_labels = [np.argmax(lst) for lst in y]

counts = np.bincount(y_labels)

labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']

print (zip(labels, counts))
def overview(start, end, X):

    """

    The function is used to plot first several pictures for overviewing inputs format

    """

    fig = plt.figure(figsize=(20,20))

    for i in range(start, end+1):

        input_img = X[i:(i+1),:,:,:]

        ax = fig.add_subplot(16,12,i+1)

        ax.imshow(input_img[0,:,:,0], cmap=plt.cm.gray)

        plt.xticks(np.array([]))

        plt.yticks(np.array([]))

        plt.tight_layout()

    plt.show()

overview(0,191, X)
## Similarly we canvisualize any input with self-defined index with following code

input_img = X[6:7,:,:,:] 

print (input_img.shape)

plt.imshow(input_img[0,:,:,0], cmap='gray')

plt.show()
from keras.models import Sequential

from keras.layers import Dense , Activation , Dropout ,Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.metrics import categorical_accuracy

from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint

from keras.optimizers import *

from keras.layers.normalization import BatchNormalization
def my_model():

    model = Sequential()

    input_shape = (48,48,1)

    model.add(Conv2D(64, (5, 5), input_shape=input_shape,activation='relu', padding='same'))

    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))

    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))

    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dense(128))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(Dense(7))

    model.add(Activation('softmax'))

    

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

    # UNCOMMENT THIS TO VIEW THE ARCHITECTURE

    #model.summary()

    

    return model

model=my_model()

model.summary()
path_model='model_filter.h5' # save model at this location after each epoch

model=my_model() # create the model

h=model.fit(x=X_train,     

            y=y_train, 

            batch_size=128, 

            epochs=100, 

            verbose=1, 

            validation_data=(X_test,y_test),

            shuffle=True,

            callbacks=[

                ModelCheckpoint(filepath=path_model),

            ]

            )
objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

y_pos = np.arange(len(objects))

print(y_pos)
def emotion_analysis(emotions):

    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.9)

    plt.tick_params(axis='x', which='both', pad=10,width=4,length=10)

    plt.xticks(y_pos, objects)

    plt.ylabel('percentage')

    plt.title('emotion')

    

plt.show()
y_pred=model.predict(X_test)

#print(y_pred)

y_test.shape
from skimage import io

from PIL import Image

import numpy as np

img = image.load_img('../input/ckplus/CK+48/anger/S010_004_00000017.png', grayscale=True, target_size=(48, 48))

show_img=image.load_img('../input/ckplus/CK+48/anger/S010_004_00000017.png', grayscale=False, target_size=(200, 200))

x = image.img_to_array(img)

x = np.expand_dims(x, axis = 0)



x /= 255



custom = model.predict(x)

#print(custom[0])

emotion_analysis(custom[0])



x = np.array(x, 'float32')

x = x.reshape([48, 48]);



plt.gray()

plt.imshow(show_img)

plt.show()



m=0.000000000000000000001

a=custom[0]

for i in range(0,len(a)):

    if a[i]>m:

        m=a[i]

        ind=i

        

print('Expression Prediction:',objects[ind])
from skimage import io

from PIL import Image

import numpy as np

img = image.load_img('../input/ckplus/CK+48/contempt/S138_008_00000008.png', grayscale=True, target_size=(48, 48))

show_img=image.load_img('../input/ckplus/CK+48/contempt/S138_008_00000008.png', grayscale=False, target_size=(200, 200))

x = image.img_to_array(img)

x = np.expand_dims(x, axis = 0)



x /= 255



custom = model.predict(x)

#print(custom[0])

emotion_analysis(custom[0])



x = np.array(x, 'float32')

x = x.reshape([48, 48]);



plt.gray()

plt.imshow(show_img)

plt.show()



m=0.000000000000000000001

a=custom[0]

for i in range(0,len(a)):

    if a[i]>m:

        m=a[i]

        ind=i

        

print('Expression Prediction:',objects[ind])
from skimage import io

from PIL import Image

import numpy as np

img = image.load_img('../input/ckplus/CK+48/happy/S011_006_00000013.png', grayscale=True, target_size=(48, 48))

show_img=image.load_img('../input/ckplus/CK+48/happy/S011_006_00000013.png', grayscale=False, target_size=(200, 200))

x = image.img_to_array(img)

x = np.expand_dims(x, axis = 0)



x /= 255



custom = model.predict(x)

#print(custom[0])

emotion_analysis(custom[0])



x = np.array(x, 'float32')

x = x.reshape([48, 48]);



plt.gray()

plt.imshow(show_img)

plt.show()



m=0.000000000000000000001

a=custom[0]

for i in range(0,len(a)):

    if a[i]>m:

        m=a[i]

        ind=i

        

print('Expression Prediction:',objects[ind])
from skimage import io

from PIL import Image

import numpy as np

img = image.load_img('../input/ckplus/CK+48/contempt/S138_008_00000009.png', grayscale=True, target_size=(48, 48))

show_img=image.load_img('../input/ckplus/CK+48/contempt/S138_008_00000009.png', grayscale=False, target_size=(200, 200))

x = image.img_to_array(img)

x = np.expand_dims(x, axis = 0)



x /= 255



custom = model.predict(x)

#print(custom[0])

emotion_analysis(custom[0])



x = np.array(x, 'float32')

x = x.reshape([48, 48]);



plt.gray()

plt.imshow(show_img)

plt.show()



m=0.000000000000000000001

a=custom[0]

for i in range(0,len(a)):

    if a[i]>m:

        m=a[i]

        ind=i

        

print('Expression Prediction:',objects[ind])
from skimage import io

from PIL import Image

import numpy as np

img = image.load_img('../input/iamgesdataset/neutral.webp', grayscale=True, target_size=(48, 48))

show_img=image.load_img('../input/iamgesdataset/neutral.webp', grayscale=False, target_size=(200, 200))

x = image.img_to_array(img)

x = np.expand_dims(x, axis = 0)



x /= 255



custom = model.predict(x)

#print(custom[0])

emotion_analysis(custom[0])



x = np.array(x, 'float32')

x = x.reshape([48, 48]);



plt.gray()

plt.imshow(show_img)

plt.show()



m=0.000000000000000000001

a=custom[0]

for i in range(0,len(a)):

    if a[i]>m:

        m=a[i]

        ind=i

        

print('Expression Prediction:',objects[ind])
from skimage import io

from PIL import Image

import numpy as np

img = image.load_img('../input/iamgesdataset/surprise.webp', grayscale=True, target_size=(48, 48))

show_img=image.load_img('../input/iamgesdataset/surprise.webp', grayscale=False, target_size=(200, 200))

x = image.img_to_array(img)

x = np.expand_dims(x, axis = 0)



x /= 255



custom = model.predict(x)

#print(custom[0])

emotion_analysis(custom[0])



x = np.array(x, 'float32')

x = x.reshape([48, 48]);



plt.gray()

plt.imshow(show_img)

plt.show()



m=0.000000000000000000001

a=custom[0]

for i in range(0,len(a)):

    if a[i]>m:

        m=a[i]

        ind=i

        

print('Expression Prediction:',objects[ind])