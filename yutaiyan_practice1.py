# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

#!pip install image-classifiers==0.2.2

!pip install keras_sequential_ascii

#!pip install keras_applications

#!pip install plot_utils

import sys

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



from PIL import Image 

def change_image_channels(image, image_path):

    # 4通道转3通道

    if image.mode == 'RGBA':

        r, g, b, a = image.split()

        image = Image.merge("RGB", (r, g, b))

        image.save(image_path)

    #  1 通道转3通道

    elif image.mode != 'RGB':

        image = image.convert("RGB")

        os.remove(image_path)

        image.save(image_path)

    return image

for i in range(212):

    if((i>=189 and i<=195) or (i>=203 and i<=207)):

        continue

    im = Image.open('/kaggle/input/fruit-detection/images/fruit{}.png'.format(i))

    change_image_channels(im,'fruit_change{}.png'.format(i))
import matplotlib.pyplot as plt

ims = np.zeros((200,300, 400, 3))

q=0

for i in range(212):

    if((i>=189 and i<=195) or (i>=203 and i<=207)):

        #there is no image in this two ranges, we can delete them after organize the output.

        continue

    try:

        im = plt.imread('fruit_change{}.png'.format(i))

        ims[q,:,:,:] = im

        q=q+1

        print(q)

    except:

        try:

            im = Image.open('fruit_change{}.png'.format(i))

        except:

            im = Image.open('/kaggle/input/fruit-detection/images/fruit{}.png'.format(i))

        imBackground = im.resize((400,300))

        imBackground.save('ProcessedImage{}.png'.format(i),'png')

        im = plt.imread('ProcessedImage{}.png'.format(i))

        size = im.shape

        x = size[0]

        y = size[1]

        channels = size[2]

        #str='%d %d,%d,%d'%(i,x,y,channels)

        #print(str)

        ims[q,0:x, 0:y, 0:channels] = im

        q=q+1

        print(q)

#plt.imshow(ims[100,:,:,:])
import xml.etree.ElementTree as ET

# define that 1 is snake fruit,2 is dragon fruit, 3 is banana, 4 is pineapple

output = np.zeros((200,4))

q=0

for i in range(212):

    if((i>=189 and i<=195) or (i>=203 and i<=207)):

        #there is no image in this two ranges, we can delete them after organize the output.

        continue

    tree = ET.parse('/kaggle/input/fruit-detection/annotations/fruit{}.xml'.format(i))

    root = tree.getroot()

    # all items data

    #print('\nAll item data:')

    for elem in root:

        for subelem in elem:

            if (subelem.tag=='name'):

                if (subelem.text=='snake fruit'):

                    output[q,0]=1

                elif (subelem.text=='dragon fruit'):

                    output[q,1]=1

                elif (subelem.text=='banana'):

                    output[q,2]=1

                elif (subelem.text=='pineapple'):

                    output[q,3]=1

                #print(subelem.text)

    q=q+1

    #print(output[q-1,:])

    #print(q)

#plt.imshow(ims[:,:,:,0])

        
import keras
from keras.utils import to_categorical

print(ims.shape)

print(output.shape)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(ims,output,test_size=0.3,random_state=0)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train  /= 255

x_test /= 255



print(y_train[0])

print(y_train.shape)

print(y_test.shape)

# Import all modules

import keras  

from keras.models import Sequential  

from keras.layers import Dense, Dropout, Flatten  

from keras.layers import Conv2D, MaxPooling2D  

from keras.utils import to_categorical  

from keras.preprocessing import image  

import numpy as np  

import pandas as pd  

import matplotlib.pyplot as plt  

from sklearn.model_selection import train_test_split  

from tqdm import tqdm  

%matplotlib inline  
model = Sequential()  

model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(300,400,3)))  

model.add(MaxPooling2D(pool_size=(2, 2)))  

model.add(Dropout(0.25))  

model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))  

model.add(MaxPooling2D(pool_size=(2, 2)))  

model.add(Dropout(0.25))  

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))  

model.add(MaxPooling2D(pool_size=(2, 2)))  

model.add(Dropout(0.25))  

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))  

model.add(MaxPooling2D(pool_size=(2, 2)))  

model.add(Dropout(0.25))  

model.add(Flatten())  

model.add(Dense(128, activation='relu'))  

model.add(Dropout(0.5))  

model.add(Dense(64, activation='relu'))  

model.add(Dropout(0.5))  

model.add(Dense(4, activation='sigmoid'))  

model.summary()  
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  

model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), batch_size=64)  

scores = model.evaluate(x_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))

predictions=model.predict(x_test[0:4,:,:,:])

print(predictions)

print(y_test[0:4,:])

#plt.imshow(ims[2])
from lime import lime_image

from lime.wrappers.scikit_image import SegmentationAlgorithm

from skimage.segmentation import mark_boundaries

image_example = ims[2]

explainer = lime_image.LimeImageExplainer(verbose = False)

explanation = explainer.explain_instance(

    image_example, 

    classifier_fn = model.predict, 

    top_labels=100, 

    hide_color=0, 

    num_samples=1000

)



temp, mask = explanation.get_image_and_mask(

    explanation.top_labels[0], 

    positive_only=False, 

    num_features=5, 

    hide_rest=False

)

plt.imshow(mark_boundaries(temp, mask))

#print(mark_boundaries(temp, mask))

#plt.imshow(ims[2])

import eli5

image_example = np.expand_dims(ims[2], axis=0)

#plt.imshow(ims[2,:,:,:])

eli5.show_prediction(model, image_example)