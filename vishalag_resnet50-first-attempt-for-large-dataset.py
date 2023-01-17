import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train_data = pd.read_csv('train_greaterthan4.csv')# list of landmarks which have more than 4 id's
test_data = pd.read_csv('test.csv')
print("Training data size",train_data.shape)
print("test data size",test_data.shape)
train_data.head()
test_data.head()
train_data.nunique()
from numpy import array
from keras.utils import to_categorical
list_train_X = train_data['id']
train_y = train_data['landmark_id']
import os
dir_file = os.listdir("resize_train_image")
from PIL import Image
from pylab import *
import itertools
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
values = array(train_y)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

onehot_encoder = OneHotEncoder(sparse=True)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
onehot_encoded.shape
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
def chunk_of_data(start,stop):

    images_input_x = None
    images_input_x = np.ndarray(shape=(2000, 224, 224,3), dtype =np.float16)
    count = 0
    y = onehot_encoded[start:stop]
    '''
    y_arr = array(y)
    print(y_arr.shape)
    categorical_y = to_categorical(y_arr, num_classes = 14951)
    '''
    
    #print(images_input_x.shape)
    for i in itertools.islice(list_train_X, start, stop):
        im = Image.open('resize_train_image/'+ i +'.jpg')
        im = im.resize([224,224],Image.ANTIALIAS)
        arr = np.array(im)
        #arr = img_to_array(im)
        #arr = arr.reshape((224, 224 , 3))
        arr = (arr - 128.0) / 128.0
        images_input_x[count] = arr 
        count+=1
        #if count%1000 == 0:
            #print(count)
            #print(im)
    #print(i)
    #plt.imshow(im)
    #print(y[stop-1])
    return images_input_x,y
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(12838, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
nb_epoch = 2
for e in range(nb_epoch):
    #print("epoch %d" % e)
    
    start = 0
    stop = 2000
    step = 2000
    for i in range (60):
        print("epoch %d" % e,i,"/60")
        x ,y = chunk_of_data(start,stop)
        y =y.todense()
        y =np.array(y)
        print("fit")
        datagen.fit(x)
        
        model.fit_generator(datagen.flow(x, y, batch_size=1),
                    steps_per_epoch=len(x) / 60, epochs=1)
        #model.fit(x, y, batch_size=8, nb_epoch=1)
        start = stop
        stop += step
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)
for layer in model.layers[:-11]:
   layer.trainable = False
for layer in model.layers[-11:]:
   layer.trainable = True
for layer in model.layers:
    print(layer, layer.trainable)
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
nb_epoch = 40
for e in range(nb_epoch):
    print("epoch %d" % e)
    
    start = 0
    stop = 6000
    step = 6000
    for i in range (202):
        print(i)
        x ,y = chunk_of_data(start,stop)
        y =y.todense()
        y =np.array(y)
        datagen.fit(x)
        model.fit_generator(datagen.flow(x, y, batch_size=32),
                    steps_per_epoch=len(x) / 32, epochs=1)
        #model.fit(x, y, batch_size=8, nb_epoch=1)
        start = stop
        stop += step

model.save('resnet50_with_weight_.h5')
model.save_weights('resnet_weights_only.h5')