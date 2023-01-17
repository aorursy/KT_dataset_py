%matplotlib inline

from keras.preprocessing.image import array_to_img,img_to_array,load_img

import numpy as np

import matplotlib.pyplot as plt

from glob import glob

from keras.utils import to_categorical



TRAIN_PATH = '../input/dogs-vs-cats-redux-kernels-edition/train/'

NUM_OF_IMAGES = 200

NUMBER_OF_ROTATIONS = 4



images,images_arr,labels = [],[],[]

label = 0

for path in glob(TRAIN_PATH+'dog*')[:NUM_OF_IMAGES]:

    #load the image

    img = load_img(path,target_size=(224,224))

    

    #convert the image to a tensor

    img_arr = img_to_array(img)

    

    #rotate the image according to the label

    img_arr = np.rot90(img_arr,label)

    

    #compute the rotated image

    img = array_to_img(img_arr)



    #save the images,tensors and labels

    images.append(img)

    images_arr.append(img_arr)

    labels.append(label)

    

    #next image will be rotated 90 degrees more

    label = (label+1)%NUMBER_OF_ROTATIONS

    

images_arr = np.asarray(images_arr)

labels = to_categorical(labels)
IMAGES_TO_PLOT = 8

_,axis = plt.subplots(1, IMAGES_TO_PLOT,figsize=(15,15))

for i,(img,label) in enumerate(zip(images,labels)):

    if i==IMAGES_TO_PLOT:

        break

        

    axis[i].imshow(img)

    axis[i].xaxis.set_visible(False) 

    axis[i].yaxis.set_visible(False)

    

    label = np.argmax(label)

    if label==0:

        axis[i].set_title('Original Image')

    else:

        axis[i].set_title('{}° Rotation'.format(label*90))
images_train = images[:NUM_OF_IMAGES//2]

images_test = images[NUM_OF_IMAGES//2:]



images_arr_train = images_arr[:NUM_OF_IMAGES//2]

images_arr_test = images_arr[NUM_OF_IMAGES//2:]



labels_train = labels[:NUM_OF_IMAGES//2]

labels_test = labels[NUM_OF_IMAGES//2:]
from keras.applications.vgg16 import VGG16

import h5py

from keras.engine import topology



def load_split_weights(model, model_path_pattern='model_%d.h5', memb_size=102400000):  

    """Loads weights from split hdf5 files.

    

    Parameters

    ----------

    model : keras.models.Model

        Your model.

    model_path_pattern : str

        The path name should have a "%d" wild card in it.  For "model_%d.h5", the following

        files will be expected:

        model_0.h5

        model_1.h5

        model_2.h5

        ...

    memb_size : int

        The number of bytes per hdf5 file.  

    """

    model_f = h5py.File(model_path_pattern, "r", driver="family", memb_size=memb_size)

    topology.load_weights_from_hdf5_group_by_name(model_f, model.layers)

    

    return model



'''

This code is taken from https://www.kaggle.com/ekkus93/keras-models-as-datasets-test

As we are running on Kaggle server, we can't download the VGG16 weights from github.

If you are running it on your machine, you can simply replace this code with:

base_model = VGG16(weights='imagenet', include_top=False)

'''

vgg16 = VGG16(include_top=False, weights=None)  

keras_models_dir = '../input/keras-models'

model_path_pattern = keras_models_dir + "/vgg16_weights_tf_dim_ordering_tf_kernels_%d.h5" 

base_model = load_split_weights(vgg16, model_path_pattern)



def pretrained_features(arr,base_model):

    features = base_model.predict(arr,batch_size=100, verbose=1)

    return features.reshape((features.shape[0],-1))
features_train = pretrained_features(images_arr_train,base_model)

features_test = pretrained_features(images_arr_test,base_model)
from keras.layers import Dense, GlobalAveragePooling2D,Dropout,Flatten

from keras.applications.vgg16 import VGG16,preprocess_input

from keras.models import Sequential

from keras.optimizers import Adam

from keras.regularizers import l2

from keras.callbacks import EarlyStopping,ReduceLROnPlateau

from keras.initializers import RandomNormal



model = Sequential()

model.add(Dense(128, input_dim=features_train.shape[1],activation='relu',

                kernel_regularizer=l2(0.1),kernel_initializer=RandomNormal(stddev=0.001)))

model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax',

                kernel_regularizer=l2(0.1),kernel_initializer=RandomNormal(stddev=0.001)))

model.compile(optimizer=Adam(lr=0.0001),

              loss='categorical_crossentropy',

              metrics=['accuracy'])



model.fit(features_train, labels_train, batch_size=50, epochs=15,

          validation_data=(features_test,labels_test),

          callbacks=[EarlyStopping(patience=0)])
from sklearn.metrics import confusion_matrix

import seaborn as sns





predictions = model.predict_classes(features_test)

true_classes = np.argmax(labels_test,axis=1)

cnf_matrix = confusion_matrix(true_classes, predictions)



sns.heatmap(cnf_matrix,annot=True,annot_kws={"size": 14})

plt.ylabel('True Class')

plt.xlabel('Predicted Class')
predictions = model.predict_proba(features_test)

pred_true_class = predictions[range(len(predictions)),true_classes]

sorted_images = [images_test[i] for i in np.argsort(pred_true_class)]



_,axis = plt.subplots(1, 4,figsize=(15,15))

pred_true_class.sort()

for i in range(4):

    axis[i].imshow(array_to_img(sorted_images[i]))

    axis[i].xaxis.set_visible(False) 

    axis[i].yaxis.set_visible(False)

    axis[i].set_title('predicted: {0:.3f}'.format(pred_true_class[i]))
predictions = model.predict_proba(features_test)

pred_true_class = predictions[range(len(predictions)),true_classes]

sorted_images = [images_test[i] for i in np.argsort(pred_true_class)]



_,axis = plt.subplots(1, 4,figsize=(15,15))

pred_true_class.sort()

for i in range(4):

    axis[i].imshow(array_to_img(sorted_images[-1*(i+1)]))

    axis[i].xaxis.set_visible(False) 

    axis[i].yaxis.set_visible(False)

    axis[i].set_title('predicted: {0:.3f}'.format(pred_true_class[-1*(i+1)]))