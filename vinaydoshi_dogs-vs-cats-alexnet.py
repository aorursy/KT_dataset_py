# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import imutils

#from imutils import paths

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.image import extract_patches_2d

import progressbar

import tqdm

from tqdm import tqdm_notebook

import json

import csv

import cv2

import h5py

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input/Dogs_vs_Cats_Alexnet_Trained_Model'):

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
#/input/dogs-vs-cats-redux-kernels-edition/test/test/

#/input/dogs-vs-cats-redux-kernels-edition/train/train/
train_path = '../input/dogs-vs-cats-redux-kernels-edition/train/train/'

final_test_path = '../input/dogs-vs-cats-redux-kernels-edition/test/test/'

#train_img=[train_path+'/'+i for i in os.listdir(train_path)]

train_img_paths = list(list_images(train_path))

final_test_img_paths = list(list_images(final_test_path))
len(train_img_paths), len(final_test_img_paths)
# os.listdir(train_path),
NUM_CLASSES=2

NUM_VAL_IMAGES = 1250*NUM_CLASSES

NUM_TEST_IMAGES = 1250*NUM_CLASSES



train_hdf5 = '/kaggle/working/train.hdf5'

val_hdf5 = '/kaggle/working/val.hdf5'

test_hdf5 = '/kaggle/working/test.hdf5'

MODEL_PATH = '/kaggle/working/alexnet_dogs_vs_cats.model'

dataset_mean = '/kaggle/working/dogs_vs_cats_mean.json'

output_path = '/kaggle/working/'
import cv2



class SimplePreprocessor:

    def __init__(self, width, height, inter = cv2.INTER_AREA):

        # store the target image width, height, and interpolation

        # method used when resizing 

        self.width = width

        self.height = height

        self.inter = inter

    def preprocess(self, image):

         # resize the image to a fixed size, ignoring the aspect ratio

        return cv2.resize(image,(self.width, self.height), interpolation = self.inter)
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
class HDF5DatasetWriter:

    def __init__(self, dims, outputPath, dataKey = 'images', bufSize=1000):

        if os.path.exists(outputPath):

            raise ValueError('The supplied "outputPath" already exists. Manually delete the file before continuing.',outputPath)

            

        self.db = h5py.File(outputPath, mode='w')

        self.data = self.db.create_dataset(dataKey, dims, dtype='float')

        self.labels = self.db.create_dataset('labels', (dims[0],), dtype='int')

        

        self.bufSize = bufSize

        self.buffer = {'data':[], 'labels':[]}

        self.idx = 0

        

    def add(self, rows, labels):

        self.buffer['data'].extend(rows)

        self.buffer['labels'].extend(labels)

        

        if len(self.buffer['data']) >= self.bufSize:

            self.flush()

    

    def flush(self):

        i = self.idx + len(self.buffer['data'])

        self.data[self.idx:i] = self.buffer['data']

        self.labels[self.idx:i] = self.buffer['labels']

        self.idx = i

        

        self.buffer = {'data':[], 'labels':[]}

        

    def storeClassLabels(self, classLabels):

        

        dt = h5py.special_dtype(vlen=str)

        labelSet = self.db.create_dataset('label_name', (len(classLabels),), dtype = dt)

        labelSet[:] = classLabels

        

    def close(self):

        if len(self.buffer['data']) > 0:

            self.flush()

        

        self.db.close()
trainLabels = []

count_rej = 0

for p in train_img_paths:

    if p.split(os.path.sep)[-1].split('.')[-1] == 'jpg':

        trainLabels.append(p.split(os.path.sep)[-1].split('.')[0])

    else:

        count_rej +=1

print(len(trainLabels),'\n', np.unique(trainLabels))
le = LabelEncoder()

trainLabels = le.fit_transform(trainLabels)

print(len(trainLabels),'\n', np.unique(trainLabels))
sns.countplot(trainLabels)
train_img_paths, test_img_paths, y_train, y_test = train_test_split(train_img_paths, trainLabels,

                                                                    test_size = NUM_TEST_IMAGES, random_state=42,

                                                                    stratify=trainLabels

                                                                   )
train_img_paths, val_img_paths, y_train, y_val = train_test_split(train_img_paths, y_train,

                                                                    test_size = NUM_VAL_IMAGES, random_state=42,

                                                                    stratify=y_train

                                                                   )
train_dataset, val_dataset, test_dataset=[], [], []
datasets = [('train',train_img_paths, y_train, train_dataset)]

#             ('val',val_img_paths, y_train, val_hdf5),

#             ('test',test_img_paths, y_train, test_hdf5)

#            ]
aap = AspectAwarePreprocessor(227, 227)

#R, G, B = [], [], []
R, G, B = [], [], []
os.chdir('/kaggle/working/')
for dtype, paths, labels, outputPath in datasets:

    print(dtype)

#     writer = HDF5DatasetWriter((len(paths),256,256,3),outputPath)

    for i,(path,label) in tqdm.tqdm(enumerate(zip(paths, labels))):

        image = cv2.imread(path)

        image = aap.preprocess(image)

        

        if dtype == 'train':

            (b,g,r) = cv2.mean(image)[:3]

            R.append(r)

            G.append(g)

            B.append(b)

#             writer.add([image],[label])

#    writer.close()
B_mean = np.mean(B)

G_mean = np.mean(G)

R_mean = np.mean(R)

B_mean, G_mean, R_mean
class MeanPreprocessor:

    def __init__(self, rMean, gMean, bMean):

        self.rMean = rMean

        self.gMean = gMean

        self.bMean = bMean

    

    def preprocess(self, image):

        (B, G, R) = cv2.split(image.astype('float32'))

        R -= self.rMean

        G -= self.gMean

        B -= self.bMean

        return cv2.merge([B,G,R])
class PatchPreprocessor:

    def __init__(self, width, height):

        self.width = width

        self.height = height

    

    def preprocess(self, image):

        (h,w) = image.shape[:2]

        if h <= self.height:

            image = aap.preprocess(image)

        elif w <= self.width:

            image = aap.preprocess(image)

        else:

            image = image

        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]
class CropPreprocessor:

    def __init__(self, height, width, horiz=True, inter = cv2.INTER_AREA):

        self.width = width

        self.height = height

        self.horiz = horiz

        self.inter = inter

    def preprocess(self,image):

        crops = []

        (h,w) = image.shape[:2]

        coords = [[0,0, self.width, self.height],

                  [w-self.width, 0, w,self.height],

                  [w-self.width, h-self.height, w,h],

                  [0, h-self.height, self.width, h]

                 ]

        dW = int(0.5*(w-self.width))

        dH = int(0.5*(h-self.height))

        coords.append([dW, dH, w-dW, h-dH])

        

        for (startX, startY, endX, endY) in coords:

            #print(image.shape)

            crop = image[startY:endY, startX:endX]

            crop = cv2.resize(crop, (self.width, self.height), interpolation = self.inter)

            crops.append(crop)

        if self.horiz:

            mirrors = [cv2.flip(c,1) for c in crops]

            crops.extend(mirrors)

        return np.array(crops)
from keras.utils import np_utils

import numpy as np

import cv2



class HDF5DatasetGenerator:

    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None, binarize=True, classes=2):

        self.dbPath =dbPath

        self.batchSize =batchSize

        self.preprocessors =preprocessors

        self.aug =aug

        self.binarize =binarize

        self.classes =classes

        

        self.db = h5py.File(dbPath)

        self.numImages = self.db['labels'].shape[0]

        

    def generator(passes = np.inf):

        epochs=0

        if epochs < passes:

            for i in np.arange(0, self.numImages, self.batchSize):

                images = self.db['images'][i:i+self.batchSize]

                labels = se;f.db['labels'][i:i+self.batchSize]

                if self.binarize:

                    labels = np_utils.to_categorical(labels, self.classes)

                

                if self.preprocessors is not None:

                    procImages=[]

                    

                    for image in images:

                        for p in self.preprocessors:

                            image = p.preprocess(image)

                            procImages.append(image)

                            

                    images = np.array(procImages)

                

                if self.aug is not None:

                    (images,labels) = next(self.aug.flow(images, labels, batch_size = self.batchSize))

                    yield (images, labels)

                    

        epochs +=1

    

    def close(self):

        sef.db.close()
from keras.models import Sequential

from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense

from keras.regularizers import l2

from keras import backend as K



class Alexnet:

    def build (width, height, depth, classes, reg = 0.0002):

        model = Sequential()

        inputShape = (height, width, depth)

        chanDim = -1

        

        if K.image_data_format() == 'channels_first':

            inputShape = (depth, height, width)

            chanDim = 1

        

        model.add(Conv2D(96, (11,11), strides=(4,4), input_shape = inputShape, padding='same', kernel_regularizer=l2(reg)))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

        model.add(Dropout(0.25))

        

        model.add(Conv2D(256, (5,5), strides=(1,1), padding='same', kernel_regularizer=l2(reg)))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

        model.add(Dropout(0.25))

        

        model.add(Conv2D(384, (3,3), strides=(1,1), padding='same', kernel_regularizer=l2(reg)))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(384, (3,3), strides=(1,1), padding='same', kernel_regularizer=l2(reg)))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', kernel_regularizer=l2(reg)))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=chanDim))

        

        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

        model.add(Dropout(0.25))

        

        model.add(Flatten())

        model.add(Dense(4096, kernel_regularizer=l2(reg)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.5))

        

        model.add(Dense(4096, kernel_regularizer=l2(reg)))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.5))

        

        model.add(Dense(classes, activation='softmax', kernel_regularizer=l2(reg)))

        

        return model
from keras.preprocessing.image import img_to_array



class imageToArrayPreprocessor:

    def __init__(self, dataFormat=None):

        self.dataFormat = dataFormat

    

    def preprocess(self, image):

        return img_to_array(image, data_format = self.dataFormat)
from keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(rotation_range = 20, zoom_range=0.15, shear_range=0.15,

                         width_shift_range = 0.2, height_shift_range=0.2,

                         horizontal_flip=True, fill_mode='nearest'

                        )
sp = SimplePreprocessor(227, 227)

pp = PatchPreprocessor(227, 227)

mp = MeanPreprocessor(R_mean, G_mean, B_mean)

iap = imageToArrayPreprocessor()

# def image_data_generator(directory_list, labels, bs = 128, mode='train', aug=None):

#     i=0

#     #imagePaths

#     while True:

#         image_batch=[]

#         label_batch=[]

#         #labels=[]

#         for j in range(bs):

#             if i == len(directory_list):

#                 i=0

#             if mode=='train':

#                 imagePath = directory_list[i]

#             #print(imagePath)

#                 image = cv2.imread(imagePath)

            

#                 image = pp.preprocess(image)

#                 image = mp.preprocess(image)

#                 image = iap.preprocess(image)

#                 label = labels[i]

#                 image_batch.append(image)

#                 label_batch.append(label)

            

#                 i+=1

#             else:

#                 imagePath = directory_list[i]

#             #print(imagePath)

#                 image = cv2.imread(imagePath)

            

#                 image = sp.preprocess(image)

#                 image = mp.preprocess(image)

#                 image = iap.preprocess(image)

#                 label = labels[i]

#                 image_batch.append(image)

#                 label_batch.append(label)

            

#                 i+=1

#         if aug is not None:

#             (image_batch,label_batch) = next(aug.flow(np.array(image_batch),label_batch, batch_size=bs))

        

#         yield (np.array(image_batch),label_batch)

            

            

            
# def image_data_generator(directory_list, labels, bs = 5, mode='train', preprocessors=None, aug=None,classes=2):

#     i=0

#     #imagePaths

#     while True:

#         image_batch=[]

#         label_batch=[]

#         #labels=[]

#         for j in range(bs):

#             if i == len(directory_list):

#                 i=0

#             #if mode=='train':

#             imagePath = directory_list[i]

#             #print(imagePath)

#             image = cv2.imread(imagePath)

#             label = labels[i]

#             if preprocessors is not None:

#                 procImages=[]

#                 labelImages=[]

#                 for p in preprocessors:

#                     image = p.preprocess(image)

#                     procImages.append(image)

#                     labelImages.append(label)

#                 image=np.array(procImages)

#                 label=np.array(labelImages)

# #           image = pp.preprocess(image)

# #           image = mp.preprocess(image)

# #           image = iap.preprocess(image)

#             #label = labels[i]

#             #label_cat= np_utils.to_categorical(label,classes)

            

#             i+=1

#             print(np.array(image).shape, label)#, label_cat)

#             if aug is not None:

#                 (image, label) = next(aug.flow(np.array(image),label, batch_size=bs))

#             image_batch.append(image)

#             label_batch.append(label)

#         print('bat',np.array(image_batch).shape, label_batch) 

#         return (np.array(image_batch),label_batch)

            

            

# #            print(image.shape)

# #             image_batch.append(image)

# #             label_batch.append(label_cat)

# #             print('bat',np.array(image_batch).shape, label_batch) 
def image_data_generator(directory_list, labels, bs = 128, mode='train', binarize=True,preprocessors=None, aug=None,classes=2):

    while True:

        for i in range(0, labels.shape[0], bs):

            images=[]

            imagePaths = directory_list[i:i+bs]

            label_vals = labels[i:i+bs]

#            print(label_vals)

            if binarize:

                label_vals = np_utils.to_categorical(label_vals, classes)

#                print(label_vals.shape)

            if preprocessors is not None:

                procImages=[]

#                 labelImages=[]     

                for path in imagePaths:

                    image = cv2.imread(path)

                    #print(image.shape, path)

                    for p in preprocessors:

                        image = p.preprocess(image)

                    procImages.append(image)

#                        labelImages.append(label_vals[x])      

                images = np.array(procImages)

#                label_vals = np.array(labelImages)

        # print(images.shape, label_vals.shape)

            if aug is not None:

                (images,label_vals) = next(aug.flow(images, label_vals, batch_size = bs))

           

            yield (images, label_vals)
y_train[:8].shape[0]
train_img_paths[:4], y_train[:4]
# train_image_batch,train_label_batch = image_data_generator(train_img_paths[:8], y_train[:8], preprocessors=[pp,mp,iap],aug=aug)

# train_image_batch.shape, len(train_image_batch), train_label_batch

yield_chk=image_data_generator(train_img_paths[:8], y_train[:8], preprocessors=[pp,mp,iap],aug=aug)

yield_chk#.Generator()


from keras.models import load_model

model = load_model('/kaggle/input/dogs-vs-cats-alexnet-trained-model/alexnet_dogs_vs_cats_model_same_padd.hdf5')
# from keras.optimizers import Adam

# opt = Adam(lr=1e-3)

# model = Alexnet.build(width=227, height=227, classes=2, depth=3, reg=0.0002)

# model.compile(loss='binary_crossentropy', optimizer = opt, metrics=['accuracy'])

# model.summary()
#model.load_weights('/kaggle/input/dogs-vs-cats-alexnet-trained-model/alexnet_dogs_vs_cats_model_same_padd.hdf5')
train_gen = image_data_generator(train_img_paths, y_train, bs=64,preprocessors=[pp,mp,iap],aug=aug)

val_gen = image_data_generator(val_img_paths, y_val, bs=64,preprocessors=[sp,mp,iap])
y_train.shape[0]
# history = model.fit_generator(train_gen, steps_per_epoch=y_train.shape[0]//64, validation_data=val_gen,

#                     validation_steps=y_val.shape[0]//64, epochs=25, max_queue_size=64*2, verbose=1

#                    )
# model.save('alexnet_dogs_vs_cats_model_same_padd.hdf5')
# from IPython.display import FileLink

# FileLink('alexnet_dogs_vs_cats_model_same_padd.hdf5')
# img_try=cv2.imread('../input/dogs-vs-cats-redux-kernels-edition/train/dog.897.jpg')
#print([p.split(os.path.sep)[-1][:3] for p in train_img_paths])

#y_train
len(test_img_paths),y_test.shape[0]
def test_data_generator(directory_list, bs=128, mode='test', binarize=False,preprocessors=None, aug=None,classes=2, passes=np.inf):

    epochs=0

    if epochs<passes:

        for i in range(0, len(directory_list), bs):

            images=[]

            imagePaths = directory_list[i:i+bs]

#            label_vals = labels[i:i+bs]

#            print(label_vals)

            if binarize:

                label_vals = np_utils.to_categorical(label_vals, classes)

#                print(label_vals.shape)

            if preprocessors is not None:

                procImages=[]

#                 labelImages=[]     

                for path in imagePaths:

                    image = cv2.imread(path)

                    #print(image.shape, path)

                    for p in preprocessors:

                        image = p.preprocess(image)

                    procImages.append(image)

#                        labelImages.append(label_vals[x])      

                images = np.array(procImages)

#                label_vals = np.array(labelImages)

        # print(images.shape, label_vals.shape)

            if aug is not None:

                (images,label_vals) = next(aug.flow(images, label_vals, batch_size = bs))

           

            yield images

        epochs+=1
testgen = test_data_generator(test_img_paths, bs=64,preprocessors=[sp,mp,iap])
testgen, y_test.shape
predictions = model.predict_generator(testgen, steps = y_test.shape[0]//64, max_queue_size = 64*2, verbose=1)
def rank5_accuracy(preds, labels):

    rank1=0

    rank5=0

    

    for pred,label in zip(preds, labels):

        pred  = np.argsort(pred)[::-1]

        

        if label in pred[:5]:

            rank5 += 1

        

        if label == pred[0]:

            rank1 += 1

    

    rank5 /= float(len(labels))

    rank1 /= float(len(labels))

    

    return (rank1, rank5)
(rank1, _) = rank5_accuracy(predictions, y_test)

rank1
import pandas as pd

final_predict=[]

cp = CropPreprocessor(227,227)

aap2 = AspectAwarePreprocessor(256,256)

predictions2=[]
# import pyprind

# pbar = pyprind.ProgBar(y_test.shape[0])

# for i,images in enumerate(test_data_generator(test_img_paths, bs=128,preprocessors=[mp], passes=1)):

#     #print(i)

#     for image in images:

#         (h,w)=image.shape[:2]

#         if h <= 227:

#             image = aap2.preprocess(image)

#         elif w <= 227:

#             image = aap2.preprocess(image)

#         else:

#             image = image

#         crops = cp.preprocess(image)

#         crops = np.array([iap.preprocess(c) for c in crops])

#         pred = model.predict(crops)

#         predictions2.append(pred.mean(axis=0))

#        # print('predictions2',len(predictions2))

#     pbar.update(i)
# (rank1, _) = rank5_accuracy(predictions2, y_test)

# rank1
import pyprind

pbar = pyprind.ProgBar(y_test.shape[0])

for i,images in enumerate(test_data_generator(final_test_img_paths, bs=128,preprocessors=[mp], passes=1)):

    #print(i)

    for image in images:

        (h,w)=image.shape[:2]

        if h <= 227:

            image = aap2.preprocess(image)

        elif w <= 227:

            image = aap2.preprocess(image)

        else:

            image = image

        crops = cp.preprocess(image)

        crops = np.array([iap.preprocess(c) for c in crops])

        pred = model.predict(crops)

        final_predict.append(pred.mean(axis=0))

       # print('predictions2',len(predictions2))

    pbar.update(i)
len(final_predict)
val_outs=[]

for i in final_predict:

    val_outs.append(i[1])

    #print(i[1])
len(val_outs)
final_img_names=[i.split(os.path.sep)[-1].split('.jpg')[0] for i in final_test_img_paths]
submission = pd.DataFrame({'id':final_img_names, 'label':val_outs})

submission.head()
submission.to_csv('submission.csv',index=False)