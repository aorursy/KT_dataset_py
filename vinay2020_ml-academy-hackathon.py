# !pip install kaggle
# !mkdir .kaggle
# import json
# token = {"username":"YOUR-USER-NAME","key":"SOME-VERY-LONG-STRING"}
# with open('/content/.kaggle/kaggle.json', 'w') as file:
#     json.dump(token, file)
# !cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json
# !kaggle config set -n path -v{/content}
# !chmod 600 /root/.kaggle/kaggle.json
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# import the necessary packages
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
import random
import cv2
import os
import pydicom
BASE_PATH = '../input/rsna-intracranial-hemorrhage-detection/'
TRAIN_IMAGES_FLDR = "stage_1_train_images"
TEST_IMAGES_FLDR = "stage_1_test_images"
def turn_input_to_dataframe(path):
    original_df = pd.read_csv(path)
    original_df['filename'] = original_df['ID'].apply(lambda x: "ID_" + x.split('_')[1] + ".dcm")
    original_df['type'] = original_df['ID'].apply(lambda x: x.split('_')[2])
    final_df = original_df[['Label', 'filename', 'type']].drop_duplicates().pivot(index='filename', columns='type', values='Label').reset_index()
    return final_df
train_data_df = turn_input_to_dataframe(BASE_PATH + 'stage_1_train.csv')
test_data_df = turn_input_to_dataframe(BASE_PATH + 'stage_1_sample_submission.csv')
train_data_df.head()
test_data_df.head()
def turn_pred_to_dataframe(data_df, pred):
    pref_df = pd.DataFrame(pred, columns=data_df.columns, index=data_df.index)
    pref_df = pref_df.stack().reset_index()
    pref_df.loc[:, "ID"] = pref_df.id.str.cat(df.subtype, sep="_")
    pref_df = pref_df.drop(["id", "subtype"], axis=1)
    submission_df = pref_df.rename({0: "Label"}, axis=1)
    return submission_df
turn_pred_to_dataframe(test_data_df)
# fix all problamtic images
def fix_pxrepr(dcm):
    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-100: return
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000
def windowed(dcm, w, l):
    px = dcm.clone()
    px_min = l - w//2
    px_max = l + w//2
    px[px<px_min] = px_min
    px[px>px_max] = px_max
    return (px-px_min) / (px_max-px_min)

#fix_pxrepr(dcm)
#windowed(dcm, w=80, l=40)
import kornia
def gauss_blur2d(x,s):
    s2 = int(s/4)*2+1
    x2 = unsqueeze(x, dim=0, n=4-x.dim())
    res = kornia.filters.gaussian_blur2d(x2, (s2,s2), (s,s), 'replicate')
    return res.squeeze()
#blurred = gauss_blur2d(dcm, 100)
def mask_from_blur(x:Tensor, window, sigma=0.3, thresh=0.05, remove_max=True):
    p = x.windowed(*window)
    if remove_max: p[p==1] = 0
    return gauss_blur2d(p, s=sigma*x.shape[-1])>thresh
def pad_square(x):
    r,c = x.shape
    d = (c-r)/2
    pl,pr,pt,pb = 0,0,0,0
    if d>0: pt,pd = int(math.floor( d)),int(math.ceil( d))        
    else:   pl,pr = int(math.floor(-d)),int(math.ceil(-d))
    return np.pad(x, ((pt,pb),(pl,pr)), 'minimum')

def crop_mask(x):
    mask = x.mask_from_blur(dicom_windows.brain)
    bb = mask2bbox(mask)
    if bb is None: return
    lo,hi = bb
    cropped = x.pixel_array[lo[0]:hi[0],lo[1]:hi[1]]
    x.pixel_array = pad_square(cropped)
def window_image(img, window_center,window_width, intercept, slope, rescale=True):

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    
    if rescale:
        img = (img - img_min) / (img_max - img_min)
    
    return img

def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
total_image_count = len(X_df)

filename = BASE_PATH + TRAIN_IMAGES_FLDR + '/' + X_df['filename'][random.randint(0, total_image_count - 1)]
data = pydicom.dcmread(filename)

window_center , window_width, intercept, slope = get_windowing(data)

#displaying the image
img = pydicom.read_file(filename).pixel_array


img = window_image(img, window_center, window_width, intercept, slope)

plt.imshow(img, cmap=plt.cm.bone)
plt.grid(False)
print(data)
from tensorflow.python.keras.utils.data_utils import Sequence

def read_dicom_image_resize(filename, width, height, channel, test=False):
    IMAGE_FLDR = TEST_IMAGES_FLDR if test else TRAIN_IMAGES_FLDR
    file_path = BASE_PATH + TRAIN_IMAGES_FLDR + '/' + filename
    data = pydicom.dcmread(file_path)
    temp_pixal_array = data.pixel_array
    window_center, window_width, intercept, slope = get_windowing(data)

    img = window_image(temp_pixal_array, 50, 100, intercept, slope)
    
    resized = resize(img, (width, height), anti_aliasing=True)
    return resized


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1, n_classes=10, shuffle=True, test=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.test = test
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #list_IDs_temp, list_label_temp = [], []
#         for k in indexes:
#             #print("{} index X {}, y {}".format(k, self.list_IDs['filename'][k], list(self.labels.iloc[k])))
#             list_IDs_temp.append(self.list_IDs['filename'][k])
#             list_label_temp.append(list(self.labels.iloc[k]))
        list_IDs_temp = [self.list_IDs['filename'][k] for k in indexes]
        list_label_temp=[[int(self.list_IDs['any'][i]),int(self.list_IDs['epidural'][i]),int(self.list_IDs['intraparenchymal'][i]),int(self.list_IDs['intraventricular'][i]),int(self.list_IDs['subarachnoid'][i]),int(self.list_IDs['subdural'][i])] for i in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp,list_label_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, list_label_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y = []
        # Generate data
        for i, filename in enumerate(list_IDs_temp):
            # Store sample
            img = read_dicom_image_resize(filename, self.dim[0], self.dim[1], self.dim[2], self.test)
            X.append(img)
            
            
            
        X=np.array(X).reshape(-1,self.dim[0],self.dim[1], self.dim[2])
        # Store class
        y = np.asarray(list_label_temp)

        return X, y
from sklearn.model_selection import train_test_split
#select 100000 images from the dataset

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing

# (trainX, testX, trainY, testY) = train_test_split(sampleX,
#     sampleY, test_size=0.30, stratify=sampleY, random_state=42)
# trainX = X_df[0:100000]
# trainY = y_df[0:100000]
# validX = X_df[100000:120000]
# validY = y_df[100000:120000]
# trainX.reset_index(inplace=True, drop=True)
# testX.reset_index(inplace=True, drop=True)
# trainY.reset_index(inplace=True, drop=True)
# testY.reset_index(inplace=True, drop=True)
#trainX.head()
#trainY.head()
#pd.DataFrame({"training": [len(trainX)], "validation": [len(validX)]}, index=["count"])#
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

def get_model(input_dim):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=input_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    return model

import tensorflow as tf
from tensorflow.python.ops import array_ops

def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)
epochs = 2
#INIT_LR = 1e-3
batch_size = 100
IMAGE_SIZE = 200

loss_function = focal_loss
# Parameters
params = {'dim': (IMAGE_SIZE, IMAGE_SIZE,1),
          'batch_size': batch_size,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}

validSamples=labels_df[100000:120000]
validSamples=validSamples.reset_index(drop=True)

trainDataGenerator = DataGenerator(labels_df[0:100000], **params)
validDataGenerator = DataGenerator(validSamples, **params)
get_model((IMAGE_SIZE,IMAGE_SIZE,1)).summary()
from keras.callbacks import ModelCheckpoint  
from keras_tqdm import TQDMNotebookCallback

model = get_model((IMAGE_SIZE,IMAGE_SIZE,1))
#opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)
model.compile(loss=loss_function, optimizer="adam", metrics=["accuracy"])

checkpointer = ModelCheckpoint(filepath='weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

history = model.fit_generator(
    generator=trainDataGenerator, 
    validation_data=validDataGenerator,
    use_multiprocessing=True, workers=-1,
    epochs=epochs, 
    callbacks=[checkpointer, TQDMNotebookCallback()], verbose=1)
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
model.load_weights('weights.best.from_scratch.hdf5')
#from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
#y_actual = np.asarray(testY)
#y_pred = model.predict_generator(testDataGenerator)
# y_pred.argmax(axis=1)
# confusion_matrix(y_actual, y_pred)
# accuracy_score(y_actual, y_pred)
# recall_score(y_actual, y_pred)
# precision_score(y_actual, y_pred)
# f1_score(y_actual, y_pred)
BASE_PATH = '../input/rsna-intracranial-hemorrhage-detection/'
TRAIN_IMAGES_FLDR = "stage_1_train_images"
TEST_IMAGES_FLDR = "stage_1_test_images"
test_data = pd.read_csv(BASE_PATH + "stage_1_sample_submission.csv")
test_data['filename'] = test_data['ID'].apply(lambda x: "ID_" + x.split('_')[1] + ".dcm")
test_data['type'] = test_data['ID'].apply(lambda x: x.split('_')[2])
test_data.head()
test_data = test_data[['Label', 'filename', 'type']].drop_duplicates().pivot(
    index='filename', columns='type', values='Label').reset_index()
test_data_filename = test_data['filename']
total_test_images = len(test_data_filename)
submission_df = pd.DataFrame({"ID":[], "Label":[]})
X_test = []
for i, filename in tqdm(enumerate(test_data_filename[:1000])):
    # Store sample
    ds=pydicom.dcmread(BASE_PATH + TEST_IMAGES_FLDR + '/' + filename)
    temp=ds.pixel_array
    window_center , window_width, intercept, slope = get_windowing(ds)
    img = window_image(temp, 50, 100, intercept, slope)
    resized = cv2.resize(img, (200, 200))
    X_test.append(resized)       
X_test=np.array(X_test).reshape(-1,200,200,1)
y_pred = model.predict(X_test)
y_pred_max = np.zeros_like(y_pred)
y_pred_max[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
for i, filename in tqdm(enumerate(test_data_filename[:1000])):
    prediction = y_pred_max[i]
    fileID = filename.split(".")[0]
    listOfSeries = [
        pd.Series([fileID+"_any", prediction[0]], index=submission_df.columns),
        pd.Series([fileID+"_epidural", prediction[1]], index=submission_df.columns),
        pd.Series([fileID+"_intraparenchymal", prediction[2]], index=submission_df.columns),
        pd.Series([fileID+"_intraventricular", prediction[3]], index=submission_df.columns),
        pd.Series([fileID+"_subarachnoid", prediction[4]], index=submission_df.columns),
        pd.Series([fileID+"_subdural", prediction[5]], index=submission_df.columns),
    ]
    submission_df = submission_df.append(listOfSeries, ignore_index=True)   
with open("stage_1_submission.csv", "w") as f:
    submission_df.to_csv(f, index=False)