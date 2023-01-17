import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D,MaxPooling2D, Flatten, Dense,Dropout
from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from keras.utils import plot_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix


'''for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''
df = pd.read_csv('/kaggle/input/documets/df.csv') #[:500]
PATH_TRAIN = '/kaggle/input/documets/JPEG/'
def create_one_represent_class(df_param):
    
    v_c_df = df_param['count_target'].value_counts().reset_index()
    one_represent = v_c_df.loc[v_c_df['count_target'] == 1, 'index'].tolist()
    df_param.loc[df_param['count_target'].isin(one_represent), 'count_target'] = 100
    return df_param

def df_split(df):
    
    count_target = df.groupby(by=['tender']).target.sum().reset_index().rename(columns={'target':'count_target'})
    count_target = create_one_represent_class(count_target)
    
    img_train, img_val  = train_test_split(count_target, test_size=0.2, random_state=42, stratify=count_target['count_target'])
    
    train_df = df[df['tender'].isin(img_train['tender'])].reset_index(drop=True)
    test_df = df[df['tender'].isin(img_val['tender'])].reset_index(drop=True)
    
    return train_df, test_df
train_df, test_df  = df_split(df)
print(train_df.shape[0], test_df.shape[0])
class Data_generator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(224,224), n_channels=3,
                 n_classes=5, shuffle=True):
        
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp):
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data

        for i, ID in enumerate(list_IDs_temp):
            X[i,] = (cv2.resize( cv2.imread(PATH_TRAIN+ID), dim ))

            # Store class
            #y[i] = self.labels[ID]
            y[i] = df[df.id == ID].target
        
        return X/255, y #keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    
    def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch
            
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
          
            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]
          
            # Generate data
            X, y = self.__data_generation(list_IDs_temp)
          
            return X, y
def create_model():
    vgg_model = keras.applications.VGG16(weights='imagenet',
                                   include_top=True)

    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

    x = layer_dict['fc2'].output
    x = Dense(5, activation='softmax')(x)

    custom_model = Model(vgg_model.input, x)

    # Make sure that the pre-trained bottom layers are not trainable
    for layer in custom_model.layers[:-3]:
        layer.trainable = False

    custom_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, amsgrad=False),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return custom_model
dim = (224,224)
EPOCHS = 10
batch_size = 64
n_classes = 5
n_channels = 3
shuffle = True
EARLY_STOPPING = 3

training_generator = Data_generator( train_df.id, train_df.target, batch_size=batch_size, dim=dim, n_channels=n_channels, n_classes=n_classes, shuffle=shuffle)
validation_generator = Data_generator(test_df.id, test_df.target, batch_size=batch_size, dim=dim, n_channels=n_channels, n_classes=n_classes, shuffle=shuffle )
custom_model = create_model()
ES = keras.callbacks.EarlyStopping(monitor='accuracy', patience=EARLY_STOPPING, verbose=False, mode='auto', restore_best_weights=True)

Check = keras.callbacks.ModelCheckpoint(  'model.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch' )
history = custom_model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                   epochs=EPOCHS,
                    callbacks = [ES, Check],
                    workers=-1)
print(train_df.shape[0], test_df.shape[0])
history.history
