import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

from prettytable import PrettyTable

from keras import optimizers

import collections

from keras.preprocessing.image import ImageDataGenerator, img_to_array

import os
curdir = os.getcwd()



label_paths = [

    "../input/data-files/y_train_smpl_0.csv",

    "../input/data-files/y_train_smpl_1.csv",

    "../input/data-files/y_train_smpl_2.csv",

    "../input/data-files/y_train_smpl_3.csv",

    "../input/data-files/y_train_smpl_4.csv",

    "../input/data-files/y_train_smpl_5.csv",

    "../input/data-files/y_train_smpl_6.csv",

    "../inputdata-files/y_train_smpl_7.csv",

    "../input/data-files/y_train_smpl_8.csv",

    "../input/data-files/y_train_smpl_9.csv",   

]
# Function we can use to load labels

def get_label (index=0, paths_array=label_paths):

    return pd.read_csv(paths_array [index])



# Load the features

data_train_all = pd.read_csv("../input/data-files/x_train_gr_smpl.csv")
from random import randint



# Use only a sample of the data to increase speed.

# Add a seed so that it returns the same numbers every time

random_int = randint(0, 1000)

def reduced_dataset(dataframe, size = 0.1):

    num_data = int(dataframe.shape[0]*size)

    np.random.seed(random_int)

    idx = np.arange(dataframe.shape[0])

    np.random.shuffle(idx)

    train_idx = idx[:num_data]

    

    return pd.DataFrame(dataframe.loc[train_idx].values)



data_train = reduced_dataset(data_train_all)
from sklearn.utils.random import sample_without_replacement



def sample_indices(test_size=0.25):

    n_samples = data_train.shape[0]

    train_size = round((1-test_size)*n_samples)

    test_size = n_samples - train_size

    

    all_indices = list(range(n_samples))

    train_indices = sample_without_replacement(n_population=data_train.shape[0], n_samples=train_size)

    test_indices = [x for x in all_indices if x not in train_indices]

    

    return train_indices, test_indices



test_size=0.25

train_idx, test_idx = sample_indices(test_size=test_size)
from keras.preprocessing import image  

from PIL import Image





# Convert the image paths to arrays

def list_to_tensor(img_array):

    # loads images as PIL

    np_image = np.asarray(img_array)

    x = np_image.reshape((48,48))

    # add first dimension

    x = np.expand_dims(x, axis=0)

    # add last dimension of 3 by copying the image

    x = np.repeat(x[..., np.newaxis], 3, -1)

    #x = np.expand_dims(x, axis=3)

    # convert 3D tensor to 4D tensor with shape (1, 48, 48, 1) and return 4D tensor

    return x



def lists_to_tensor(data):

    list_of_tensors = [list_to_tensor(data.loc[i]) for i in range(data.shape[0])]

    return np.vstack(list_of_tensors)
from keras.preprocessing.image import ImageDataGenerator



batch_size = 32



# Transforms

datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=50,

    width_shift_range=0.1,  # randomly shift images horizontally 

    height_shift_range=0.1,  # randomly shift images vertically

    horizontal_flip=True,

    fill_mode='nearest')



datagen_test = ImageDataGenerator(

    rescale=1./255)
# Lets have a look at some of our images

labels_0 = reduced_dataset(get_label(0))

tensor_data = lists_to_tensor(data_train)

toy_generator = datagen.flow(tensor_data, labels_0, batch_size=32, shuffle=True)

images, labels = toy_generator.next()



indices_2_class = {1: 'Not speed limit 60', 0: 'Speed limit 60'}





fig = plt.figure(figsize=(20,10))

fig.subplots_adjust(wspace=0.2, hspace=0.4)



# Lets show the first 32 images of a batch

for i, img in enumerate(images[:32]):

    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])

    ax.imshow(img.squeeze())

    image_idx = labels[i][0]

    ax.set(title=indices_2_class[image_idx])



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

from keras.applications.resnet50 import ResNet50

from keras.models import Model

"""

model = Sequential()



### TODO: Define your architecture.

model.add(Conv2D(filters=16, kernel_size=2, padding='same',

                  input_shape=(48,48,3), kernel_initializer='he_normal'))



model.add(Conv2D(32, (3,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(64, (3,3)))

model.add(Conv2D(128, (3,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(256, (3,3)))

model.add(MaxPooling2D(pool_size=2))



model.add(Flatten())

model.add(Dense(1000))

model.add(Activation('elu'))

model.add(Dropout(0.5))

model.add(Dense(500))

model.add(Activation('elu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))

"""
def create_model():

    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(48,48,3), pooling=max)



    x = Flatten()(base_model.output)

    x = Dense(1000, activation='elu')(x)

    x = Dropout(0.8)(x)

    predictions = Dense(1, activation='sigmoid')(x)



    # this is the model we will train

    model = Model(inputs=base_model.input, outputs=predictions)



    # Freeze/Unfreeze the parameters of the pre-trained model

    for layer in base_model.layers:

        layer.trainable = True

        

    # compile the model (should be done *after* setting layers to non-trainable)

    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy',

             metrics=['accuracy'])

        

    return model
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1,

                              patience=5, min_lr=1e-8, verbose=1)



early_stopper = EarlyStopping(monitor='val_loss', patience=8,

                              verbose=1, restore_best_weights=True)

# Sets a threshold to the data and rounds it to 0 or 1. Useful for some metrics.

def set_threshold(data, threshold = 0.5):

    rounded = np.array([1 if x >= threshold else 0 for x in data])

    return rounded





def run_metrics(y_true, y_pred, threshold):

    y_pred_rounded = set_threshold(y_pred, threshold)



    roc_score = round(roc_auc_score(y_true, y_pred), 4)

    f_score = round(f1_score(y_true, y_pred_rounded), 4)

    recall = round(recall_score(y_true, y_pred_rounded), 4)

    precision = round(precision_score(y_true, y_pred_rounded), 4)

    accuracy = round(accuracy_score(y_true, y_pred_rounded), 4)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_rounded).ravel()

    

    

    return {

        "roc_score": roc_score,

        "f_score": f_score,

        "recall": recall,

        "precision": precision,

        "accuracy": accuracy,

        "TP": tp,

        "FP": fp}



def run_metrics_all(predictions_all, threshold = 0.5, labels_idx=[]):

    x = PrettyTable()

    x.field_names = ["", "Roc_AUC_score", "f score", "recall", "precision", "accuracy", "TP", "FP"]

    for i, (columnName, y_pred) in enumerate(predictions_all.iteritems()):

        y_true = get_label(index=i, paths_array=label_paths)

        

        if len(labels_idx) != 0:

            y_true = y_true.loc[labels_idx]

            

        metrics = run_metrics(y_true, y_pred, threshold=threshold)

        x.add_row([columnName, metrics["roc_score"], metrics["f_score"],

                   metrics["recall"], metrics["precision"], metrics["accuracy"], metrics["TP"], metrics["FP"]])

        

    print(x)
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

seed = np.random.seed(0)



num_train = int(data_train.shape[0]*(1-test_size))

num_test = data_train.shape[0] - num_train



def train_net(train_idx, test_idx, label_paths=label_paths):

    results = pd.DataFrame()



    for i in range(6, len(label_paths)):

        print("\nRUNNING NET FOR y_train_smpl_{}".format(i))

        

        # Saves the best model

        checkpointer = ModelCheckpoint(filepath='weights.best_labels{}.hdf5'.format(i), 

                               verbose=1, save_best_only=True)

        

        target = reduced_dataset(pd.DataFrame(get_label(index=i, paths_array=label_paths)))

        #target = to_categorical(target, 2)

        

        data_training_x, data_training_y = tensor_data[train_idx], target.values[train_idx]

        data_test_x, data_test_y = tensor_data[test_idx], target.values[test_idx]

        

        train_generator = datagen.flow(data_training_x, data_training_y, batch_size=batch_size, shuffle=True)

        test_generator = datagen_test.flow(data_test_x, data_test_y, batch_size=1, shuffle=False)

        

        model = create_model()

        

        model.fit_generator(train_generator,

                   steps_per_epoch=num_train//batch_size,

                   epochs=15,

                   verbose=1,

                   callbacks=[scheduler, early_stopper, checkpointer],

                   validation_data=test_generator,

                   validation_steps=num_test//batch_size)

        

        # Load the best model

        #model.load_weights('weights.best_labels{}.hdf5'.format(i))

        # Predict on Val data

        predictions = model.predict_generator(test_generator)

        results["y_train_smpl_{}".format(i)] = predictions.flatten()

        print(run_metrics(data_test_y, predictions, 0.5))

    return results

        

predictions = train_net(train_idx, test_idx)

predictions.head()