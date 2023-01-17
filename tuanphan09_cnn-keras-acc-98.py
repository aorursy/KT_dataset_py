import keras

import numpy as np

import os

import pickle

import cv2

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report



import tensorflow as tf

from keras.utils.np_utils import to_categorical

from keras.backend.tensorflow_backend import set_session

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from keras.optimizers import Adadelta, RMSprop

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.preprocessing.image import ImageDataGenerator
# Create CNN model

def CNN_model(input_shape, nb_classes):

    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                    activation ='relu', input_shape = input_shape))

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                    activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                    activation ='relu'))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                    activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(512, activation = "relu"))

    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, activation = "softmax"))

    return model
def load_data(data_dir="./data_43/train", DIM=(50, 50, 3), wrong_prediction='misclassified.csv', prob_thredhold=0.7):

    X = []

    y = []

    list_labels = [] 

    misclassified = {} # mapping misclassified training sample with new label

    if os.path.isfile(wrong_prediction):

        f = open(wrong_prediction, "r")

        for i, line in enumerate(f):

            if i > 0:

                imageId, label_pred, proba = line.split(',')

                if float(proba) >= prob_thredhold:

                # training sample was misclassified with high confidence

                    misclassified[imageId] = int(label_pred)

    relabel = 0

    for dir in os.listdir(data_dir):

        sub_dir = os.path.join(data_dir, dir)

        if os.path.isdir(sub_dir):

            list_labels.append(dir)

            idx_label = list_labels.index(dir)

            for file in os.listdir(sub_dir):

                if DIM[2] == 1: # load gray image (channel=1)

                    img = cv2.imread(os.path.join(sub_dir, file), 0)

                else: # load RGB image

                    img = cv2.imread(os.path.join(sub_dir, file))

                img = cv2.resize(img, (DIM[0], DIM[1]))

                X.append(img)

                if dir + "/" + file in misclassified: # change image's label

                    y.append(misclassified[dir + "/" + file])

                    relabel += 1

                else:

                    y.append(idx_label)

    print("Change label: {}|{} of {}".format(relabel, len(misclassified), len(y)))

    X = np.array(X, dtype=np.float32)

    X = np.reshape(X, (-1, *DIM))

    y =  np.array(y, dtype=int)

    y = to_categorical(y, len(list_labels))

    return X, y, list_labels
N_EPOCHS = 40

BATCH_SIZE = 128

DIM = (50, 50, 3)



X, y, list_labels = load_data(data_dir="./data_43/train", DIM=DIM, prob_thredhold=0.7)  

# Split dataset

X_train, X_val, y_train, y_val =  train_test_split(X, y, test_size=0.2, random_state=9)
# Data augmentation

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        brightness_range=(0.5, 1.3),

        rescale=1./255,

        rotation_range=12,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range=0.1, # Randomly zoom image 

        shear_range=0.1,

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False # randomly flip images

    )   
model = CNN_model(DIM, len(list_labels))

# model.load_weights("models/CNN/weights-32-0.89.hdf5")

optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])



model_file = "./models/weights-{epoch:02d}-{val_loss:.2f}.hdf5"

# Save model

checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=False)

tbCallBack = TensorBoard(log_dir='./tensorboard/{}'.format(model_type), write_graph=True, write_images=True)

# Reduce the LR by half if the loss is not decreased after 3 epochs.

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.000001)

callbacks_list = [checkpoint, tbCallBack, learning_rate_reduction]

# Training

model.fit_generator(

    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),

    epochs = N_EPOCHS, 

    validation_data = (X_val/255, y_val),

    verbose = 1, 

    steps_per_epoch=X_train.shape[0] // BATCH_SIZE, 

    callbacks=callbacks_list,

    # max_queue_size=20,

    # workers=5,

    # use_multiprocessing=True

)
# Load model and predict training data

def load_data(data_dir="./data_43/train", DIM=(50, 50, 3)):

    X = []

    y = []

    labels = []

    list_files = []



    for dir in os.listdir(data_dir):

        sub_dir = os.path.join(data_dir, dir)

        if os.path.isdir(sub_dir):

            labels.append(dir)

            idx_label = labels.index(dir)

            for file in os.listdir(sub_dir):

                if DIM[2] == 1:

                    img = cv2.imread(os.path.join(sub_dir, file), 0)

                else:

                    img = cv2.imread(os.path.join(sub_dir, file))

                img = cv2.resize(img, (DIM[0], DIM[1]))

                X.append(img)

                y.append(idx_label)

                list_files.append(dir + "/" + file)

                

    X = np.array(X, dtype=np.float32)

    X = np.reshape(X, (-1, *DIM))

    X /= 255

    y =  np.array(y, dtype=int)

    return X, y, labels, list_files





DIM = (50, 50, 3)

X, y, labels, list_files = load_data(DIM=DIM)

model_path = "./models/CNN/weights-40-0.40.hdf5" 

model = CNN_model(DIM, len(labels))

model.load_weights(model_path)

proba = model.predict(X, batch_size=64, verbose=1)

y_pred = np.argmax(proba, axis=1)



idxes, = np.where(y != y_pred)



f = open("misclassified.csv", "w")

f.write("ImageId,Label,Proba\n")

for i in idxes:

    f.write("{},{},{}\n".format(list_files[i], y_pred[i], proba[i, y_pred[i]]))

f.close()
# Submission

model_path = "./models/CNN/weights-40-0.03.hdf5"

model.load_weights(model_path)

print("Done loaded model!")

Y_pred = model.predict(X, batch_size=64, verbose=1)

y_pred = np.argmax(Y_pred, axis=1)



f = open("submit.csv", "w")

f.write("ImageId,Label\n")

for i in range(len(list_files)):

    f.write("{},{}\n".format(list_files[i], labels[y_pred[i]]))

f.close()