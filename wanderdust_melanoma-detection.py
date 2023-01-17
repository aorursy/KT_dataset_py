import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from IPython.display import display # Allows the use of display() for DataFrames

from time import time

import matplotlib.pyplot as plt

import seaborn as sns # Plotting library

import keras

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator, img_to_array

from keras.utils import np_utils

from sklearn.datasets import load_files   

from tqdm import tqdm

from collections import Counter





print(os.listdir("../input"))
data_train_path = '../input/skin-lesion-analysis-towards-melanoma-detection/train/train'

data_valid_path = '../input/skin-lesion-analysis-towards-melanoma-detection/valid/valid'

data_test_path = '../input/skin-lesion-analysis-towards-melanoma-detection/test/test'
# define function to load train, test, and validation datasets

def load_data_raw (path):

    data = load_files(path)

    files = np.array(data['filenames'])

    targets = np_utils.to_categorical(np.array(data['target']), 3)

    

    return files, targets



train_filenames, train_targets = load_data_raw(data_train_path)
filenames_trimmed = [filename.split('/')[-2] for filename in train_filenames]

classes_count = Counter(filenames_trimmed)



# Plot the classes

plt.bar(classes_count.keys(), classes_count.values(), color=['blue', 'orange', 'green'])
def plot_n_samples(filenames):

    filenames_trimmed = [filename.split('/')[-2] for filename in filenames]

    classes_count = Counter(filenames_trimmed)



    # Plot the classes

    plt.bar(classes_count.keys(), classes_count.values(), color=['blue', 'orange', 'green'])
from sklearn.utils import resample, shuffle



# Choose one of the 3 for the feature_name

feature_names = {0: 'melanoma', 1: 'nevus', 2: 'seborrheic_keratosis'}



def upsample(filenames, targets, feature_name, n_samples = 1372):

    upsample_idx = []

    



    # Find all the indices for nevus

    for i, path in enumerate(filenames):

        # If feature matches, save the index

        if feature_name in path.split('/'):

            upsample_idx.append(i)

    

    # Remove selected features from filenames to add the upsampled after

    new_filenames = [filename for i, filename in enumerate(filenames) if i not in upsample_idx]

    new_targets = [target for i, target in enumerate(targets) if i not in upsample_idx]



    # Upsample

    resampled_x, resampled_y = resample(filenames[upsample_idx], targets[upsample_idx], n_samples=n_samples, random_state=0)



    # Add the upsampled features to new_filenames and new_targets

    new_filenames += list(resampled_x)

    new_targets += list(resampled_y) 

    

    return np.array(new_filenames), np.array(new_targets)

    

# We upsample twice: once for each feature we want upsampled

upsample_train_x, upsample_train_y = upsample(train_filenames, train_targets, feature_names[0])

upsample_train_x, upsample_train_y = upsample(upsample_train_x, upsample_train_y, feature_names[2])



plot_n_samples(upsample_train_x)
'''

# Use only if not using the up-sampling function

def downsample(filenames, targets, n_samples = 370):

    nevus_idx = []

    

    # Find all the indices for nevus

    for i, path in enumerate(filenames):

        # If nevus, save the index

        if 'nevus' in path.split('/'):

            nevus_idx.append(i)

    

    nevus_idx = np.sort(shuffle(nevus_idx)[n_samples:]) # shuffle indices



    # Downsample

    new_filenames = [filename for i, filename in enumerate(filenames) if i not in nevus_idx]

    new_targets = [target for i, target in enumerate(targets) if i not in nevus_idx]

    

    

    return new_filenames, new_targets

            

downsample_train_x, downsample_train_y = downsample(train_filenames, train_targets)



plot_n_samples(downsample_train_x)

'''
from keras.preprocessing import image   



# Convert the image paths to tensors Manually

def path_to_tensor(img_path):

    # loads RGB image as PIL.Image.Image type

    img = image.load_img(img_path, target_size=(224,224))

    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)

    x = image.img_to_array(img)

    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor

    return np.expand_dims(x, axis=0)



def paths_to_tensor(img_paths):

    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]

    return np.vstack(list_of_tensors)





train_filenames = paths_to_tensor(upsample_train_x)

train_targets = upsample_train_y
batch_size=60



# Transforms

datagen_train = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.1,  # randomly shift images horizontally 

    height_shift_range=0.1,  # randomly shift images vertically

    horizontal_flip=True)



datagen_valid = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.1,  # randomly shift images horizontally

    height_shift_range=0.1,  # randomly shift images vertically

    horizontal_flip=True)



datagen_test = ImageDataGenerator(

    rescale=1./255)
# Generators

'''

train_generator = datagen_train.flow_from_directory(

        data_train_path,

        target_size=(224, 224),

        batch_size=batch_size,

        class_mode='categorical')

'''



train_generator = datagen_train.flow(train_filenames, train_targets, batch_size=batch_size)



valid_generator = datagen_valid.flow_from_directory(

        data_valid_path,

        target_size=(224, 224),

        batch_size=batch_size,

        class_mode='categorical',

        shuffle=False)



test_generator = datagen_test.flow_from_directory(

        data_test_path,

        target_size=(224, 224),

        batch_size=1,

        class_mode='categorical',

        shuffle=False)
num_train = len(train_filenames)

num_valid = len(valid_generator.filenames)

num_test = len(test_generator.filenames)



print(num_train, num_valid, num_test)
# Class name to the index

#class_2_indices = train_generator.class_indices

class_2_indices = {'melanoma': 0, 'nevus': 1, 'seborrheic_keratoses': 2}

print("Class to index:", class_2_indices)



# Reverse dict with the class index to the class name

indices_2_class = {v: k for k, v in class_2_indices.items()}

print("Index to class:", indices_2_class)
# Lets have a look at some of our images

images, labels = train_generator.next()



fig = plt.figure(figsize=(20,10))

fig.subplots_adjust(wspace=0.2, hspace=0.4)



# Lets show the first 32 images of a batch

for i, img in enumerate(images[:32]):

    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])

    ax.imshow(img)

    image_idx = np.argmax(labels[i])

    ax.set(title=indices_2_class[image_idx])
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D

from keras.applications import ResNet50

from keras.models import Model

from keras_tqdm import TQDMNotebookCallback



base_model = ResNet50(weights='imagenet', include_top=False)



# add a global spatial average pooling layer

x = base_model.output

x = GlobalAveragePooling2D()(x)

# x = MaxAveragePooling2D()(x)

# let's add a fully-connected layer

x = Dense(1024, activation='elu')(x)

x = Dropout(0.95)(x)

# and a logistic layer

predictions = Dense(3, activation='softmax')(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)



# first: train only the top layers (which were randomly initialized)

for layer in base_model.layers:

    layer.trainable = True



#model.summary()
'''

model = Sequential()



### TODO: Define your architecture.

model.add(Conv2D(filters=16, kernel_size=2, padding='same',

                  input_shape=(224,224,3), kernel_initializer='he_normal'))



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

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(500))

model.add(Activation('elu'))

model.add(Dropout(0.5))

model.add(Dense(3))

model.add(BatchNormalization())

model.add(Activation('softmax'))



          

model.summary()

'''
from keras.optimizers import Adam



# compile the model (should be done *after* setting layers to non-trainable)

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',

             metrics=['accuracy'])
from sklearn.utils import class_weight



# Convert one hot encoded labels to ints

train_targets_classes = [np.argmax(label) for label in train_targets]



# Compute the weights

class_weights = class_weight.compute_class_weight('balanced',

                                                  np.unique(train_targets_classes),

                                                  train_targets_classes)



class_weights_dict = dict(enumerate(class_weights))

print(class_weights_dict)
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



# train the model

checkpointer = ModelCheckpoint(filepath='aug_model.weights.best.hdf5', verbose=1, 

                               save_best_only=True)



scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1,

                              patience=5, min_lr=1e-8, verbose=1)



early_stopper = EarlyStopping(monitor='val_loss', patience=10,

                              verbose=0, restore_best_weights=True)



history = model.fit_generator(train_generator,

                    class_weight= class_weights_dict,

                    steps_per_epoch=num_train//batch_size,

                    epochs=40,

                    verbose=0,

                    callbacks=[checkpointer, scheduler, TQDMNotebookCallback(), early_stopper],

                    validation_data=valid_generator,

                    validation_steps=num_valid//batch_size)
# load the weights that yielded the best validation accuracy

model.load_weights('aug_model.weights.best.hdf5')
score = model.evaluate_generator(test_generator, steps=num_test//1, verbose=1)

print('\n', 'Test accuracy:', score[1])
predictions = model.predict_generator(test_generator, steps=num_test)



task_1 = pd.DataFrame(data=[desease[0] for desease in predictions])

task_2 = pd.DataFrame(data=[desease[2] for desease in predictions])
from sklearn.metrics import roc_auc_score, accuracy_score



ground_truth = pd.read_csv("../input/udacitydermatologistai/repository/udacity-dermatologist-ai-2ec0ca9/ground_truth.csv")

labels = np_utils.to_categorical(np.array(test_generator.classes), 3)



roc_auc_all = roc_auc_score(labels, predictions)

roc_auc_task_1 = roc_auc_score(ground_truth['task_1'], task_1)

roc_auc_task_2 = roc_auc_score(ground_truth['task_2'], task_2)



print('Roc auc score for all data is: {}'.format(roc_auc_all))

print('Roc auc score for task 1 is: {}'.format(roc_auc_task_1))

print('Roc auc score for task 2 is: {}'.format(roc_auc_task_2))
test_filenames, test_targets = load_data_raw(data_test_path)
def plot_prediction(img_file, img_target):



    img = image.load_img(img_file, target_size=(224,224))

    img = image.img_to_array(img)/255

    img_expand = np.expand_dims(img, axis=0)

    

    # Make a prediction

    prediction = model.predict(img_expand, steps=1)

    image_idx = np.argmax(prediction[0])

    prediction_string = indices_2_class[image_idx]

    

    # Get the real label's name

    label_idx = np.argmax(img_target)

    real_label = indices_2_class[label_idx]

    

    # Plot predictions

    title = "Prediction: {}\nReal: {}".format(prediction_string, real_label)

    

    plt.imshow(img)

    plt.title(title)

    

    pred_df = pd.DataFrame({'Cancer type':['melanoma', 'nevus', 'seborrheic keratosis'], 'val':prediction[0]})

    ax = pred_df.plot.barh(x='Cancer type', y='val', title="Predictions", grid=True)

    

random_index = np.random.randint(0, len(test_generator.filenames))

plot_prediction(test_filenames[random_index], test_targets[random_index])
plts, (ax1, ax2) = plt.subplots(1,2, figsize=(20,5))



# summarize history for accuracy

ax1.plot(history.history['acc'])

ax1.plot(history.history['val_acc'])

ax1.set_title('model accuracy')

ax1.set(xlabel='epoch', ylabel='accuracy')

ax1.legend(['train', 'val'], loc='upper left')



ax2.plot(history.history['loss'])

ax2.plot(history.history['val_loss'])

ax2.set_title('model loss')

ax2.set(xlabel='epoch', ylabel='loss')

ax2.legend(['train', 'val'], loc='upper left')
# submission

submission = pd.read_csv("../input/udacitydermatologistai/repository/udacity-dermatologist-ai-2ec0ca9/sample_predictions.csv")

submission['task_1'] = task_1

submission['task_2'] = task_2

submission.to_csv("submission_dermatologist.csv", index=False)

display(submission.head())
from sklearn.metrics import confusion_matrix



# Confusion matrix for all classes

y_true = test_generator.classes

y_pred = [np.argmax(x) for x in predictions]



labels = ["melanoma", "nevus", "keratoses"]

cm = confusion_matrix(y_true, y_pred)

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize confusion matrix

ax = sns.heatmap(cm, annot=True)

ax.xaxis.set_ticklabels(labels)

ax.yaxis.set_ticklabels(labels)
def plot_distribution(pred_target_y, filenames):

    melanoma_idx = []

      

    # Find all the indices for nevus

    for i, path in enumerate(filenames):

        # If feature matches, save the index

        if 'melanoma' in path.split('/'):

            melanoma_idx.append(i)

            

    bening_preds = [pred for i, pred in enumerate(pred_target_y) if i not in melanoma_idx]

    malignant_preds = [pred for i, pred in enumerate(pred_target_y) if i in melanoma_idx]

    

    fig, ax = plt.subplots(1,1,figsize=(15,6))

    

    ax.set_title('Malignant vs. Bening')

    sns.distplot(bening_preds, hist=True, kde=True, label="Benign", bins=35)

    sns.distplot(malignant_preds, hist=True, kde=True, label="Malignant", bins=35, axlabel="Probability Malignant")

    ax.legend()

    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1))



plot_distribution(task_1.values, test_generator.filenames)