# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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
#importing the nesce
import numpy as np
import itertools
import pandas as pd
import os
import math
import random
import cv2
import sys
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
train_dir = "../input/plant-seedlings-classification/train/"
test_dir = "../input/plant-seedlings-classification/test/"
save_dir = "/kaggle/working/plant-seedlings-classification/train"
target_size = (224, 224)
# Get names of all the categories 
categories = [category for category in sorted(os.listdir(train_dir))]

# Get the number of images in each cateogry
images_per_category = [len(os.listdir(os.path.join(train_dir, category))) for category in categories]

# Plot to see the distribution
plt.figure(figsize=(24,12))
sns.barplot(categories, images_per_category)

def preprocessing_pipeline(path, target_size):
    """Accepts a path and returns a processed image involving reading and resizing"""
    image = cv2.resize(cv2.imread(path), target_size, interpolation = cv2.INTER_NEAREST)
    return image

def show_sample_images(train_dir,data_og):
    categories = [category for category in sorted(os.listdir(train_dir))]
    random_indices = random.sample(range(0, len(data_og)), 4)
    
    # Plot some sample images from the dataset
    _, axs = plt.subplots(1, 4, figsize=(20, 20))
    for i in range(4):
        axs[i].imshow(data_og[random_indices[i]])

data_og = [preprocessing_pipeline(os.path.join(train_dir, category, img_path),target_size) for category in categories for img_path in os.listdir(os.path.join(train_dir, category))]
show_sample_images(train_dir,data_og)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imgaug import augmenters as iaa
from tqdm.notebook import tqdm
# Augment passed images
def augment_images(class_images):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(rotate=(-45, 45)),
        iaa.TranslateX(percent=(-0.1, 0.1)),
        iaa.TranslateY(percent=(-0.1, 0.1))
    ], random_order=True)

    images_aug = seq(images = class_images)
    return images_aug

# Helper Function 5
# Randomly sample images from a set of passed images
def random_unique_sampling(class_images, remainder):
    random_unique_indices = random.sample(range(0, len(class_images)), remainder)
    random_unique_images = [class_images[idx] for idx in random_unique_indices]
    return random_unique_images
    

def augmentation_pipeline(class_images, number_of_images):
    """Accepts a batch of images (of a single class) and returns a required number of augmented images"""

    if number_of_images == 0:
            return []

    elif number_of_images >= len(class_images):
        batches = math.floor(number_of_images / len(class_images))
        remainder = number_of_images % len(class_images)
        remainder_images = random_unique_sampling(class_images, remainder)
        class_images = class_images * batches
        class_images.extend(remainder_images)
        images_aug = augment_images(class_images)
        return images_aug

    else:
        assert number_of_images < len(class_images)
        class_images = random_unique_sampling(class_images, number_of_images)
        images_aug = augment_images(class_images)
        return images_aug


def balance_dataset(save_dir,train_dir):
    """Create augmented data to balance classes from the passed training data path"""

    # Make a directory for augmented dataset
    os.makedirs(save_dir, exist_ok=True)

    # Get categories
    categories = [category for category in sorted(os.listdir(train_dir))]

    # Get the maximum amount of images that exists in a class
    max_in_class = max([len(os.listdir(os.path.join(train_dir, category))) for category in categories])

    # Find out the number of images that exist in each class
    images_per_category = {category : len(os.listdir(os.path.join(train_dir, category))) for category in categories}

    # Find out the augmentations required for each class
    required_augmentations = dict(zip(categories,  [max_in_class - num_in_class for num_in_class in list(images_per_category.values())]))

    # Augment each unbalanced class and save the new dataset to disk
    # We preferring saving the data to disk
    # Because we prefer to not hold large numpy arrays in the RAM
    # This allows for large models to be loaded and trained on
    # We use for loops here instead of list comprehensions for readiblity
    for category in tqdm(categories):
        try:
            os.mkdir(os.path.join(save_dir, category))
        except FileExistsError:
            pass
        class_images = list()

        # Preprocessing and Augmentation
        for img_path in sorted(os.listdir(os.path.join(train_dir, category))):
            image = preprocessing_pipeline(os.path.join(train_dir, category, img_path),target_size)
            class_images.append(image)
        augmented_images = augmentation_pipeline(class_images, required_augmentations[category])
        class_images.extend(augmented_images)

        # Writing the augmented data to disk
        for image_number, class_image in enumerate(class_images):
            cv2.imwrite(os.path.join(save_dir, category, "{}.png".format(image_number + 1)), class_image)

balance_dataset(save_dir,train_dir)


#InceptionV3 

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import inception_v3

datagen = ImageDataGenerator(preprocessing_function = inception_v3.preprocess_input, validation_split=0.15)
target_size = (299, 299)

train_generator = datagen.flow_from_directory(
        directory= os.path.join(save_dir),
        target_size= target_size,
        class_mode = "categorical",
        batch_size=32,
        shuffle=True,
        subset='training'
    )

val_generator = datagen.flow_from_directory(
        directory= os.path.join(save_dir),
        target_size= target_size,
        class_mode = 'categorical',
        batch_size=32,
        shuffle=False,
        subset='validation'
    )


#Callbacks

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

model_save_path = '/kaggle/working/model_inceptionv3.h5'
checkpoint = ModelCheckpoint(filepath='/kaggle/working/model_inceptionv3.h5', monitor='val_loss', mode='min', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.00000001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min', restore_best_weights=True)
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU, Flatten, Activation
from tensorflow.keras.activations import swish
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

Inception_base = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(299, 299, 3))
for layer in Inception_base.layers[:-22]:
    layer.trainable = False
x = Flatten() (Inception_base.output)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
x = Dense(256)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(12, activation='softmax')(x)
model = Model(inputs = Inception_base.input, outputs = predictions)

# Freeze the earlier layers
#for layer in model.layers[:-22]:
    #layer.trainable = False
    
    
# Compile the model    
model.compile(Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])



# Train the model
history_inception_v3 = model.fit(train_generator,
                      steps_per_epoch = 196,
                      validation_data = val_generator,
                      #validation_steps = 48,
                      epochs = 5,
                      verbose = 1,
                      callbacks = [reduce_lr, checkpoint, early_stop])


#Save the best model
model.save('inception_best_model.h5')

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size
def convert_bytes(size, unit=None):
    if unit == "KB":
        return print("File size: " + str(round(size/1024,3)) + " Kilobytes")
    elif unit =="MB":
        return print("File size: " + str(round(size/(1024*1024),3)) + " Megabytes")
    else:
        return print("File size: " + str(size)+ " bytes")            
convert_bytes(get_file_size('inception_best_model.h5'), "MB")
test_loss,test_acc = model.evaluate(val_generator,verbose =2)

print('test accuracy:', test_acc)
import pathlib
import time
tflite_models_dir = pathlib.Path(os.path.join(os.getcwd(),'tflite_models'))
tflite_models_dir.mkdir(exist_ok=True, parents=True)
converter_vgg = tf.lite.TFLiteConverter.from_keras_model(model)
# Convert to TF Lite without quantization
vgg16_tflite_file = tflite_models_dir/"vgg16.tflite"
vgg16_tflite_file.write_bytes(converter_vgg.convert())


interpreter = tf.lite.Interpreter(model_path='./tflite_models/vgg16.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("== Input details ==")
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])
print("\n== Output details ==")
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])
interpreter.resize_tensor_input(input_details[0]['index'], (32, 299, 299, 3))
interpreter.resize_tensor_input(output_details[0]['index'], (32,12 ))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("== Input details ==")
print("shape:", input_details[0]['shape'])
print("\n== Output details ==")
print("shape:", output_details[0]['shape'])
X_valid, Y_valid = next(iter(val_generator))
# Set batch of images into input tensor
interpreter.set_tensor(input_details[0]['index'], X_valid)
# Run inference
interpreter.invoke()
# Get prediction results
tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
predictions = tflite_model_predictions.argmax()
print("Prediction results shape:", predictions.shape)

predictions=pd.get_dummies(tflite_model_predictions)
predictions = np.array(predictions)
print(predictions)
def run_tflite_model(tflite_file, test_image_indices):
    global X_valid

  # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros((len(test_image_indices)), dtype=int)
    for i, test_image_index in enumerate(test_image_indices):
        test_image = X_valid[test_image_index]
        test_label = Y_valid[test_image_index]

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        predictions[i] = output.argmax()

    return predictions
def evaluate_model(tflite_file, model_type):
    global X_valid
    global Y_valid

    test_image_indices = range(X_valid.shape[0])
    predictions = run_tflite_model(tflite_file, test_image_indices)
    
    predictions=pd.get_dummies(predictions)
    predictions = np.array(predictions)
    print(predictions)
    
    accuracy = accuracy_score(y_true=Y_valid, y_pred=predictions) 

    #accuracy = (np.sum(Y_valid== predictions) * 100) / len(X_valid)

    return accuracy





op = evaluate_model(vgg16_tflite_file, model_type="Float")
print("Accuracy of TFLite - VGG : {}".format(op))


val_label_batch = np.argmax(val_label_batch,axis=1)

interpreter.set_tensor(input_details[0]['index'], val_image_batch)
# Run inference
interpreter.invoke()
# Get prediction results
tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
print("Prediction results shape:", tflite_model_predictions.shape)
rounded_labels=np.argmax(tflite_model_predictions, axis=0)

rounded_labels
from sklearn.metrics import accuracy_score
acc = accuracy_score(rounded_labels,val_label_batch)

acc

def plot_curves(train_dir, save_dir, history):
    plt.style.use('seaborn')
    '''
    Args:
    history(History callback): which has a history attribute containing the lists of successive losses and other metrics
    '''
    plt.style.use('seaborn')
    NUM_EPOCHS = len(history.history['loss'])
    plt.style.use("ggplot")
    plt.figure(figsize=(16,10))
    plt.plot(np.arange(0, NUM_EPOCHS), history.history['loss'], label='train_loss')
    plt.plot(np.arange(0, NUM_EPOCHS), history.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, NUM_EPOCHS), history.history['accuracy'], label='train_acc')
    plt.plot(np.arange(0, NUM_EPOCHS), history.history['val_accuracy'], label='val_acc')
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc = 'upper left')
    plt.tight_layout()
    plt.show()
plot_curves(train_dir,save_dir,history_inception_v3)
def plot_classification_metrics(categories, model, val_generator):

    predictiions = model.predict_generator(val_generator, 48)
    y_pred = np.argmax(predictiions, axis=1)
    cf_matrix = confusion_matrix(val_generator.classes, y_pred)
    print('Classification Report')
    print(classification_report(val_generator.classes, y_pred, target_names=categories))
    plt.figure(figsize=(20,20))
    sns.heatmap(cf_matrix, annot=True, xticklabels=categories, yticklabels=categories, cmap='Blues')
plot_classification_metrics(categories, model, val_generator)
#ResNet50
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import resnet50
from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

datagen = ImageDataGenerator(preprocessing_function = resnet50.preprocess_input, samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range = 0.15, 
                              width_shift_range = 0.15, 
                              rotation_range = 5, 
                              shear_range = 0.01,
                              fill_mode = 'nearest',
                              zoom_range=0.2, validation_split=0.15)
target_size = (224, 224)

train_generator = datagen.flow_from_directory(
                directory= os.path.join(save_dir),
                target_size= target_size,
                class_mode = "categorical",
                batch_size=32,
                shuffle=True,
                subset='training'
            )

val_generator = datagen.flow_from_directory(
                directory= os.path.join(save_dir),
                target_size= target_size,
                class_mode = 'categorical',
                batch_size=32,
                shuffle=False,
                subset='validation'
            )
#CallBacks
model_save_path = '/kaggle/working/model_resent50.h5'
checkpoint = ModelCheckpoint(filepath=model_save_path, monitor='val_loss', mode='min', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.00000001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', restore_best_weights=True)
resnet50_base = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
for layer in resnet50_base.layers[:-5]:
    layer.trainable = False
    
x = Flatten()(resnet50_base.output)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(12, activation='softmax')(x)

model_resnet50 = Model(resnet50_base.input, outputs = predictions)



model_resnet50.compile(Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])



# Train the model
history_resnet50 = model_resnet50.fit(train_generator,
                      steps_per_epoch = 196,
                      validation_data = val_generator,
                      #validation_steps = 48,
                      epochs = 25,
                      verbose = 1,
                      callbacks = [reduce_lr, checkpoint, early_stop])


#Save the best model
model_resnet50.save('resnet_best_model.h5')      
       
plot_curves(train_dir,save_dir,history_resnet50)
plot_classification_metrics(categories, model_resnet50, val_generator)
#VGG16

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import vgg16

datagen = ImageDataGenerator(preprocessing_function = vgg16.preprocess_input,samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range = 0.15, 
                              width_shift_range = 0.15, 
                              rotation_range = 5, 
                              shear_range = 0.01,
                              fill_mode = 'nearest',
                              zoom_range=0.2, validation_split=0.15)
target_size = (224, 224)

train_generator = datagen.flow_from_directory(
                directory= os.path.join(save_dir),
                target_size= target_size,
                class_mode = "categorical",
                batch_size=32,
                shuffle=True,
                subset='training'
            )

val_generator = datagen.flow_from_directory(
                directory= os.path.join(save_dir),
                target_size= target_size,
                class_mode = 'categorical',
                batch_size=32,
                shuffle=False,
                subset='validation'
            )
    

#callbacks

model_save_path = '/kaggle/working/model_vgg16.h5'
checkpoint = ModelCheckpoint(filepath=model_save_path, monitor='val_loss', mode='min', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.00000001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', restore_best_weights=True)
vgg16_base = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
for layer in vgg16_base.layers[:-11]:
    layer.trainable = False
x = vgg16_base.output
x = Dropout(0.5)(x)
    
predictions = Dense(12, activation='softmax')(x)
vgg16_model = Model(inputs = vgg16_base.input, outputs = predictions)


    
    
# Compile the model    
vgg16_model.compile(Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])



# Train the model
history_vgg16 = vgg16_model.fit(train_generator,
                      steps_per_epoch = 196,
                      validation_data = val_generator,
                      #validation_steps = 48,
                      epochs = 25,
                      verbose = 1,
                      callbacks = [reduce_lr, checkpoint, early_stop])
#Save the best model
vgg16_model.save('vgg_best_model.h5')
plot_curves(train_dir,save_dir,history_vgg16)
plot_classification_metrics(categories, vgg16_model, val_generator)
loaded_vgg16 = load_model("vgg_best_model.h5")
loaded_resnet50 = load_model("resnet_best_model.h5")
loaded_inception = load_model("inception_best_model.h5")