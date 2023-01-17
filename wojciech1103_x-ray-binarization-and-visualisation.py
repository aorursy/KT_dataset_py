import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd 

import imageio



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from PIL import Image, ImageOps

import scipy.ndimage as ndi



from keras.models import Sequential

from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Activation, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing import image

from keras.utils import plot_model
dirname = '/kaggle/input'

train_path = os.path.join(dirname, 'chest-xray-pneumonia/chest_xray/chest_xray/train')

train_nrml_pth = os.path.join(train_path, 'NORMAL')

train_pnm_pth = os.path.join(train_path, 'PNEUMONIA')

test_path = os.path.join(dirname, 'chest-xray-pneumonia/chest_xray/chest_xray/test')

test_nrml_pth = os.path.join(test_path, 'NORMAL')

test_pnm_pth = os.path.join(test_path, 'PNEUMONIA')

val_path = os.path.join(dirname, 'chest-xray-pneumonia/chest_xray/chest_xray/test')

val_nrml_pth = os.path.join(val_path, 'NORMAL')

val_pnm_pth = os.path.join(val_path, 'PNEUMONIA')
def plot_imgs(item_dir, num_imgs=25, cmap='viridis'):

    all_item_dirs = os.listdir(item_dir)

    item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:num_imgs]



    plt.figure(figsize=(10, 10))

    for idx, img_path in enumerate(item_files):

        plt.subplot(5, 5, idx+1)



        img = plt.imread(img_path)

        plt.imshow(img, cmap)



    plt.tight_layout()
plot_imgs(train_nrml_pth, cmap='magma')
plot_imgs(train_pnm_pth, cmap='magma')


os.mkdir('/kaggle/chest_xray/')

os.mkdir('/kaggle/chest_xray/train')

os.mkdir('/kaggle/chest_xray/train/NORMAL')

os.mkdir('/kaggle/chest_xray/train/PNEUMONIA')







os.mkdir('/kaggle/chest_xray/test')

os.mkdir('/kaggle/chest_xray/test/NORMAL')

os.mkdir('/kaggle/chest_xray/test/PNEUMONIA')







os.mkdir('/kaggle/chest_xray/val')

os.mkdir('/kaggle/chest_xray/val/NORMAL')

os.mkdir('/kaggle/chest_xray/val/PNEUMONIA')



dirname_work = '/kaggle'

dir_chest_xray = os.path.join('/kaggle', 'chest_xray')



train_path_work = os.path.join(dir_chest_xray, 'train')

train_nrml_pth_work = os.path.join(train_path_work, 'NORMAL')

train_pnm_pth_work = os.path.join(train_path_work, 'PNEUMONIA')



test_path_work = os.path.join(dir_chest_xray, 'test')

test_nrml_pth_work = os.path.join(test_path_work, 'NORMAL')

test_pnm_pth_work = os.path.join(test_path_work, 'PNEUMONIA')



val_path_work = os.path.join(dirname, '/chest_xray/val')

val_nrml_pth_work = os.path.join(val_path_work, 'NORMAL')

val_pnm_pth_work = os.path.join(val_path_work, 'PNEUMONIA')
def image_binarization(path_from, path_to):



    i=1

    files = os.listdir(path_from)

    for file in files: 

        try:

            file_dir = os.path.join(path_from, file)

            file_dir_save = os.path.join(path_to, file)

            img = Image.open(file_dir)

            img = img.convert("1")

            img.save(file_dir_save) 

            i=i+1

        except:

            continue
image_binarization(train_nrml_pth, train_nrml_pth_work)
image_binarization(train_pnm_pth, train_pnm_pth_work)
image_binarization(test_nrml_pth, test_nrml_pth_work)

image_binarization(test_pnm_pth, test_pnm_pth_work)
def plot_imgs_bin(item_dir, num_imgs=25):

    all_item_dirs = os.listdir(item_dir)

    item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:num_imgs]



    plt.figure(figsize=(10, 10))

    for idx, img_path in enumerate(item_files):

        plt.subplot(5, 5, idx+1)



        img = plt.imread(img_path)

        plt.imshow(img, 'binary')



    plt.tight_layout()
train_nrml_pth_list = os.listdir(train_nrml_pth)

train_nrml_pth_work_list = os.listdir(train_nrml_pth_work)

img_bfr = plt.imread(os.path.join(train_nrml_pth, train_nrml_pth_list[0]))

img_aftr = plt.imread(os.path.join(train_nrml_pth_work,train_nrml_pth_list[0]))

plt.subplot(2,3,1)

plt.title("Before: {}".format(train_nrml_pth_list[0]))

plt.imshow(img_bfr)

plt.subplot(2,3,3)

plt.title("After: {}".format(train_nrml_pth_list[0]))

plt.imshow(img_aftr, 'binary')

print("Shape before: {} \nShape after: {}".format(img_bfr.shape, img_aftr.shape))
plot_imgs_bin(train_nrml_pth_work)
plot_imgs_bin(train_pnm_pth_work)
img_size_h = 300

img_size_w = 300



input_shape = (img_size_h, img_size_w, 1) 
model = Sequential([

    Conv2D(32, (3,3), input_shape=input_shape),

    MaxPool2D((2, 2)),

    

    Conv2D(32, (3,3)),

    MaxPool2D((2, 2)),

    

    Conv2D(64, (3,3)),

    MaxPool2D((2, 2)),

    

    Conv2D(64, (3,3)),

    MaxPool2D((2, 2)),

    

    Flatten(),

    

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(1, activation='sigmoid')

    

    

])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
train_datagen = ImageDataGenerator(

    rescale=1./255,    

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=45,

    width_shift_range=0.5,

    height_shift_range=0.5,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



val_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=45,

    width_shift_range=0.5,

    height_shift_range=0.5,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)

batch_size = 32

train_generator = train_datagen.flow_from_directory(

    train_path_work,

    target_size=(img_size_h, img_size_w),

    color_mode='grayscale', #we use grayscale images I think

    batch_size=batch_size,

    class_mode='binary',

    shuffle=True, #we shuffle our images for better performance

    seed=8)



validation_generator = val_datagen.flow_from_directory(

    test_path_work,

    target_size=(img_size_h, img_size_w),

    color_mode='grayscale',

    batch_size=batch_size,

    class_mode='binary',

    shuffle=True,

    seed=8)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.0001) #0.00001

callback = [learning_rate_reduction]
history = model.fit_generator(

    train_generator,

    epochs=20,

    validation_data=validation_generator,

    callbacks = callback

    )
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
img_test_path = os.path.join(test_nrml_pth, 'NORMAL2-IM-0337-0001.jpeg')

img_train_path_ill = os.path.join(train_pnm_pth, 'person1787_bacteria_4634.jpeg')

img_p = image.load_img(img_test_path, target_size=(img_size_h, img_size_w), color_mode='grayscale')

img_arr_p = np.array(img_p)

img_arr_p = np.expand_dims(img_arr_p, axis=0)

img_arr_p = np.expand_dims(img_arr_p, axis=3)

images_p = np.vstack([img_arr_p])
def predict_illness(image_path):

    imge = plt.imread(image_path)

    plt.imshow(imge, cmap='magma')



    img = image.load_img(image_path, target_size=(img_size_h, img_size_w), color_mode='grayscale')

    x = image.img_to_array(img) 

    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])



    classes = model.predict_classes(images, batch_size=10)

    if classes[0][0] == 0:

        print("They are healthy!")

    else:

        print("They got pneumonia!")

predict_illness(img_test_path)