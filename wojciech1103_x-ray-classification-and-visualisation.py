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
def plot_imgs(item_dir, num_imgs=25):

    all_item_dirs = os.listdir(item_dir)

    item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:num_imgs]



    plt.figure(figsize=(10, 10))

    for idx, img_path in enumerate(item_files):

        plt.subplot(5, 5, idx+1)



        img = plt.imread(img_path)

        plt.imshow(img)



    plt.tight_layout()
plot_imgs(train_nrml_pth)
plot_imgs(train_pnm_pth)


def plot_img_hist(item_dir, num_img=6):

  all_item_dirs = os.listdir(item_dir)

  item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:num_img]

  

  #plt.figure(figsize=(10, 10))

  for idx, img_path in enumerate(item_files):

    fig1 = plt.figure(idx,figsize=(10, 10))

    fig1.add_subplot(2, 2, 1)

    img = mpimg.imread(img_path, )

    plt.imshow(img)

    fig1.add_subplot(2, 2, 2)

    plt.hist(img.ravel(),bins=256, fc='k', ec='k')

  

  plt.tight_layout()
plot_img_hist(train_pnm_pth,2)
def plot_img_hist_ndi(item_dir, num_img=6):

  all_item_dirs = os.listdir(item_dir)

  item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:num_img]

  

  #plt.figure(figsize=(10, 10))

  for idx, img_path in enumerate(item_files):

    im = imageio.imread(img_path)

    hist = ndi.histogram(im, min=0, max=255, bins=256)

    cdf = hist.cumsum() / hist.sum()

    

    fig1 = plt.figure(idx,figsize=(10, 10))

    fig1.add_subplot(2, 3, 1)

    img = mpimg.imread(img_path, )

    plt.title("No. {}".format(idx))

    plt.imshow(img)

    fig1.add_subplot(2, 3, 2)

    plt.title("Histogram")

    plt.plot(hist)

    fig1.add_subplot(2, 3, 3)

    plt.title("CDF")

    plt.plot(cdf)



  plt.tight_layout()
plot_img_hist_ndi(train_pnm_pth,2)
dirname_work = '/kaggle'

dir_chest_xray = os.path.join('/kaggle', 'chest_xray')

os.mkdir('/kaggle/chest_xray/')

os.mkdir('/kaggle/chest_xray/train')

os.mkdir('/kaggle/chest_xray/train/NORMAL')

os.mkdir('/kaggle/chest_xray/train/PNEUMONIA')

train_path_work = os.path.join(dir_chest_xray, 'train')

train_nrml_pth_work = os.path.join(train_path_work, 'NORMAL')

train_pnm_pth_work = os.path.join(train_path_work, 'PNEUMONIA')





os.mkdir('/kaggle/chest_xray/test')

os.mkdir('/kaggle/chest_xray/test/NORMAL')

os.mkdir('/kaggle/chest_xray/test/PNEUMONIA')

test_path_work = os.path.join(dir_chest_xray, 'test')

test_nrml_pth_work = os.path.join(test_path_work, 'NORMAL')

test_pnm_pth_work = os.path.join(test_path_work, 'PNEUMONIA')





# os.mkdir('/kaggle/chest_xray/val')

# os.mkdir('/kaggle/chest_xray/val/NORMAL')

# os.mkdir('/kaggle/chest_xray/val/PNEUMONIA')

# val_path = os.path.join(dirname, '/chest_xray/val')

# val_nrml_pth = os.path.join(val_path, 'NORMAL')

# val_pnm_pth = os.path.join(val_path, 'PNEUMONIA')

def image_resizing(path_from, path_to, height=500, width=500):

    size = height, width

    i=1

    files = os.listdir(path_from)

    for file in files: 

        try:

            file_dir = os.path.join(path_from, file)

            file_dir_save = os.path.join(path_to, file)

            img = Image.open(file_dir)

            img = img.resize(size, Image.ANTIALIAS)

            img = img.convert("RGB")

            img.save(file_dir_save) 

            i=i+1

        except:

            continue
image_resizing(train_nrml_pth, train_nrml_pth_work, 300, 300)
image_resizing(train_pnm_pth, train_pnm_pth_work, 300, 300)
image_resizing(test_nrml_pth, test_nrml_pth_work, 300, 300)

image_resizing(test_pnm_pth, test_pnm_pth_work, 300, 300)
plot_imgs(train_nrml_pth_work)
plot_imgs(train_pnm_pth_work)
def  hist_equal(path_from, path_to):

    i=1

    files = os.listdir(path_from)

    for file in files: 

        try:

            file_dir = os.path.join(path_from, file)

            file_dir_save = os.path.join(path_to, file)

            img = Image.open(file_dir)

            img = ImageOps.equalize(img)

            #img = img.convert("RGB") #konwersja z RGBA do RGB, usuniecie kanału alfa zeby zapisać do jpg

            img.save(file_dir_save) 

            i=i+1

        except:

            continue
hist_equal(train_pnm_pth_work, train_pnm_pth_work)

hist_equal(train_nrml_pth_work, train_nrml_pth_work)



hist_equal(test_pnm_pth_work, test_pnm_pth_work)

hist_equal(test_nrml_pth_work, test_nrml_pth_work)
plot_img_hist(train_pnm_pth_work, 2)
plot_img_hist_ndi(train_pnm_pth_work, 2)
def plot_hist_comparison(item_dir_before,item_dir_after, num_img=1):

    all_item_dirs = os.listdir(item_dir_before)

    item_files_before = [os.path.join(item_dir_before, file) for file in all_item_dirs][:num_img]

    item_files_after = [os.path.join(item_dir_after, file) for file in all_item_dirs][:num_img]

  

  #plt.figure(figsize=(10, 10))

    for idx, img_path in enumerate(item_files_before):

        im_b = imageio.imread(img_path)

        hist_b = ndi.histogram(im_b, min=0, max=255, bins=256)

        cdf_b = hist_b.cumsum() / hist_b.sum()

        

        img_path_a = item_files_after[idx]

        im_a = imageio.imread(img_path_a)

        hist_a = ndi.histogram(im_a, min=0, max=255, bins=256)

        cdf_a = hist_a.cumsum() / hist_a.sum()



        fig1 = plt.figure(idx,figsize=(10, 10))

        fig1.add_subplot(2, 4, 1)

        img_b = mpimg.imread(img_path, )

        plt.title("Before. {}".format(idx))

        plt.imshow(img_b, cmap='gray')

        fig1.add_subplot(2, 4, 3)

        plt.title("Histogram before")

        plt.plot(hist_b)

        fig1.add_subplot(2, 4, 4)

        plt.title("CDF before")

        plt.plot(cdf_b)

        

        fig2 = plt.figure(idx,figsize=(10, 10))

        fig2.add_subplot(2, 4, 2)

        img_a = mpimg.imread(img_path_a, )

        plt.title("Before. {}".format(idx))

        plt.imshow(img_a)

        fig2.add_subplot(2, 4, 3)

        plt.title("Histogram before")

        plt.plot(hist_a)

        fig1.add_subplot(2, 4, 4)

        plt.title("CDF before")

        plt.plot(cdf_a)



    plt.tight_layout()
plot_hist_comparison(train_nrml_pth, train_nrml_pth_work,2);
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
plot_model(model) #you can save picture with addin option to_file='model.png'
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
#we don't need it right now

training_examples = 5216

validation_examples = 624
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

    plt.imshow(imge)



    img = image.load_img(image_path, target_size=(img_size_h, img_size_w), color_mode='grayscale')

    x = image.img_to_array(img) 

    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])



    classes = model.predict_classes(images, batch_size=10)

    if classes[0][0] == 0:

        print("They got healthy!")

    else:

        print("They got pneumonia!")

predict_illness(img_train_path_ill)
predict_illness(img_test_path)
from keras.models import Model

layer_outputs = [layer.output for layer in model.layers[:len(model.layers)]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(images_p)



first_layer_activation = activations[0]

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
model.layers[:-1]# Droping The Last Dense Layer
layer_names = []

for layer in model.layers[:-1]:

    layer_names.append(layer.name) 

images_per_row = 16

zipped_layers = zip(layer_names, activations)

for layer_name, layer_activation in zipped_layers: #this loop     

    if layer_name.startswith('conv'):

        n_features = layer_activation.shape[-1]

        size = layer_activation.shape[1]

        n_cols = n_features // images_per_row

        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):

            for row in range(images_per_row):

                channel_image = layer_activation[0,:, :, col * images_per_row + row]

                channel_image -= channel_image.mean()

                channel_image /= channel_image.std()

                channel_image *= 64

                channel_image += 128

                channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                display_grid[col * size : (col + 1) * size,

                             row * size : (row + 1) * size] = channel_image

        scale = 1. / size

        plt.figure(figsize=(scale * display_grid.shape[1],

                            scale * display_grid.shape[0]))

        plt.title(layer_name)

        plt.grid(False)

        plt.imshow(display_grid, aspect='auto', cmap='viridis')
layer_names = []

for layer in model.layers[:-1]:

    layer_names.append(layer.name) 

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):

    if layer_name.startswith('max'):

        n_features = layer_activation.shape[-1]

        size = layer_activation.shape[1]

        n_cols = n_features // images_per_row

        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):

            for row in range(images_per_row):

                channel_image = layer_activation[0,:, :, col * images_per_row + row]

                channel_image -= channel_image.mean()

                channel_image /= channel_image.std()

                channel_image *= 64

                channel_image += 128

                channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                display_grid[col * size : (col + 1) * size,

                             row * size : (row + 1) * size] = channel_image

        scale = 1. / size

        plt.figure(figsize=(scale * display_grid.shape[1],

                            scale * display_grid.shape[0]))

        plt.title(layer_name)

        plt.grid(False)

        plt.imshow(display_grid, aspect='auto', cmap='viridis')