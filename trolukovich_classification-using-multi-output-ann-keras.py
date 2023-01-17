import numpy as np

import pandas as pd

import cv2 as cv

import matplotlib.pyplot as plt

import seaborn as sns

import os

import tensorflow as tf

import re

import requests



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelBinarizer



from keras.models import Sequential, Model

from keras.layers import Conv2D, Dense, BatchNormalization, Dropout 

from keras.layers import Flatten, Lambda, Input, MaxPooling2D

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

from keras.regularizers import l1, l2



%config InlineBackend.figure_format = 'svg'
# Input shape for ANN and to resize images

input_shape = (110, 110, 3)
X = [] # Lists for images data

Y_cat = [] # List to store category of image (dress, shirt etc)

Y_col = [] # List to store color of apparel



# Path to datast

dataset = '../input/apparel-images-dataset'



# Loop through all categories in dataset

for folder in os.listdir(dataset):

    

    # Path to images in certain category

    folder_path = os.path.join(dataset, folder)

    

    # Loop through all images in category

    for i in os.listdir(folder_path):

        

        # Path to certain image (Example: ../input/apparel-images-dataset/blue_dress/11d5fd4203b08f26dac2e7fa2294c6f01babee15.jpg)

        path_to_image = os.path.join(folder_path, i)

        

        # Reading and resizing image to 96x96 pixels size

        image = cv.imread(path_to_image)

        image = cv.resize(image, (input_shape[1], input_shape[0]))

        

        # using regex to extract labels from path_to_image

        labels = re.findall(r'\w+\_\w+', path_to_image) # Gives us ['blue_dress']

        labels = labels[0].split('_') # Gives us ['blue', 'dress']

        

        # Adding data and labels to lists

        X.append(image)

        Y_cat.append(labels[1])

        Y_col.append(labels[0])
# Convert X to numpy array

X = np.array(X) / 255.0



# Binarizing labels

lb_cat = LabelBinarizer()

Y_cat = lb_cat.fit_transform(Y_cat)



lb_col = LabelBinarizer()

Y_col = lb_col.fit_transform(Y_col)
# Labels

print('Category classes:')

[print(i) for i in zip(lb_cat.classes_, np.unique(np.argmax(Y_cat, axis = 1)))]



print('\nColor classes:')

[print(i) for i in zip(lb_col.classes_, np.unique(np.argmax(Y_col, axis = 1)))]
# test_x, cat_y_test, col_y_test - for final validation

x, test_x, cat_y, cat_y_test, col_y, col_y_test = train_test_split(X, Y_cat, Y_col, test_size = 0.1, shuffle = True, random_state = 1)



# Train and validation data

train_x, val_x, cat_y_train, cat_y_val, col_y_train, col_y_val = train_test_split(x, cat_y, col_y, test_size = 0.2)
inputs = Input(shape = input_shape)



# Category branch

cat = Lambda(lambda z: tf.image.rgb_to_grayscale(z))(inputs)



cat = Conv2D(32, 5, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(cat)

cat = BatchNormalization()(cat)

cat = Conv2D(32, 5, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(cat)

cat = BatchNormalization()(cat)

cat = MaxPooling2D()(cat)

cat = Dropout(0.25)(cat)



cat = Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(cat)

cat = BatchNormalization()(cat)

cat = Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(cat)

cat = BatchNormalization()(cat)

cat = MaxPooling2D()(cat)

cat = Dropout(0.25)(cat)



cat = Conv2D(128, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(cat)

cat = BatchNormalization()(cat)

cat = Conv2D(128, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(cat)

cat = BatchNormalization()(cat)

cat = MaxPooling2D()(cat)

cat = Dropout(0.25)(cat)



cat = Flatten()(cat)

cat = Dense(128, activation = 'relu', kernel_initializer = 'he_normal')(cat)

cat = BatchNormalization()(cat)

cat = Dropout(0.5)(cat)

cat = Dense(lb_cat.classes_.shape[0], activation = 'softmax', name = 'cat')(cat)





#Color branch

col = Conv2D(16, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(inputs)

col = BatchNormalization(axis = -1)(col)

col = MaxPooling2D(3)(col)

col = Dropout(0.25)(col)



col = Conv2D(32, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(col)

col = BatchNormalization(axis = -1)(col)

col = MaxPooling2D()(col)

col = Dropout(0.25)(col)



col = Conv2D(32, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(col)

col = BatchNormalization(axis = -1)(col)

col = MaxPooling2D()(col)

col = Dropout(0.25)(col)



col = Flatten()(col)

col = Dense(128, activation = 'relu', kernel_initializer = 'he_normal')(col)

col = BatchNormalization()(col)

col = Dropout(0.5)(col)

col = Dense(lb_col.classes_.shape[0], activation = 'softmax', name = 'col')(col)





model = Model(inputs = inputs, outputs = [cat, col])

losses = {'cat': 'categorical_crossentropy',

         'col': 'categorical_crossentropy'}

loss_weights = {'cat': 1.0, 'col': 1.0}



checkpoint = ModelCheckpoint('../working/best_model.hdf5', save_best_only = True, verbose = 1, monitor = 'val_loss')



model.compile(optimizer = 'adam', loss = losses, 

              loss_weights = loss_weights, metrics = ['accuracy'])



history = model.fit(train_x, {'cat': cat_y_train, 'col': col_y_train},

         validation_data = (val_x, {'cat': cat_y_val, 'col': col_y_val}),

         batch_size = 64, epochs = 30, callbacks = [checkpoint])

# Plot learning curves

H = history.history

fig = plt.figure(figsize = (13, 5))



for i, c in enumerate(('cat', 'col')):

    plt.subplot(f'12{i+1}')

    plt.plot(H[f'{c}_accuracy'], label = f'{c}_acc')

    plt.plot(H[f'val_{c}_accuracy'], label = f'val_{c}_acc')

    plt.plot(H[f'{c}_loss'], label = f'{c}_loss')

    plt.plot(H[f'val_{c}_loss'], label = f'val_{c}_loss')

    plt.title(f'{c} learning curves')

    plt.legend()

    plt.grid()
# Load best weights

model.load_weights('../working/best_model.hdf5')



# Predicting test images

cat_preds, col_preds = model.predict(test_x)

cat_preds = np.argmax(cat_preds, axis = 1)

col_preds = np.argmax(col_preds, axis = 1)



# Creating confusion matrices

cat_confusion = confusion_matrix(np.argmax(cat_y_test, axis = 1), cat_preds)

col_confusion = confusion_matrix(np.argmax(col_y_test, axis = 1), col_preds)
# Plotting confusion matrices

fig = plt.figure(figsize = (13, 5))



# Category confusion matrix plot

plt.subplot(121)

cat_l = list(lb_cat.classes_)

sns.heatmap(cat_confusion, square = True, annot = True, fmt = 'd', cbar = False, 

            xticklabels = cat_l, yticklabels = cat_l, cmap = 'coolwarm').set_title('Category confusion matrix')



# Color confusion matrix plot

plt.subplot(122)

col_l = list(lb_col.classes_)

sns.heatmap(col_confusion, square = True, annot = True, fmt = 'd', cbar = False, 

            xticklabels = col_l, yticklabels = col_l, cmap = 'coolwarm').set_title('Color confusion matrix')



plt.show()
def plot_images(urls, rows, cols):

    fig = plt.figure(figsize = (13, rows * 5))



    for i, url in enumerate(urls):

        plt.subplot(f'{rows}{cols}{i+1}')



        # Sending request to the URL

        r = requests.get(url, stream = True).raw



        # Reading image, convert it to np array and decode

        image = np.asarray(bytearray(r.read()), dtype="uint8")

        image = cv.imdecode(image, cv.IMREAD_COLOR)



        # Resize, scale and reshape image before making predictions

        resized = cv.resize(image, (input_shape[1], input_shape[0]))

        resized = (resized / 255.0).reshape(-1, input_shape[1], input_shape[0], input_shape[2])



        # Predict results

        preds_cat, preds_col = model.predict(resized)

        preds_cat = (lb_cat.classes_[np.argmax(preds_cat[0])], round(preds_cat[0].max() * 100, 2))

        preds_col = (lb_col.classes_[np.argmax(preds_col[0])], round(preds_col[0].max() * 100, 2))



        # Showing image

        plt.imshow(image[:, :, ::-1])

        plt.title(f'{preds_cat[0]}: {preds_cat[1]}% \n {preds_col[0]}: {preds_col[1]}%')

        plt.axis('off')
urls = [

    'http://picture-cdn.wheretoget.it/e07ql5-l-610x610-dress-little+black+dress-skater+dress-nastygal-deep+vneck-short-formal-short+formal+dress-prom-short+prom+dress-black-lbd-short+black+dress-prom+dress-black+dress-blackdress-short+.jpg',

    'https://img.simplydresses.com/_img/SDPRODUCTS/2103981/500/navy-dress-JU-TI-T0468-a.jpg',

    'https://d2euz5hho4dp59.cloudfront.net/images/detailed/40/main_jean_419.jpg',

    'https://sc02.alicdn.com/kf/HTB1QbZ_dzgy_uJjSZJnq6zuOXXaq/Wholesale-scratch-pants-damaged-denim-women-s.jpg_350x350.jpg',

    'https://i.ebayimg.com/00/s/NjAwWDYwMA==/z/pakAAOSwVtta6SN8/$_1.JPG?set_id=880000500F',

    'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSFA1Q-O44dQWt1lvsnOQyoMcQ3myaxY-GscMHgmPtmyWT14ZJU',

    'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTJYyBOAy35RM7m0JzNGHo_-VTSf6bPMh9hACbhhqxsdoMXHQvD',

    'https://cdn.shopify.com/s/files/1/1359/6121/products/7481209_LF0A0919_1024x1024.jpg?v=1511982241',    

]



plot_images(urls, 2, 4)
urls = [    

    'https://lp2.hm.com/hmgoepprod?set=source[/3c/5f/3c5fa4806ee4a7a9b958041041fd67b1f9a0829b.jpg],origin[dam],category[],type[DESCRIPTIVESTILLLIFE],res[m],hmver[1]&call=url[file:/product/main]',

    'https://is4.revolveassets.com/images/p4/n/d/AXIS-WD346_V1.jpg',

    'https://cdn-img.prettylittlething.com/f/4/3/a/f43a25de8714d69982b21c107a64f54e16647b08_clz6006_1.jpg',

    'https://cdn11.bigcommerce.com/s-9p1fzopril/images/stencil/500x659/products/352/1416/Screen_Shot_2019-08-10_at_9.02.38_am__31648.1565392335.png?c=2',

    'https://img0.junaroad.com/uiproducts/15054911/zoom_0-1524133662.jpg',

    'https://cdn.shopify.com/s/files/1/0021/2343/2045/products/Attachment_1550340699_1200x1200.jpeg?v=1550587174',

    'https://underarmour.scene7.com/is/image/Underarmour/V5-1350196-600_FC?template=v65GridSmallRetina&$wid=300&$hei=368',

    'https://pa.namshicdn.com/product/A8/13685W/1-zoom-desktop.jpg'    

]



plot_images(urls, 2, 4)