#Import libraries.

import numpy as np 

import pandas as pd 

import cv2

from skimage import restoration, filters, img_as_ubyte



from sklearn.preprocessing import scale, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss



from scipy.stats import mode

from scipy.ndimage.filters import uniform_filter

from scipy.ndimage.measurements import variance



from keras import applications, regularizers

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Input, SeparableConv2D, Add, Average

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, concatenate

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam, SGD, Nadam

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import Concatenate

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.applications.xception import Xception

from keras.applications.inception_v3 import InceptionV3

from keras.applications.resnet50 import ResNet50

from keras.applications.vgg19 import VGG19

from keras.applications.vgg16 import VGG16

from keras import backend as K



import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Define a function to plot the radar signals as images.

def img_plot(img_arr, ax, title):

    ax.imshow(img_arr)

    ax.axis('off')

    ax.grid('off')

    if title == 1:

        ax.set_title('Iceberg')

    elif title == 0:

        ax.set_title('Boat')
# Create the Lee filter function.

def lee_filter(img, size):

    img_mean = uniform_filter(img, (size, size))

    img_sqr_mean = uniform_filter(img**2, (size, size))

    img_variance = img_sqr_mean - img_mean**2



    overall_variance = variance(img)



    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)

    img_output = img_mean + img_weights * (img - img_mean)

    return img_output
#Now, define a funciton to use the Lee filter, denoise_nl_means and bilateral functions sequentially to remove noise.

#The filter parameters are defined in the function. 

def noise_removal(signal, img_h=75, img_w=75):

    #Convert the signal array to 2d array/image

    #Apply three filters sequentially to remove noise

    image = signal

    if len(signal.shape) == 1:

        image = signal.reshape(img_h, img_w)

    lee = lee_filter(image, 5)

    final_image = restoration.denoise_nl_means(lee, patch_size=5, patch_distance=10, h=2, 

                                               multichannel=False, fast_mode=True)

    return(final_image)
# Read the training dataset.

data_df = pd.read_json('../input/statoil-iceberg-classifier-challenge/train.json')
# Remove noise and save the images in new columns, ch1 and ch2. 

# Add another new column, ch3, which is the sum of band1 and band2 data.

# This step is time consuming!!

data_df = pd.concat([data_df, pd.DataFrame(columns = ['band3'], dtype='object')])

for row in range(0,len(data_df)):

    arr1 = np.array(data_df.loc[row,'band_1'])

    arr2 = np.array(data_df.loc[row,'band_2'])

    data_df.at[row,'band3'] = noise_removal(arr1 + arr2)
#plot the raw data in 2-D images

fig, axes = plt.subplots(nrows=2, ncols=6, sharex=False, sharey=False, figsize=(12,5))

for d, ax in zip([55,26,1369,1126,65,142,8,31,41,44,93,63], axes.flat):

    raw1 = np.array(data_df.loc[d, 'band_1']).reshape(75,75)

    title = data_df.loc[d, 'is_iceberg']

    img_plot(raw1, ax, title)



plt.tight_layout(pad=-1.2)
#plot the denoised "band3" images.

fig, axes = plt.subplots(nrows=2, ncols=6, sharex=False, sharey=False, figsize=(12,5))

for d, ax in zip([55,26,1369,1126,65,142,8,31,41,44,93,63], axes.flat):

    ch3 = data_df.loc[d, 'band3']

    title = data_df.loc[d, 'is_iceberg']

    img_plot(ch3, ax, title)



plt.tight_layout(pad=-1.2)
#Define a function to get the centroid of a contour.

#The centroid can be used to determined if a contour is too close to the edges of the image.

def get_center(contour):

    M = cv2.moments(contour)

    if M["m00"] != 0:

        cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]) 

    else:

        cX, cY = 0, 0 

    return(cX, cY)
#Define a function to list and remove all contours that are too close to the edges of the image.

#If all contours are close to the edges (no central object), no contour will be removed.

def get_central_cnts(cnts_lst):

    edge_lst = []

    central_lst = []

    for c in cnts_lst:

        cX, cY = get_center(c)

        if cX < 15 or cX > 60 or cY < 15 or cY > 60:

            edge_lst.append(c)

        else:

            central_lst.append(c)

    

    #If there is no central object, do not remove any object near the edges.

    if (len(edge_lst) == len(cnts_lst)) and (len(central_lst) == 0):

        edge_lst = []

        central_lst = cnts_lst

    

    return(central_lst, edge_lst)
#Define a function to find contours and create masks.

def cnt_to_mask(binary_image):

    #Find contours in the binary image.

    _, cnts, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:

        print('No contour is found!')

    

    #Create a mask and draw the contours on the mask to record the background coordinates.

    mask_bg = np.zeros((75,75), dtype='uint8')

    for c in cnts:

        cv2.drawContours(mask_bg, [c], -1, 1, -1)

    

    #Remove the contours that are too close to the edges.

    cnts_clean, cnts_edge = get_central_cnts(cnts)

    

    #Create a mask and draw the cleaned contours on the mask to record the foreground coordinates.

    mask_fg = np.zeros((75,75), dtype='uint8')

    for c in cnts_clean:

        cv2.drawContours(mask_fg, [c], -1, 1, -1)

    

    #Create a mask and draw the edge contours on the mask to record the edge coordinates.

    mask_eg = np.zeros((75,75), dtype='uint8')

    for c in cnts_edge:

        cv2.drawContours(mask_eg, [c], -1, 1, -1)

   

    return(mask_bg, mask_fg, mask_eg) 
# Define a function to generate binary images and create masks.

def get_mask(input_img, cutoff = filters.threshold_triangle):

    #Calculate the threshold and create the binary image.

    thresh = cutoff(input_img, nbins=256)   

    binary_img = input_img > thresh

    

    #Get the masks

    mask_bg, mask_fg, mask_eg = cnt_to_mask(img_as_ubyte(binary_img)) 

      

    #If any foreground contour touches the edge, the threshold probably didn't precisely cut out the object.

    #Modify the threshold, and recreate the binary image and masks.

    if sum(mask_fg[0,:]) > 1 or sum(mask_fg[74,:]) > 1 or sum(mask_fg[:,0]) > 1 or sum(mask_fg[74,:]) > 1:

        fg_coord = np.where(mask_fg.flatten() == 1)

        fg_mean = np.mean(np.take(input_img.flatten(), fg_coord[0]))

        fg_max = np.max(np.take(input_img.flatten(), fg_coord[0]))

        thresh_new = thresh + (fg_max - fg_mean) / 3

        binary_img_new = input_img > thresh_new

        mask_bg, mask_fg, mask_eg = cnt_to_mask(img_as_ubyte(binary_img_new)) 

    

    return(mask_bg, mask_fg, mask_eg)
#Check if the masks can be properly created.

fig, axes = plt.subplots(nrows=2, ncols=6, sharex=False, sharey=False, figsize=(12,5))

for d, ax in zip([55,26,1369,1126,65,142,8,31,41,44,93,63], axes.flat):

    mask_bg, mask_fg, mask_eg = get_mask(np.array(data_df.loc[d, 'band3']))

    final = cv2.bitwise_and(data_df.loc[d, 'band3'], data_df.loc[d, 'band3'], mask=mask_fg)

    title = data_df.loc[d, 'is_iceberg']

    img_plot(final, ax, title)



plt.tight_layout(pad=-1.2)
#Replace the objects on the edges with mean background signal. 

#Save the cleaned images in new columns, band1_cl and band2_cl.

data_df = pd.concat([data_df, pd.DataFrame(columns = ['band1_cl','band2_cl'], dtype='object')])

for r in range(0,len(data_df)):

    mask_bg, _, mask_eg = get_mask(np.array(data_df.loc[r,'band3']))    

    coord_bg = np.where(mask_bg.flatten() == 0)

    bg1 = np.mean(np.take(np.array(data_df.at[r,'band_1']), coord_bg[0]))

    bg2 = np.mean(np.take(np.array(data_df.at[r,'band_2']), coord_bg[0]))

    if np.sum(mask_eg) > 0:

        coord_eg = np.where(mask_eg.flatten() == 1)

        

        band1 = np.array(data_df.at[r,'band_1'])

        np.put(band1, coord_eg, bg1)

        data_df.at[r,'band1_cl'] = band1.reshape(75,75)

        

        band2 = np.array(data_df.at[r,'band_2'])

        np.put(band2, coord_eg, bg2)

        data_df.at[r,'band2_cl'] = band2.reshape(75,75)

    

    else:

        data_df.at[r,'band1_cl'] = np.array(data_df.at[r,'band_1']).reshape(75,75)

        data_df.at[r,'band2_cl'] = np.array(data_df.at[r,'band_2']).reshape(75,75)  
#Check if the images are cleaned.

fig, axes = plt.subplots(nrows=2, ncols=6, sharex=False, sharey=False, figsize=(12,5))

for d, ax in zip([55,26,1369,1126,65,142,8,31,41,44,93,63], axes.flat):

    image = data_df.at[d,'band1_cl']

    title = data_df.loc[d, 'is_iceberg']

    img_plot(image, ax, title)



plt.tight_layout(pad=-1.2)
# Now do noise_removal again using the cleaned images, and save the new images in new columns ch1-ch3.

data_df = pd.concat([data_df, pd.DataFrame(columns = ['ch1','ch2','ch3'], dtype='object')])

for row in range(0,len(data_df)):

    arr1 = data_df.loc[row,'band1_cl']

    arr2 = data_df.loc[row,'band2_cl']

    data_df.at[row,'ch1'] = noise_removal(arr1)

    data_df.at[row,'ch2'] = noise_removal(arr2)

    data_df.at[row,'ch3'] = noise_removal(arr1 + arr2)
#plot the denoised clean ch3 images.

fig, axes = plt.subplots(nrows=2, ncols=6, sharex=False, sharey=False, figsize=(12,5))

for d, ax in zip([55,26,1369,1126,65,142,8,31,41,44,93,63], axes.flat):

    ch3 = data_df.loc[d, 'ch3']

    title = data_df.loc[d, 'is_iceberg']

    img_plot(ch3, ax, title)



plt.tight_layout(pad=-1.2)
#Create new masks using the newly denoised images.

#Extract background and object backscattering values.

#Put the values in new columns, eng1 and eng2.

data_df = pd.concat([data_df, pd.DataFrame(columns = ['eng1','eng2'], dtype='float')])

for r in range(0,len(data_df)):

    #Create masks

    mask_bg, mask_fg, _ = get_mask(data_df.loc[r,'ch3'],cutoff = filters.threshold_otsu)



    #Extract backscattering signals from objects.

    coord_bg = np.where(mask_bg.flatten() == 0)

    coord_fg = np.where(mask_fg.flatten() == 1)

    bg1 = np.mean(np.take(np.array(data_df.at[r,'band1_cl']), coord_bg[0]))

    bg2 = np.mean(np.take(np.array(data_df.at[r,'band2_cl']), coord_bg[0]))

    sig1 = np.take(data_df.at[r,'band1_cl'], coord_fg[0])

    sig2 = np.take(data_df.at[r,'band2_cl'], coord_fg[0])

    data_df.at[r,'eng1'] = np.mean(sig1) - bg1

    data_df.at[r,'eng2'] = np.mean(sig2) - bg2
#Standardize the energy values.

data_df['eng1_std'] = (data_df['eng1'] - data_df['eng1'].mean()) / data_df['eng1'].std()

data_df['eng2_std'] = (data_df['eng2'] - data_df['eng2'].mean()) / data_df['eng2'].std()
#Pair plot the energy values to see whether they can be used for classification.

g = sns.PairGrid(data=data_df, 

                 hue='is_iceberg', #Variables in data for different colors.

                 hue_order=None, 

                 palette='husl', 

                 hue_kws={"marker": ["+", "x"]}, 

                 vars=['eng1_std','eng2_std'], #list of columns to use, otherwise use all columns

                 diag_sharey=True, 

                 size=3, #each facet

                 aspect=1, 

                 despine=True, 

                 dropna=True)

    

g = g.map_diag(plt.hist, edgecolor="black", linewidth=0.5)

g = g.map_offdiag(plt.scatter, linewidth=0.5, s=50)  

g = g.add_legend()
#plot a heatmap to check the multicollinearity.

cmap = sns.diverging_palette(150, 10, n=9, s=90, l=50, center='light', as_cmap=True)

fig, ax = plt.subplots(figsize=(4,4))

sns.heatmap(data_df[['eng1_std','eng2_std']].corr(), vmin=-1, vmax=1, cmap=cmap, annot=True, square=True, ax=ax, cbar_kws={'shrink':0.7})
# Scale the images and put them in new columns.

# In this case, I chose to scale them to [-1, 1].

data_df = pd.concat([data_df, pd.DataFrame(columns = ['scale_111','scale_112','scale_113',

                                                      'scale_b1','scale_b2','scale_b3'], dtype='object')])

for row in range(0,len(data_df)):

    arr1 = data_df.loc[row,'band1_cl']

    arr2 = data_df.loc[row,'band2_cl']

    arr3 = data_df.loc[row,'ch3']

    arr_b1 = np.array(data_df.loc[row,'band_1']).reshape(75,75)

    arr_b2 = np.array(data_df.loc[row,'band_2']).reshape(75,75)

    arr_b3 = data_df.loc[row,'band3']

    

#     # Rescale the image data to 0 to 1.

#     data_df.at[row,'n011'] = (arr1 - np.min(arr1))/(np.max(arr1)-np.min(arr1))

#     data_df.at[row,'n012'] = (arr2 - np.min(arr2))/(np.max(arr2)-np.min(arr2))

#     data_df.at[row,'n013'] = (arr3 - np.min(arr3))/(np.max(arr3)-np.min(arr3))

#     data_df.at[row,'n014'] = (arr4 - np.min(arr4))/(np.max(arr4)-np.min(arr4))

#     data_df.at[row,'n015'] = (arr5 - np.min(arr5))/(np.max(arr5)-np.min(arr5))

    

    # Rescale the image data to -1 to 1.

    data_df.at[row,'scale_111'] = 2*(arr1 - np.min(arr1))/(np.max(arr1)-np.min(arr1))-1

    data_df.at[row,'scale_112'] = 2*(arr2 - np.min(arr2))/(np.max(arr2)-np.min(arr2))-1

    data_df.at[row,'scale_113'] = 2*(arr3 - np.min(arr3))/(np.max(arr3)-np.min(arr3))-1    

    data_df.at[row,'scale_b1'] = 2*(arr_b1 - np.min(arr_b1))/(np.max(arr_b1)-np.min(arr_b1))-1

    data_df.at[row,'scale_b2'] = 2*(arr_b2 - np.min(arr_b2))/(np.max(arr_b2)-np.min(arr_b2))-1  

    data_df.at[row,'scale_b3'] = 2*(arr_b3 - np.min(arr_b3))/(np.max(arr_b3)-np.min(arr_b3))-1
# Prepare the arrays.

arr_111 = np.stack(data_df['scale_111'], axis=0)

arr_112 = np.stack(data_df['scale_112'], axis=0)

arr_113 = np.stack(data_df['scale_113'], axis=0)

arr_b1 = np.stack(data_df['scale_b1'], axis=0)

arr_b2 = np.stack(data_df['scale_b2'], axis=0)

arr_b3 = np.stack(data_df['scale_b3'], axis=0)

data_img = np.concatenate([arr_b1[:,:,:,np.newaxis], arr_b2[:,:,:,np.newaxis], arr_b3[:,:,:,np.newaxis]], axis=3)

print(data_img.shape)

data_y = np.array(data_df['is_iceberg'].astype(int))

print(data_y.shape)

data_feat = np.array(data_df[['eng1_std','eng2_std']].astype(float))

print(data_feat.shape)
# Data split into the training and test sets

img_train, img_test, feat_train, feat_test, y_train, y_test = train_test_split(data_img, 

                                                                               data_feat,

                                                                               data_y, 

                                                                               random_state=4, 

                                                                               test_size=0.25)

print(img_train.shape, feat_train.shape, img_test.shape, feat_test.shape)
input_shape = (75, 75, 3)
# Define data augmentation parameters

datagen_train = ImageDataGenerator(rotation_range=40.,

                                   shear_range=0.,

                                   zoom_range=0.,

                                   width_shift_range=0.1,

                                   height_shift_range=0.1,

                                   vertical_flip=True,

                                   horizontal_flip=True,

                                   samplewise_center=False,

                                   samplewise_std_normalization=False,

                                   fill_mode='nearest')



datagen_test = ImageDataGenerator(rotation_range=20.,

                                  vertical_flip=True,

                                  horizontal_flip=True,

                                  samplewise_center=False,

                                  samplewise_std_normalization=False)
#Define a function to make a customized data generator, combining image and non-image data.

def combo_gen(gen1, gen2):

    while True:

        set1 = gen1.next()

        set2 = gen2.next()

        yield [set2[0], set2[1]], set1[1]
# Define the VGG16-derived transfer learing model.

def vgg16_model_2input():

    base_model = VGG16(weights=None, input_shape=input_shape, include_top=False, classes=1)

    base_model.load_weights('../input/vgg16wgts/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    # Fix the lower-level 7 layer (no training). 

    for layer in base_model.layers[0:7]:

        layer.trainable = False     

    for layer in base_model.layers[7:]:

        layer.trainable = True   

    x = base_model.get_layer('block5_pool').output

    x = Flatten()(x)

    # Add a second input.

    eng_input = Input(shape=(2,), name="energy")

    combined = concatenate([x, eng_input])    

    combined = Dense(512, activation='relu', name='fc1')(combined)

    combined = Dropout(0.3)(combined)

    combined = Dense(128, activation='relu', name='fc2')(combined)

    combined = Dropout(0.3)(combined)

    predictions = Dense(1, activation='sigmoid')(combined)    

    combined_model = Model(inputs=[base_model.input, eng_input], outputs=predictions)

    sgd = SGD(lr=0.001, decay=0.001/1000, momentum=0.9, nesterov=True)

#     nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

#     adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

#     The Adam and Nadam optimizers didn't work! 

    combined_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return(combined_model)
# Create data generators for both image and non-image data.

batch_size = 8

train_gen1 = datagen_train.flow(img_train, y_train, shuffle=False, seed=1, batch_size=batch_size)

train_gen2 = datagen_train.flow(img_train, feat_train, shuffle=False, seed=1, batch_size=batch_size)

test_gen1 = datagen_test.flow(img_test, y_test, shuffle=False, seed=1, batch_size=batch_size)

test_gen2 = datagen_test.flow(img_test, feat_test, shuffle=False, seed=1, batch_size=batch_size)

# Create customized data generators with 2 inputs.

train_combo_gen = combo_gen(train_gen1, train_gen2)

test_combo_gen = combo_gen(test_gen1, test_gen2)

# Set up callbacks parameters. These are commonly used.

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')
model_vgg16_2input = vgg16_model_2input()
model_vgg16_2input.fit_generator(train_combo_gen, 

                                 epochs=50,

                                 steps_per_epoch=img_train.shape[0] // 8.0,

                                 verbose=1,

                                 callbacks=[earlyStopping, mcp_save, reduce_lr_loss],

                                 validation_data=test_combo_gen,

                                 validation_steps=img_test.shape[0] // 8.0

                                 )
model_vgg16_2input.load_weights('.mdl_wts.hdf5')

print('Training scores')

print(model_vgg16_2input.evaluate([img_train,feat_train], y_train))

print('Testing scores')

print(model_vgg16_2input.evaluate([img_test,feat_test], y_test))
del model_vgg16_2input

K.clear_session()