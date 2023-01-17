# set seeds to ensure repeatability of results

from numpy.random import seed

seed(101)

from tensorflow import set_random_seed

set_random_seed(101)



import pandas as pd

import numpy as np

import os

import cv2



import shutil



import tensorflow 



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, ZeroPadding2D

from tensorflow.keras.layers import BatchNormalization, LeakyReLU

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import categorical_crossentropy

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.metrics import binary_accuracy



import matplotlib.pyplot as plt

%matplotlib inline



from skimage.io import imread, imshow

from skimage.transform import resize





# Don't Show Warning Messages

import warnings

warnings.filterwarnings('ignore')
IMG_HEIGHT = 128

IMG_WIDTH = 128

IMG_CHANNELS = 3



PADDING = 40



NUM_TEST_IMAGES = 10
# ==============================    

# Make a List of Test Set Masks

# ==============================



test_id_list = ['SIMCEPImages_A15_C61_F1_s03_w2.TIF',

 'SIMCEPImages_A21_C87_F1_s03_w1.TIF',

 'SIMCEPImages_A01_C1_F1_s02_w1.TIF',

 'SIMCEPImages_A04_C14_F1_s05_w1.TIF',

 'SIMCEPImages_A18_C74_F1_s01_w1.TIF',

 'SIMCEPImages_A04_C14_F1_s18_w1.TIF',

 'SIMCEPImages_A18_C74_F1_s09_w2.TIF',

 'SIMCEPImages_A13_C53_F1_s10_w2.TIF',

 'SIMCEPImages_A08_C31_F1_s13_w2.TIF',

 'SIMCEPImages_A19_C78_F1_s15_w2.TIF']



num_cells = [61, 87, 1, 14, 74, 14, 74, 53, 31, 78]





# =====================    

# Create X_test

# ===================== 



# create an empty matrix

X_test = np.zeros((len(test_id_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)) #, dtype=np.bool)





for i, mask_id in enumerate(test_id_list):

    

    path_mask = '../input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth/' + mask_id

    

    # read the file as an array

    cv2_image = cv2.imread(path_mask)

    # resize the image

    cv2_image = cv2.resize(cv2_image, (IMG_HEIGHT, IMG_WIDTH))

    # save the image at the destination as a jpg file

    cv2.imwrite('mask.jpg', cv2_image)



    

    # read the image using skimage

    mask = imread('mask.jpg')



    

    # resize the image

    #mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    

    # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)

    #mask = np.expand_dims(mask, axis=-1)

    

    # insert the image into X_Train

    X_test[i] = mask



    

    

# =====================    

# Display the masks

# =====================  



# set up the canvas for the subplots

plt.figure(figsize=(10,10))

plt.axis('Off')



# Our subplot will contain 3 rows and 3 columns

# plt.subplot(nrows, ncols, plot_number)





# == row 1 ==



# image

plt.subplot(3,3,1)

test_image = X_test[1, :, :, 0]

plt.imshow(test_image)

plt.title('Num Cells: 87', fontsize=14)

plt.axis('off')



# image

plt.subplot(3,3,2)

test_image = X_test[2, :, :, 0]

plt.imshow(test_image)

plt.title('Num Cells: 1', fontsize=14)

plt.axis('off')





# image

plt.subplot(3,3,3)

test_image = X_test[3, :, :, 0]

plt.imshow(test_image)

plt.title('Num Cells: 14', fontsize=14)

plt.axis('off')





# == row 2 ==



# image

plt.subplot(3,3,4)

test_image = X_test[4, :, :, 0]

plt.imshow(test_image)

plt.title('Num Cells: 74', fontsize=14)

plt.axis('off')



# image

plt.subplot(3,3,5)

test_image = X_test[5, :, :, 0]

plt.imshow(test_image)

plt.title('Num Cells: 14', fontsize=14)

plt.axis('off')



# image

plt.subplot(3,3,6)

test_image = X_test[6, :, :, 0]

plt.imshow(test_image)

plt.title('Num Cells: 74', fontsize=14)

plt.axis('off')





# == row 3 ==



# image

plt.subplot(3,3,7)

test_image = X_test[7, :, :, 0]

plt.imshow(test_image)

plt.title('Num Cells: 53', fontsize=14)

plt.axis('off')



# image

plt.subplot(3,3,8)

test_image = X_test[8, :, :, 0]

plt.imshow(test_image)

plt.title('Num Cells: 31', fontsize=14)

plt.axis('off')





# image

plt.subplot(3,3,9)

test_image = X_test[9, :, :, 0]

plt.imshow(test_image)

plt.title('Num Cells: 78', fontsize=14)

plt.axis('off')





plt.tight_layout()

plt.show()

# get a list of files in each folder



img_list = os.listdir('../input/bbbc005_v1_images/BBBC005_v1_images')

mask_list = os.listdir('../input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth')



# create a dataframe

df_images = pd.DataFrame(img_list, columns=['image_id'])



# filter out the non image file that's called .htaccess

df_images = df_images[df_images['image_id'] != '.htaccess']







# Example file name: SIMCEPImages_A13_C53_F1_s23_w2.TIF





# ======================================================

# Add a column showing how many cells are on each image

# ======================================================



def get_num_cells(x):

    # split on the _

    a = x.split('_')

    # choose the third item

    b = a[2] # e.g. C53

    # choose second item onwards and convert to int

    num_cells = int(b[1:])

    

    return num_cells



# create a new column called 'num_cells'

df_images['num_cells'] = df_images['image_id'].apply(get_num_cells)





# ================================================

# Add a column indicating if an image has a mask.

# ================================================



# Keep in mind images and masks have the same file names.



def check_for_mask(x):

    if x in mask_list:

        return 'yes'

    else:

        return 'no'

    

# create a new column called 'has_mask'

df_images['has_mask'] = df_images['image_id'].apply(check_for_mask)







# ===========================================================

# Add a column showing how much blur was added to each image

# ===========================================================



def get_blur_amt(x):

    # split on the _

    a = x.split('_')

    # choose the third item

    b = a[3] # e.g. F1

    # choose second item onwards and convert to int

    blur_amt = int(b[1:])

    

    return blur_amt



# create a new column called 'blur_amt'

df_images['blur_amt'] = df_images['image_id'].apply(get_blur_amt)

df_images.head()
df_masks = df_images[df_images['has_mask'] == 'yes']



# create a new column called mask_id that is just a copy of image_id

df_masks['mask_id'] = df_masks['image_id']



df_masks.shape
df_masks.head()
# Create a new directory

base_dir = 'base_dir'

os.mkdir(base_dir)





#[CREATE FOLDERS INSIDE THE BASE DIRECTORY]



# now we create 3 folders inside 'base_dir':



    # bground_dir



    # cell_dir



    # noise_dir





# bground_dir

bground_dir = os.path.join(base_dir, 'bground_dir')

os.mkdir(bground_dir)



# cell_dir

cell_dir = os.path.join(base_dir, 'cell_dir')

os.mkdir(cell_dir)



# noise_dir

noise_dir = os.path.join(base_dir, 'noise_dir')

os.mkdir(noise_dir)
# check that the folders have been created



os.listdir('base_dir')
mask_id = 'SIMCEPImages_A04_C14_F1_s05_w1.TIF'



path_mask = '../input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth/' + mask_id

    

# read the file as an array

image = cv2.imread(path_mask)



print(image.shape)



plt.imshow(image)



plt.show()
fname = 'bground_1.png'



y = 100

x = 100

h = 150

w = 150



bground = image[y:y+h, x:x+w].copy()



plt.imshow(bground)



path = 'base_dir/bground_dir/' + fname

cv2.imwrite(path, bground)
# cell_1



fname = 'cell_1.png'



y=445

x=265

h=40

w=45



cell_1 = image[y:y+h, x:x+w]



plt.imshow(cell_1)



# save the image at the destination as a png file

path = 'base_dir/cell_dir/' + fname

cv2.imwrite(path, cell_1)



print(cell_1.shape)
# cell_2 



fname = 'cell_2.png'



y=20

x=195

h=45

w=40



cell_2 = image[y:y+h, x:x+w]



plt.imshow(cell_2)



# save the image at the destination as a png file

path = 'base_dir/cell_dir/' + fname

cv2.imwrite(path, cell_2)



print(cell_2.shape)
# cell_3



fname = 'cell_3.png'



y=155

x=450

h=45

w=45



cell_3 = image[y:y+h, x:x+w]



plt.imshow(cell_3)



# save the image at the destination as a png file

path = 'base_dir/cell_dir/' + fname

cv2.imwrite(path, cell_3)



print(cell_3.shape)
# DEFINE THE FUNCTION TO CREATE ARTIFICIAL IMAGES







# ==============================



# set max x and y to prevent cells from extending outside the bachground image i.e.

# if the get place too close to the edge.

max_x = 696 - 45

max_y = 520 - 45



# ==============================





def generate_artificial_images(num_images_wanted=10, min_cells_on_image=1, 

                               max_cells_on_image=100, fname='images_A_'):

    

    # delete the existing images in new_image_dir by deleting the folder if it exists

    #if os.path.isdir('new_images_dir') == True: # return true if the directory exists



            #shutil.rmtree('new_images_dir')



    # Create a new folder if it does not exist

    

    if os.path.isdir('new_images_dir') == False:

        new_images_dir = 'new_images_dir'

        os.mkdir(new_images_dir)







    for i in range(0, num_images_wanted):





        # randomly choose the number of cells to put in the image

        num_cells_on_image = np.random.randint(min_cells_on_image, max_cells_on_image+1)



        # Name the image.

        # The number of cells is included in the file name.

        image_name = str(fname) + '_' + str(i) + '_'  + str(num_cells_on_image) + '_.png'

       



        # =========================

        # 1. Create the background

        # =========================



        path = 'base_dir/bground_dir/bground_1.png'



        # read the image

        bground_comb = cv2.imread(path)



        # add random rotation to the background

        num_k = np.random.randint(0,3)

        bground_comb = np.rot90(bground_comb, k=num_k)



        # resize the background to match what we want

        bground_comb = cv2.resize(bground_comb, (696, 520))









        # ===============================

        # 2. Add cells to the background

        # ===============================



        for j in range(0, num_cells_on_image):



            path = 'base_dir/bground_dir/bground_1.png'



            # read the image

            bground = cv2.imread(path)

            # add rotation to the background

            bground = np.rot90(bground, k=num_k)

            # resize the background to match what we want

            bground = cv2.resize(bground, (696, 520))





            # randomly choose a type of cell to add to the image

            cell_type = np.random.randint(1,3+1)



            if cell_type == 1:



                # cell_1 path

                cell_1 = cv2.imread('base_dir/cell_dir/cell_1.png')



                # add a random rotation to the cell

                cell_1 = np.rot90(cell_1, k=np.random.randint(0,3))



                # get the shape after rotation

                shape = cell_1.shape



                # get a random x-coord

                y=np.random.randint(0, max_y)

                # get a random y-coord

                x=np.random.randint(0, max_x)

                # set the width and height

                h=shape[0]

                w=shape[1]



                # add the cell to the background

                bground[y:y+h, x:x+w] = 0

                bground[y:y+h, x:x+w] = cell_1





            if cell_type == 2:



                cell_2 = cv2.imread('base_dir/cell_dir/cell_2.png')



                # add a random rotation to the cell

                cell_2 = np.rot90(cell_2, k=np.random.randint(0,3))



                shape = cell_2.shape



                y=np.random.randint(0,max_y)

                x=np.random.randint(0,max_x)

                h=shape[0]

                w=shape[1]



                bground[y:y+h, x:x+w] = 0

                bground[y:y+h, x:x+w] = cell_2





            if cell_type == 3:



                # cell_3



                cell_3 = cv2.imread('base_dir/cell_dir/cell_3.png')



                # add a random rotation to the cell

                cell_3 = np.rot90(cell_3, k=np.random.randint(0,3))



                shape = cell_3.shape



                y=np.random.randint(0,max_y)

                x=np.random.randint(0,max_x)

                h=shape[0]

                w=shape[1]



                bground[y:y+h, x:x+w] = 0

                bground[y:y+h, x:x+w] = cell_3





            bground_comb = np.maximum(bground_comb, bground)





        # ===============================

        # 3. Save the image

        # ===============================



        path = 'new_images_dir/' + image_name

        cv2.imwrite(path, bground_comb)







    print('Num images created: ', num_images_wanted)

    
# CALL THE FUNCTION



# All artificial images from both function calls below are stored in the same folder.

# X_artificial will be created by referencing the files in this filder.



# generate 1000 images with a random number of cells (1 to 100) on each image

generate_artificial_images(num_images_wanted=2000, min_cells_on_image=1,

                           max_cells_on_image=100, fname='imagesA')



# generate 1000 images with 1 to 14 cells on the image

generate_artificial_images(num_images_wanted=1000, min_cells_on_image=1,

                           max_cells_on_image=14, fname='imagesB')
# check that the artificial images exist



len(os.listdir('new_images_dir'))
image_list = os.listdir('new_images_dir')





# set up the canvas for the subplots

plt.figure(figsize=(10,10))

plt.axis('Off')



# Our subplot will contain 3 rows and 3 columns

# plt.subplot(nrows, ncols, plot_number)





# == row 1 ==



# image

plt.subplot(3,3,1)



fname = image_list[1]

path = 'new_images_dir/' + fname

name = fname.split('_')

num_cells = 'Num Cells: ' + str(name[2])



image = cv2.imread(path)

plt.imshow(image)

plt.title(num_cells, fontsize=14)

plt.axis('off')



# image

plt.subplot(3,3,2)



fname = image_list[2]

path = 'new_images_dir/' + fname

name = fname.split('_')

num_cells = 'Num Cells: ' + str(name[2])



image = cv2.imread(path)

plt.imshow(image)

plt.title(num_cells, fontsize=14)

plt.axis('off')





# image

plt.subplot(3,3,3)



fname = image_list[3]

path = 'new_images_dir/' + fname

name = fname.split('_')

num_cells = 'Num Cells: ' + str(name[2])



image = cv2.imread(path)

plt.imshow(image)

plt.title(num_cells, fontsize=14)

plt.axis('off')



# == row 2 ==



# image

plt.subplot(3,3,4)



fname = image_list[4]

path = 'new_images_dir/' + fname

name = fname.split('_')

num_cells = 'Num Cells: ' + str(name[2])



image = cv2.imread(path)

plt.imshow(image)

plt.title(num_cells, fontsize=14)

plt.axis('off')



# image

plt.subplot(3,3,5)



fname = image_list[5]

path = 'new_images_dir/' + fname

name = fname.split('_')

num_cells = 'Num Cells: ' + str(name[2])



image = cv2.imread(path)

plt.imshow(image)

plt.title(num_cells, fontsize=14)

plt.axis('off')



# image

plt.subplot(3,3,6)



fname = image_list[6]

path = 'new_images_dir/' + fname

name = fname.split('_')

num_cells = 'Num Cells: ' + str(name[2])



image = cv2.imread(path)

plt.imshow(image)

plt.title(num_cells, fontsize=14)

plt.axis('off')



# == row 3 ==



# image

plt.subplot(3,3,7)



fname = image_list[7]

path = 'new_images_dir/' + fname

name = fname.split('_')

num_cells = 'Num Cells: ' + str(name[2])



image = cv2.imread(path)

plt.imshow(image)

plt.title(num_cells, fontsize=14)

plt.axis('off')



# image

plt.subplot(3,3,8)



fname = image_list[8]

path = 'new_images_dir/' + fname

name = fname.split('_')

num_cells = 'Num Cells: ' + str(name[2])



image = cv2.imread(path)

plt.imshow(image)

plt.title(num_cells, fontsize=14)

plt.axis('off')





# image

plt.subplot(3,3,9)



fname = image_list[9]

path = 'new_images_dir/' + fname

name = fname.split('_')

num_cells = 'Num Cells: ' + str(name[2])



image = cv2.imread(path)

plt.imshow(image)

plt.title(num_cells, fontsize=14)

plt.axis('off')



plt.tight_layout()

plt.show()
# X_train



# Get lists of images and their masks.

art_id_list = os.listdir('new_images_dir')



# Create empty arrays

# It's very important to set the datatype as dtype=np.uint8 or when

# we print an image from X_artificial there will be snow around the cells.

X_artificial = np.zeros((len(art_id_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

#X_artificial = np.zeros((len(art_id_list), 520, 696, 3), dtype=np.uint8)

y_artificial = np.zeros(len(art_id_list))



for i, mask_id in enumerate(art_id_list):

    

    path_mask = 'new_images_dir/' + mask_id

    

    # read the file as an array

    cv2_image = cv2.imread(path_mask)

    # resize the image

    cv2_image = cv2.resize(cv2_image, (IMG_HEIGHT, IMG_WIDTH))

    # save the image at the destination as a jpg file

    cv2.imwrite('mask.jpg', cv2_image)

    

    # read the image using skimage

    mask = imread('mask.jpg')

    

    # resize the image

    #mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    

    # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)

    #mask = np.expand_dims(mask, axis=-1)

    

    # insert the image into X_train

    X_artificial[i] = mask



    

    

    # y_artificial

    

    # extract the number of cells from the file name

    name = mask_id.split('_')

    num_cells = int(name[2])

    

    y_artificial[i] = num_cells

    



print(X_artificial.shape)

print(y_artificial.shape)
img = X_artificial[1,:,:,:]



plt.imshow(img)
test_id_list = ['SIMCEPImages_A15_C61_F1_s03_w2.TIF',

 'SIMCEPImages_A21_C87_F1_s03_w1.TIF',

 'SIMCEPImages_A01_C1_F1_s02_w1.TIF',

 'SIMCEPImages_A04_C14_F1_s05_w1.TIF',

 'SIMCEPImages_A18_C74_F1_s01_w1.TIF',

 'SIMCEPImages_A04_C14_F1_s18_w1.TIF',

 'SIMCEPImages_A18_C74_F1_s09_w2.TIF',

 'SIMCEPImages_A13_C53_F1_s10_w2.TIF',

 'SIMCEPImages_A08_C31_F1_s13_w2.TIF',

 'SIMCEPImages_A19_C78_F1_s15_w2.TIF']



num_cells = [61, 87, 1, 14, 74, 14, 74, 53, 31, 78]





# test_id_list and num_cells were defined in the introduction section.

df_test = pd.DataFrame(test_id_list, columns=['mask_id'])



# add a new column with the number of cells on each mask

df_test['num_cells'] = num_cells



# Reset the index.

# This is so that we can use loc to access mask id's later.

df_test = df_test.reset_index(drop=True)





# Select only rows that are not part of the test set.

# Note the use of ~ to execute 'not in'.

df_masks = df_masks[~df_masks['mask_id'].isin(test_id_list)]



print(df_masks.shape)

print(df_test.shape)
df_test.head()
# X_train



# Get lists of images and their masks.

mask_id_list = list(df_masks['mask_id'])



# Create empty arrays

X_train = np.zeros((len(mask_id_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)





for i, mask_id in enumerate(mask_id_list):

    

    path_mask = '../input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth/' + mask_id

    

    # read the file as an array

    cv2_image = cv2.imread(path_mask)

    # resize the image

    cv2_image = cv2.resize(cv2_image, (IMG_HEIGHT, IMG_WIDTH))

    # save the image at the destination as a jpg file

    cv2.imwrite('mask.jpg', cv2_image)

    

    # read the image using skimage

    mask = imread('mask.jpg')

    

    # resize the image

    #mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    

    # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)

    #mask = np.expand_dims(mask, axis=-1)

    

    # insert the image into X_Train

    X_train[i] = mask



    

    

# y_train



y_train = df_masks['num_cells'] #.astype(np.float16)



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)
X_train[1, :, :, :].shape
X_train.shape
X_artificial.shape
X_train = np.vstack((X_train, X_artificial))



y_train = np.hstack((y_train, y_artificial))



print(X_train.shape)

print(y_train.shape)
kernel_size = (3,3)

pool_size= (2,2)

first_filters = 32

second_filters = 64

third_filters = 128



dropout_conv = 0.3

dropout_dense = 0.3





model = Sequential()



# Input layer for rgb image. For grayscale image use the same channel 3 times

# to maintain the shape that the model requires.

model.add(Conv2D(first_filters, kernel_size, activation = 'relu', 

                 input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))



model.add(ZeroPadding2D(padding=(PADDING, PADDING), data_format=None))



model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))

model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))

model.add(MaxPooling2D(pool_size = pool_size)) 

model.add(Dropout(dropout_conv))



model.add(Conv2D(second_filters, kernel_size, activation ='relu'))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))

model.add(MaxPooling2D(pool_size = pool_size))

model.add(Dropout(dropout_conv))



model.add(Conv2D(third_filters, kernel_size, activation ='relu'))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))

model.add(MaxPooling2D(pool_size = pool_size))

model.add(Dropout(dropout_conv))



model.add(Conv2D(third_filters, kernel_size, activation ='relu'))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))

model.add(MaxPooling2D(pool_size = pool_size))

model.add(Dropout(dropout_conv))



model.add(Flatten())



model.add(Dense(1024))

model.add(LeakyReLU())

model.add(BatchNormalization())



model.add(Dense(512))

model.add(LeakyReLU())

model.add(BatchNormalization())



model.add(Dense(1, activation='relu')) # set activation='relu' to keep all values positive



model.summary()
model.compile(Adam(lr=0.001), loss='mean_squared_error', 

              metrics=['mse'])



filepath = "model.h5"



earlystopper = EarlyStopping(patience=15, verbose=1)



checkpoint = ModelCheckpoint(filepath, monitor='val_mean_squared_error', verbose=1, 

                             save_best_only=True, mode='min')



callbacks_list = [earlystopper, checkpoint]



history = model.fit(X_train, y_train, validation_split=0.1, batch_size=16, epochs=100, 

                    callbacks=callbacks_list)
# display the loss and accuracy curves



import matplotlib.pyplot as plt



mean_squared_error = history.history['mean_squared_error']

val_mean_squared_error = history.history['val_mean_squared_error']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(mean_squared_error) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.figure()



plt.plot(epochs, mean_squared_error, 'bo', label='Training mse')

plt.plot(epochs, val_mean_squared_error, 'b', label='Validation mse')

plt.title('Training and validation mse')

plt.legend()

plt.figure()
# use the best epoch

model.load_weights('model.h5')



preds = model.predict(X_test)

preds
# add the preds to df_test

df_test['preds'] = np.round(preds)



# change the preds to integers to improve the look of the displayed results

df_test['preds'] = df_test['preds'].apply(np.int)



# create a dataframe caled df_results

df_results = df_test[['mask_id', 'num_cells', 'preds']]



# add a new column with the difference between the true and predicted values.

df_results['difference'] = abs(df_results['num_cells'] - df_results['preds'])

df_results.head(10)
# What was the max difference?



max_diff = df_results['difference'].max()

min_diff = df_results['difference'].min()



print('Max Error: ', max_diff)

print('Min Error: ', min_diff)
# =====================    

# Display the masks

# =====================  



# set up the canvas for the subplots

plt.figure(figsize=(10,10))

plt.axis('Off')



# Our subplot will contain 3 rows and 3 columns

# plt.subplot(nrows, ncols, plot_number)





# == row 1 ==



# image

plt.subplot(3,3,1)

test_image = X_test[1, :, :, 0]

plt.imshow(test_image)



true = df_results.loc[1, 'num_cells']

pred = df_results.loc[1, 'preds']

result = 'True: ' + str(true) + ' Pred: ' + str(pred)



plt.title(result, fontsize=14)

plt.axis('off')



# image

plt.subplot(3,3,2)

test_image = X_test[2, :, :, 0]

plt.imshow(test_image)



true = df_results.loc[2, 'num_cells']

pred = df_results.loc[2, 'preds']

result = 'True: ' + str(true) + ' Pred: ' + str(pred)



plt.title(result, fontsize=14)

plt.axis('off')





# image

plt.subplot(3,3,3)

test_image = X_test[3, :, :, 0]

plt.imshow(test_image)



true = df_results.loc[3, 'num_cells']

pred = df_results.loc[3, 'preds']

result = 'True: ' + str(true) + ' Pred: ' + str(pred)



plt.title(result, fontsize=14)

plt.axis('off')





# == row 2 ==



# image

plt.subplot(3,3,4)

test_image = X_test[4, :, :, 0]

plt.imshow(test_image)



true = df_results.loc[4, 'num_cells']

pred = df_results.loc[4, 'preds']

result = 'True: ' + str(true) + ' Pred: ' + str(pred)



plt.title(result, fontsize=14)

plt.axis('off')



# image

plt.subplot(3,3,5)

test_image = X_test[5, :, :, 0]

plt.imshow(test_image)



true = df_results.loc[5, 'num_cells']

pred = df_results.loc[5, 'preds']

result = 'True: ' + str(true) + ' Pred: ' + str(pred)



plt.title(result, fontsize=14)

plt.axis('off')



# image

plt.subplot(3,3,6)

test_image = X_test[6, :, :, 0]

plt.imshow(test_image)



true = df_results.loc[6, 'num_cells']

pred = df_results.loc[6, 'preds']

result = 'True: ' + str(true) + ' Pred: ' + str(pred)



plt.title(result, fontsize=14)

plt.axis('off')





# == row 3 ==



# image

plt.subplot(3,3,7)

test_image = X_test[7, :, :, 0]

plt.imshow(test_image)



true = df_results.loc[7, 'num_cells']

pred = df_results.loc[7, 'preds']

result = 'True: ' + str(true) + ' Pred: ' + str(pred)



plt.title(result, fontsize=14)

plt.axis('off')



# image

plt.subplot(3,3,8)

test_image = X_test[8, :, :, 0]

plt.imshow(test_image)



true = df_results.loc[8, 'num_cells']

pred = df_results.loc[8, 'preds']

result = 'True: ' + str(true) + ' Pred: ' + str(pred)



plt.title(result, fontsize=14)

plt.axis('off')



# image

plt.subplot(3,3,9)

test_image = X_test[9, :, :, 0]

plt.imshow(test_image)



true = df_results.loc[9, 'num_cells']

pred = df_results.loc[9, 'preds']

result = 'True: ' + str(true) + ' Pred: ' + str(pred)



plt.title(result, fontsize=14)

plt.axis('off')





plt.tight_layout()

plt.show()

# Kaggle allows a max of 500 files to be saved.



if os.path.isdir('new_images_dir') == True: # return true if the directory exists

    

    shutil.rmtree('new_images_dir')

        

if os.path.isdir('base_dir') == True: # return true if the directory exists

    

    shutil.rmtree('base_dir')