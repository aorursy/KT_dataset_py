# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#identify and set up directories

print(os.getcwd())

work_dir = os.getcwd()



chp_id = "cnn"

print(work_dir)
import pandas as pd  # data frame operations  

import sklearn

import plotly

import plotly.graph_objs as go

import time

import numpy as np

import os

import sys

import re # regular expressions

import scipy

import cv2

import seaborn as sns  # pretty plotting, including heat map

from functools import partial

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from matplotlib.backends.backend_pdf import PdfPages





# Python ≥3.5 is required

assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required

assert sklearn.__version__ >= "0.20"





# TensorFlow ≥2.0 is required

import tensorflow as tf

from tensorflow import keras

from keras import backend as K

assert tf.__version__ >= "2.0"



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



#Set enviorment varaibles

random_seed=1
!unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip -d train
!unzip ../input/dogs-vs-cats-redux-kernels-edition/test.zip -d test
outdir = work_dir +'/cats_dogs_arrays'

os.mkdir('/kaggle/working/cats_dogs_arrays')

print(outdir)
image_dir_name = work_dir +'/train/train'

print(image_dir_name)
os.mkdir('/kaggle/working/tmp')
#Standard Functions

# Sorting of file names facilitated by

def tryint(s):

    try:

        return int(s)

    except:

        return s



def alphanum_key(s):

    """ Turn a string into a list of string and number chunks.

        "z23a" -> ["z", 23, "a"]

    """

    return [ tryint(c) for c in re.split('([0-9]+)', s) ]



def sort_nicely(l):

    """ Sort the given list in the way that humans expect.

    """

    l.sort(key=alphanum_key)

    

# Generate list of file names, excluding hidden files    

def directory_list (dir_name,str1):

    start_list = os.listdir(dir_name)

    end_list = []

    for file in start_list:

        if (not file.startswith(str1)):

            end_list.append(file) 

    end_list.sort(key = alphanum_key)        

    return(end_list)        



cat_file_names = directory_list(image_dir_name, "cat")

dog_file_names = directory_list(image_dir_name, "dog") 
test_image_dir_name = work_dir +'/test/test'

print(test_image_dir_name)
#S3a Define Function to Create CNN - Soure Geron Chap14

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):

    path = os.path.join(work_dir +'/', fig_id + "." + fig_extension)

    print("Saving figure", fig_id)

    if tight_layout:

        plt.tight_layout()

    plt.savefig(path, format=fig_extension, dpi=resolution)



def plot_image(image):

    plt.imshow(image, cmap="gray", interpolation="nearest")

    plt.axis("off")



def plot_color_image(image):

    plt.imshow(image, interpolation="nearest")

    plt.axis("off")



def feature_map_size(input_size, kernel_size, strides=1, padding="SAME"):

    if padding == "SAME":

        return (input_size - 1) // strides + 1

    else:

        return (input_size - kernel_size) // strides + 1



def dist_plot(var1, var2, var3):

    tmp_plt=sns.countplot(var1, palette="Blues").set_title(var2)

    tmp_fig = tmp_plt.get_figure()

    tmp_fig.savefig(var3 + ".png", 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25)

    return(tmp_plt)



def pad_before_and_padded_size(input_size, kernel_size, strides=1):

    fmap_size = feature_map_size(input_size, kernel_size, strides)

    padded_size = max((fmap_size - 1) * strides + kernel_size, input_size)

    pad_before = (padded_size - input_size) // 2

    return pad_before, padded_size



def manual_same_padding(images, kernel_size, strides=1):

    if kernel_size == 1:

        return images.astype(np.float32)

    batch_size, height, width, channels = images.shape

    top_pad, padded_height = pad_before_and_padded_size(height, kernel_size, strides)

    left_pad, padded_width  = pad_before_and_padded_size(width, kernel_size, strides)

    padded_shape = [batch_size, padded_height, padded_width, channels]

    padded_images = np.zeros(padded_shape, dtype=np.float32)

    padded_images[:, top_pad:height+top_pad, left_pad:width+left_pad, :] = images

    return padded_images

#Tensorboard Logs

root_logdir = os.path.join(os.curdir, "tf_logs")

def get_run_logdir():

    import time

    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")

    return os.path.join(root_logdir, run_id)
print(len(dog_file_names))

print(len(cat_file_names))
#   Convert image to numpy array. 3 channels for color  and 1 converted to grayscale

#   Info on npy binary format for saving numpy arrays https://towardsdatascience.com/

def parse_grayscale(image_file_path):

    image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)

    return(image)

    

def parse_color(image_file_path):

    image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)

    # Default cv2 is BGR... need RGB

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return(image)

  

def parse_grayscale_and_resize(image_file_path, size = (64, 64)):

    image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, size)

    return(image)



def parse_color_and_resize(image_file_path, size = (64, 64)):

    image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)

    # Default cv2 is BGR... need RGB

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, size)

    return(image)  

    

def show_grayscale_image(image):

    plt.imshow(image, cmap = 'gray') 

    plt.axis('off')

    plt.show()



def show_color_image(image):

    plt.imshow(image) 

    plt.axis('off')

    plt.show()   
#Examine dimensions of original raster images 

print(len(dog_file_names))

print(len(cat_file_names))

cats_shapes = []

for ifile in range(len(cat_file_names)):

    image_file_path = os.path.join(image_dir_name, cat_file_names[ifile])

    image = parse_color(image_file_path)

    cats_shapes.append(image.shape)

#print('\n\nCat image file shapes:\n')    

#print(cats_shapes)    



dogs_shapes = []

for ifile in range(len(dog_file_names)):

    image_file_path = os.path.join(image_dir_name, dog_file_names[ifile])

    image = parse_color(image_file_path)

    dogs_shapes.append(image.shape)    

#print('\n\nDog image file shapes:\n') 

#print(dogs_shapes)
# Create Numpy Image Arrays

#----------------------------------------------------------------------

print('\nProcessing image files to 64x64 color or grayscale arrays')



# Create cats_1000_64_64_1 and numpy array for 12500 cat images in grayscale

cats_1000_64_64_1 = np.zeros((12500, 64, 64, 1))  

for ifile in range(len(cat_file_names)):

    image_file_path = os.path.join(image_dir_name, cat_file_names[ifile])

    image = parse_grayscale_and_resize(image_file_path, size = (64, 64))

    cats_1000_64_64_1[ifile,:,:,0] = image

       

# Create dogs_1000_64_64_1 and numpy array for 12500 dog images in grayscale   

dogs_1000_64_64_1 = np.zeros((12500, 64, 64, 1))  

for ifile in range(len(dog_file_names)):

    image_file_path = os.path.join(image_dir_name, dog_file_names[ifile])

    image = parse_grayscale_and_resize(image_file_path, size = (64, 64))

    dogs_1000_64_64_1[ifile,:,:,0] = image
np.save(os.path.join(outdir, 'cats_1000_64_64_1.npy'), cats_1000_64_64_1)

np.save(os.path.join(outdir, 'dogs_1000_64_64_1.npy'), dogs_1000_64_64_1)
#get the final test data ready here

#want to see if there is enough space



path = test_image_dir_name

#os.listdir(path)



X_test1 = []

id_line = []

def create_test1_data(path):

    for p in os.listdir(path):

        id_line.append(p.split(".")[0])

        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

        new_img_array = cv2.resize(img_array, dsize=(64, 64))

        X_test1.append(new_img_array)

create_test1_data(path)

X_test1 = np.array(X_test1).reshape(-1,64,64,1)

X_test1 = X_test1/255
os.listdir(outdir) # returns list
RANDOM_SEED=1



#Reset Graphs for Tensorboard

def reset_graph(seed= RANDOM_SEED):

    tf.reset_default_graph()

    tf.set_random_seed(seed)

    np.random.seed(seed)

    

    

#Save images to working directory

def save_fig(fig_id, tight_layout=True):

    path = os.path.join(work_dir, "images", chp_id, fig_id + ".png")

    print("Saving figure", fig_id)

    if tight_layout:

        plt.tight_layout()

    plt.savefig(path, format='png', dpi=300)

    



#Randomly Sort Batches

def shuffle_batch(X, y, batch_size):

    rnd_idx = np.random.permutation(len(X))

    n_batches = len(X) // batch_size

    for batch_idx in np.array_split(rnd_idx, n_batches):

        X_batch, y_batch = X[batch_idx], y[batch_idx]

        yield X_batch, y_batch

        



from matplotlib import pyplot as plt  # for display of images

def show_grayscale_image(image):

    plt.imshow(image, cmap='gray')

    plt.axis('off')

    plt.show()

    

    

#Check distribtion of test , valid and train

def dist_plot(var1, var2, var3):

    tmp_plt=sns.countplot(var1, palette="Blues").set_title(var2)

    tmp_fig = tmp_plt.get_figure()

    tmp_fig.savefig(var3 + ".png", 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25, frameon=None)

    return(tmp_plt)

  

  #Optimize memory

def get_model_params():

    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}



def restore_model_params(model_params):

    gvar_names = list(model_params.keys())

    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")

                  for gvar_name in gvar_names}

    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}

    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}

    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

        

#S4 Set enviorment varaibles



height = 64

width = 64  
# CatsDogs  dataset # 

# Documentation on npy binary format for saving numpy arrays for later use

#     https://towardsdatascience.com/why-you-should-start-using-npy-file-more-often-df2a13cc0161

# Under the working directory, data files are in directory cats_dogs_64_128 

# Read in cats and dogs grayscale 64x64 files to create training data

cats_1000_64_64_1 = np.load(outdir+'/cats_1000_64_64_1.npy')

dogs_1000_64_64_1 = np.load(outdir+'/dogs_1000_64_64_1.npy')



print("Shape of cat data: ",cats_1000_64_64_1.shape)

print("Shape of dog data: ",dogs_1000_64_64_1.shape)
# Examine first cat and first dog grayscale images

show_grayscale_image(cats_1000_64_64_1[0,:,:,0])

show_grayscale_image(dogs_1000_64_64_1[0,:,:,0])
#S6 Create modeling dataset - stack cat and dog array

X_cat_dog= np.concatenate((cats_1000_64_64_1, dogs_1000_64_64_1), axis = 0) 

#Drop last column in array will add back after scaling process

X_cat_dog=X_cat_dog[:,:,:,-1]

X_cat_dog.shape



#Assign labels

y_cat_dog = np.concatenate((np.zeros((12500), dtype = np.int32), 

                            np.ones((12500), dtype = np.int32)), axis = 0)

#S7 Split Train, Validate and Test

X_train, X_test_ds, y_train, y_test_ds= train_test_split(X_cat_dog, y_cat_dog, 

                                                         test_size=0.5, random_state= random_seed)

X_test, X_valid, y_test, y_valid = train_test_split(X_test_ds, y_test_ds, 

                                                    test_size=0.30, random_state = random_seed)

#S8 Scale images/numpy array

X_mean = X_train.mean(axis=0, keepdims=True)

X_std = X_train.std(axis=0, keepdims=True) + 1e-7

X_train = (X_train - X_mean) / X_std

X_valid = (X_valid - X_mean) / X_std

X_test = (X_test - X_mean) / X_std



X_train = X_train[..., np.newaxis]

X_valid = X_valid[..., np.newaxis]

X_test = X_test[..., np.newaxis]



#Review Distribution

print(X_train.shape)

print(X_test.shape)

print(X_valid.shape)

print(y_train.shape)

print(y_test.shape)

print(y_valid.shape)
#S9 Check distribtion of test , valid and train

cd_plt_trn=dist_plot(y_train, 'Train', "TrainDistCatDog")

cd_plt_trn.get_figure().show()

cd_plt_tst=dist_plot(y_test, 'Test', "TestDistCatDog")

cd_plt_tst.get_figure().show()
cd_plt_vld=dist_plot(y_valid, 'Valid', "ValidDistCatDog")

cd_plt_vld.get_figure().show()
#Compile Model 1

model = keras.models.Sequential([

    keras.layers.Conv2D(filters=64, kernel_size=7, activation='relu', padding='SAME', input_shape=[64, 64, 1]),

    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='SAME'),

    keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='SAME'),

    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='SAME'),

    keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='SAME'),

    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Flatten(),

    keras.layers.Dense(units=128, activation='relu'),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(units=64, activation='relu'),

    keras.layers.Dropout(0.5),

    #keras.layers.Dense(units=2, activation='softmax'),

    keras.layers.Dense(1, activation='sigmoid'),

])
#S11 Clear and Reset log

keras.backend.clear_session()

np.random.seed(1)

tf.random.set_seed(1)

#Reset Log Directory

run_logdir = get_run_logdir()
# Execution with early Stopping Model 1

start_time_train = time.process_time()

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

checkpoint_cb = keras.callbacks.ModelCheckpoint(work_dir+"/tmp/my_keras_model.h5", save_best_only=True)

early_stopping_cb=keras.callbacks.EarlyStopping(monitor='val_loss', mode ='min', min_delta=1, patience = 75)

#optimizer = keras.optimizers.Nadam(lr=1e-4, beta_1=0.9, beta_2=0.999)

optimizer = keras.optimizers.RMSprop(lr=1e-4, rho=0.9)

n_epochs = 100



model.compile(loss='binary_crossentropy', optimizer =optimizer, metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=n_epochs, 

                    validation_data=[X_test, y_test],

                    callbacks=[checkpoint_cb, tensorboard_cb, early_stopping_cb])

score = model.evaluate(X_valid, y_valid)

X_new = X_test[:10] # pretend we have new images

y_pred = model.predict(X_new)

end_time_train = time.process_time()

m1_time_train = end_time_train-start_time_train

print(m1_time_train)
#Model Summary 

model.summary()
# View History

history.params

pd.DataFrame(history.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0, 1)

#save_fig("keras_learning_curves_plot")

plt.show()
# Create Predicted Probabilties

y_proba = model.predict(X_valid)

y_proba.round(2)
#Create Predicted Value

y_pred = model.predict_classes(X_valid)
#View actual to predicted

print("Predicted classes:", np.reshape(y_pred[:20], (1, 20)))

print("Actual classes:   ", y_valid[:20])
#Create Kaggle Submission

predictions = model.predict(X_test1)

predicted_val = [int(round(p[0])) for p in predictions]

submission_df = pd.DataFrame({'id':id_line, 'label':predicted_val})

submission_df.to_csv("submission1.csv", index=False)
#define swish

def swish(x):

    return K.sigmoid(x) * x
#Model #2

#same as model 1, but replace 'relu' with swish function

model2 = keras.models.Sequential([

    keras.layers.Conv2D(filters=64, kernel_size=7, activation=swish, padding='SAME', input_shape=[64, 64, 1]),

    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Conv2D(filters=128, kernel_size=3, activation=swish, padding='SAME'),

    keras.layers.Conv2D(filters=128, kernel_size=3, activation=swish, padding='SAME'),

    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Conv2D(filters=256, kernel_size=3, activation=swish, padding='SAME'),

    keras.layers.Conv2D(filters=256, kernel_size=3, activation=swish, padding='SAME'),

    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Flatten(),

    keras.layers.Dense(units=128, activation=swish),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(units=64, activation=swish),

    keras.layers.Dropout(0.5),

    #keras.layers.Dense(units=2, activation='softmax'),

    keras.layers.Dense(1, activation='sigmoid'),

])
# Clear and Reset log Model #2

keras.backend.clear_session()

np.random.seed(1)

tf.random.set_seed(1)

#Reset Log Directory

run_logdir = get_run_logdir()
# Execution with early Stopping Model #2

start_time_train = time.process_time()

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

checkpoint_cb = keras.callbacks.ModelCheckpoint(work_dir+"/tmp/my_keras_model.h5", save_best_only=True)

early_stopping_cb=keras.callbacks.EarlyStopping(monitor='val_loss', mode ='min', min_delta=1, patience=75)

#optimizer = keras.optimizers.Nadam(lr=1e-4, beta_1=0.9, beta_2=0.999)

optimizer = keras.optimizers.RMSprop(lr=1e-4, rho=0.9)

n_epochs = 100



model2.compile(loss='binary_crossentropy', optimizer =optimizer, metrics=["accuracy"])

history = model2.fit(X_train, y_train, epochs=n_epochs, 

                    validation_data=[X_test, y_test],

                    callbacks=[checkpoint_cb, tensorboard_cb, early_stopping_cb])

score2 = model2.evaluate(X_valid, y_valid)

X_new = X_test[:10] # pretend we have new images

y_pred_2 = model2.predict(X_new)

end_time_train = time.process_time()

m2_time_train = end_time_train-start_time_train

print(m2_time_train)
#Model Summary 

model2.summary()
# View History Model #2

history.params

pd.DataFrame(history.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0, 1)

#save_fig("keras_learning_curves_plot")

plt.show()
#S14 Create Predicted Probabilties Model #2

y_proba2 = model2.predict(X_valid)

y_proba2.round(2)
#Create Predicted Value Model #2

y_pred2 = model2.predict_classes(X_valid)
#View actual to predicted Model #2

print("Predicted classes:", np.reshape(y_pred2[:20], (1, 20)))

print("Actual classes:   ", y_valid[:20])
#Create Kaggle Submission Model #2

predictions = model2.predict(X_test1)

predicted_val = [int(round(p[0])) for p in predictions]

submission_df = pd.DataFrame({'id':id_line, 'label':predicted_val})

submission_df.to_csv("submission2.csv", index=False)
#Compile Model 3 same as model 1, but with strides = 2 instead of pooling

model3 = keras.models.Sequential([

    keras.layers.Conv2D(filters=64, kernel_size=7,strides=(2, 2), activation='relu', padding='SAME', input_shape=[64, 64, 1]),

    #keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Conv2D(filters=128, kernel_size=3,strides=(2, 2), activation='relu', padding='SAME'),

    keras.layers.Conv2D(filters=128, kernel_size=3,strides=(2, 2), activation='relu', padding='SAME'),

    #keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2, 2),activation='relu', padding='SAME'),

    keras.layers.Conv2D(filters=256, kernel_size=3,strides=(2, 2), activation='relu', padding='SAME'),

    #keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Flatten(),

    keras.layers.Dense(units=128, activation='relu'),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(units=64, activation='relu'),

    keras.layers.Dropout(0.5),

    #keras.layers.Dense(units=2, activation='softmax'),

    keras.layers.Dense(1, activation='sigmoid'),

])
#S11 Clear and Reset log

keras.backend.clear_session()

np.random.seed(1)

tf.random.set_seed(1)

#Reset Log Directory

run_logdir = get_run_logdir()
# Execution with early Stopping Model 3

start_time_train = time.process_time()

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

checkpoint_cb = keras.callbacks.ModelCheckpoint(work_dir+"/tmp/my_keras_model.h5", save_best_only=True)

early_stopping_cb=keras.callbacks.EarlyStopping(monitor='val_loss', mode ='min', min_delta=1, patience=75)

#optimizer = keras.optimizers.Nadam(lr=1e-4, beta_1=0.9, beta_2=0.999)

optimizer = keras.optimizers.RMSprop(lr=1e-4, rho=0.9)

n_epochs = 100



model3.compile(loss='binary_crossentropy', optimizer =optimizer, metrics=["accuracy"])

history = model3.fit(X_train, y_train, epochs=n_epochs, 

                    validation_data=[X_test, y_test],

                    callbacks=[checkpoint_cb, tensorboard_cb, early_stopping_cb])

score3 = model3.evaluate(X_valid, y_valid)

X_new = X_test[:10] # pretend we have new images

y_pred3 = model3.predict(X_new)

end_time_train = time.process_time()

m3_time_train = end_time_train-start_time_train

print(m3_time_train)
#Model Summary 

model3.summary()
# View History

history.params

pd.DataFrame(history.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0, 1)

#save_fig("keras_learning_curves_plot")

plt.show()
# Create Predicted Probabilties Model #3

y_proba3 = model3.predict(X_valid)

y_proba3.round(2)
#Create Predicted Value Model #3

y_pred3 = model3.predict_classes(X_valid)
#View actual to predicted Model #3

print("Predicted classes:", np.reshape(y_pred3[:20], (1, 20)))

print("Actual classes:   ", y_valid[:20])
#Create Kaggle Submission Model #3

predictions = model3.predict(X_test1)

predicted_val = [int(round(p[0])) for p in predictions]

submission_df = pd.DataFrame({'id':id_line, 'label':predicted_val})

submission_df.to_csv("submission3.csv", index=False)
#Model 4 same as Model 2, swish, but with 2 strides, no pooling

model4 = keras.models.Sequential([

    keras.layers.Conv2D(filters=64, kernel_size=7, strides=(2, 2),activation=swish, padding='SAME', input_shape=[64, 64, 1]),

    #keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2),activation=swish, padding='SAME'),

    keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2),activation=swish, padding='SAME'),

    #keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2, 2),activation=swish, padding='SAME'),

    keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2, 2),activation=swish, padding='SAME'),

    #keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Flatten(),

    keras.layers.Dense(units=128, activation=swish),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(units=64, activation=swish),

    keras.layers.Dropout(0.5),

    #keras.layers.Dense(units=2, activation='softmax'),

    keras.layers.Dense(1, activation='sigmoid'),

])
# Clear and Reset log Model #4

keras.backend.clear_session()

np.random.seed(1)

tf.random.set_seed(1)

#Reset Log Directory

run_logdir = get_run_logdir()
# Execution with early Stopping Model #4

start_time_train = time.process_time()

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

checkpoint_cb = keras.callbacks.ModelCheckpoint(work_dir+"/tmp/my_keras_model.h5", save_best_only=True)

early_stopping_cb=keras.callbacks.EarlyStopping(monitor='val_loss', mode ='min', min_delta=1, patience=75)

#optimizer = keras.optimizers.Nadam(lr=1e-4, beta_1=0.9, beta_2=0.999)

optimizer = keras.optimizers.RMSprop(lr=1e-4, rho=0.9)

n_epochs = 100



model4.compile(loss='binary_crossentropy', optimizer =optimizer, metrics=["accuracy"])

history = model4.fit(X_train, y_train, epochs=n_epochs, 

                    validation_data=[X_test, y_test],

                    callbacks=[checkpoint_cb, tensorboard_cb, early_stopping_cb])

score4 = model4.evaluate(X_valid, y_valid)

X_new = X_test[:10] # pretend we have new images

y_pred_4 = model4.predict(X_new)

end_time_train = time.process_time()

m4_time_train = end_time_train-start_time_train

print(m4_time_train)
#Model Summary 

model4.summary()
# View History Model #4

history.params

pd.DataFrame(history.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0, 1)

#save_fig("keras_learning_curves_plot")

plt.show()
# Create Predicted Probabilties Model #4

y_proba4 = model4.predict(X_valid)

y_proba4.round(2)
#Create Predicted Value Model #4

y_pred4 = model4.predict_classes(X_valid)
#View actual to predicted Model #4

print("Predicted classes:", np.reshape(y_pred4[:20], (1, 20)))

print("Actual classes:   ", y_valid[:20])
#Create Kaggle Submission Model #4

predictions = model4.predict(X_test1)

predicted_val = [int(round(p[0])) for p in predictions]

submission_df = pd.DataFrame({'id':id_line, 'label':predicted_val})

submission_df.to_csv("submission4.csv", index=False)
#print model train times and scores

print('Model 1 Time = ',m1_time_train, ', Score = ', score)

print('Model 2 Time = ',m2_time_train, ', Score = ', score2)

print('Model 3 Time = ',m3_time_train, ', Score = ', score3)

print('Model 4 Time = ',m4_time_train, ', Score = ', score4)