from numpy.random import seed
seed(101)
from tensorflow import set_random_seed
set_random_seed(101)

import pandas as pd
import numpy as np
import math
import pydicom
import pylab
import os
import pickle

from sklearn.model_selection import train_test_split
from skimage.transform import resize

import matplotlib.pyplot as plt
%matplotlib inline


# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')

import gc; gc.enable()

os.listdir('../input')
# load the pickled dataframes

df_train = pickle.load(open('../input/python-generators-to-reduce-ram-usage-part-1/dftrain.pickle','rb'))
df_test = pickle.load(open('../input/python-generators-to-reduce-ram-usage-part-1/dftest.pickle','rb'))


print(df_train.shape)
print(df_test.shape)
# Source: https://www.kaggle.com/peterchang77/exploratory-data-analysis

def parse_data(df):
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:

      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': '../input/stage_1_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed
# define a function to output a row containing all box info incl. confidence scores

def create_bounding_rows(df_train):
    
    """
    Takes each patientId and creates a row of combined bounding boxes and 
    also includes their confidence scores. All patientId's are 
    included in one matrix.
    This fuction is based on a max of 4 bounding boxes per patientId.
    Output: Numpy matrix of shape(len(df_train), 20) 
    """
    
    # read in the dataframe that will be parsed by the function parse_data(df)
    df_boxes = \
    pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_1_train_labels.csv')

    
    # set the length depending on how many bounding boxes we want the model to output
    length = 20
    
    h = np.ones(20)
    k = np.zeros(20)

    # create an empty numpy matrix matching the size of the output matrix
    y = np.zeros((len(df_train),length))

    # run the function
    # this must be here because this must be run each time this script is run or
    # the resulting matrix will have errors.
    parsed = parse_data(df_boxes)


    for i in range(0,len(df_train)):

        # get the patientId
        patientId = df_train.loc[i, 'patientId']

        # extract the bounding boxes for a particular patient
        box = parsed[patientId]['boxes']
        if len(box) == 0:

            # the first row becomes a dummy row of ones this must be deleted later
            # k is an array of zeros
            h = np.vstack((h,k))

        if len(box) != 0:


            # insert 1 as the first entry in each bounding box
            # the 1 represents confidence for that bounding box
            a=[]
            for i in range(0,len(box)):
                box[i].insert(0,1)
                a = a + box[i]

            # calculate how much padding to add
            b = length - len(a)

            # pad the list because not all lists have 4 bounding boxes
            # we want all lists to have the same length
            for i in range(0,b):
                a.insert(len(a),0)

            # reshape to horizontal because the above code makes the list vertical
            a = np.array(a).reshape(1,length)
            
            # stack
            h = np.vstack((h,a))

    # delete the first row because we added this row just to make the code run
    h = np.delete(h, 0, axis=0)
    
    return h


# call the function
box_rows = create_bounding_rows(df_train)
# concat box_rows with df_y

# put box_rows in a dataframe
df_y = pd.DataFrame(box_rows)

# rename the columns in df_box_rows
new_names = ['conf_1', 'x_1', 'y_1', 'width_1', 'height_1',
           'conf_2', 'x_2', 'y_2', 'width_2', 'height_2',
           'conf_3', 'x_3', 'y_3', 'width_3', 'height_3',
           'conf_4', 'x_4', 'y_4', 'width_4', 'height_4']

df_y.columns = new_names

# Let's choose only the first two bounding boxes for each sample
df_y = df_y[['conf_1', 'x_1', 'y_1', 'width_1', 'height_1',
           'conf_2', 'x_2', 'y_2', 'width_2', 'height_2']]

# add the patientId column to df_y
df_y['patientId'] = df_train['patientId']
df_y.shape
df_y.head(2)
# shuffle df_y
from sklearn.utils import shuffle

df_y = shuffle(df_y)

df_train_images, df_val_images = train_test_split(df_y, test_size=0.20,
                                                   random_state=5)

print(df_train_images.shape)
print(type(df_train_images))
print(df_val_images.shape)
print(type(df_val_images))
# Reset the index of df_train_images and df_val_images.

# We do this because we are going to loop through these dataframes in the next step so
# we need the index to be sequential, starting from 0.

df_train_images.reset_index(inplace=True)

df_val_images.reset_index(inplace=True)
# Create a version without any unnecessary columns.

df_train = df_train_images.drop(['index', 'patientId'], axis=1)

df_val = df_val_images.drop(['index','patientId'], axis=1)

# check that we have only 10 columns
print(df_train.shape)
print(df_val.shape)
df_train_images.head(1)
df_val_images.head(1)
df_train.head(1)
df_val.head(1)
# We have 20547 train images and 5137 validation images.
def train_generator(df_train_images, df_train, batch_size, num_rows, num_cols):
    
    '''
    Input: Dataframes, df_train_images and df_train
    
    Outputs one batch (X_train, y_train) on each iteration of the for loop.
    
    X_train:
    Reads images from a folder, converts the images to a numpy array 
    with shape: (batch_size, num_rows, num_cols, 1)
    
    y_train:
    Takes data from a pandas dataframe. Converts the data into a numpy array
    with shape (batch_size, num_rows, num_cols, 1)
    
    '''
    
    
    while True: 

        batch = []
        k = 0


        # note that we are rounding down.
        num_batches = math.ceil(df_train_images.shape[0]/batch_size)

        # create an empty numpy array matching the number of images
        image_array = np.zeros((batch_size,num_rows,num_cols))



        # this loop runs only once each time the next() function is called.
        for i in range(0,num_batches): # 20547 rows in train_images. we are using only 20000 of them

            if i < num_batches-1:

                # [1] Create X_train

                # carve out 1000 rows of the 'patientId' column
                batch = list(df_train_images['patientId'][k:(i+1)*batch_size])

                #for patientId in batch:
                for j in range(0,len(batch)):
                    patientId = batch[j]


                    path = \
                '../input/rsna-pneumonia-detection-challenge/stage_1_train_images/%s.dcm' % patientId

                    dcm_data = pydicom.read_file(path)

                    # get the image as a numpy array
                    image = dcm_data.pixel_array

                    # resize the image
                    small_image = resize(image,(num_rows,num_cols))

                    # add the image to the empty numpy array
                    image_array[j,:,:] = small_image

                # reshape the array and normalize
                X_train = image_array.reshape(batch_size,num_rows,num_cols,1)/255

                # [2] Create y_train

                # note: Here we use df_train instead of df_train_images
                # because we don't want the output to have the patientId column.

                # carve out 1000 rows
                y_train = df_train[k:(i+1)*batch_size]

                # convert to a numpy array
                y_train = y_train.values

            # to cater for the last batch i.e. the fractional part
            if i == num_batches-1: 

                batch_size_fractional = df_train.shape[0] - (batch_size*(num_batches-1)) # -1

                # create an empty numpy array matching the number of images
                image_array = np.zeros((batch_size_fractional,num_rows,num_cols))

                # select rows from the tail of df_test upwards
                batch1 = list(df_train_images['patientId'][-batch_size_fractional:]) #1000

                #for patientId in batch:
                for j in range(0,len(batch1)):
                    patientId = batch1[j]

                    path = \
            '../input/rsna-pneumonia-detection-challenge/stage_1_train_images/%s.dcm' % patientId

                    dcm_data = pydicom.read_file(path)

                    # get the image as a numpy array
                    image = dcm_data.pixel_array

                    # resize the image
                    small_image = resize(image,(num_rows,num_cols))

                    # add the image to the empty numpy array
                    image_array[j,:,:] = small_image

                # reshape the array and normalize
                X_train = image_array.reshape(batch_size_fractional,num_rows,num_cols,1)/255

                # [2] Create y_train

                # note: Here we use df_val instead of df_val_images
                # because we don't want the output to have the patientId column.

                # carve out 1000 rows
                y_train = df_train[-batch_size_fractional:]

                # convert to a numpy array
                y_train = y_train.values


            k = k + batch_size

            # For testing the generator so we can see how many batches it outputs
            # by calling next(). Uncomment the next line for testing.
            #print(i)

            # Keras requires a tuple in the form (inputs,targets)
            yield (X_train.astype(np.float32), y_train)
            
    

def val_generator(df_val_images, df_val, batch_size, num_rows, num_cols):
    
    '''
    Input: Dataframes, df_val_images and df_val
    
    Outputs one batch (X_val, y_val) on each iteration of the for loop.
    
    X_val:
    Reads images from a folder, converts the images to a numpy array 
    with shape: (batch_size, num_rows, num_cols, 1)
    
    y_val:
    Takes data from a pandas dataframe. Converts the data into a numpy array
    with shape (batch_size, num_rows, num_cols, 1)
    
    '''
    
    
    while True: 

        batch = []
        k = 0

        # note that we are rounding up.
        num_batches = math.ceil(df_val_images.shape[0]/batch_size)

        # Create an empty numpy array that matches the batch size.
        image_array = np.zeros((batch_size,num_rows,num_cols))


         # this loop runs only once each time the next() function is called.
        for i in range(0,num_batches): 
            
            if i < num_batches-1:

                # [1] Create X_train

                # carve out a batch of rows of the 'patientId' column
                batch = list(df_val_images['patientId'][k:(i+1)*batch_size])

                #for patientId in batch:
                for j in range(0,len(batch)):
                    patientId = batch[j]

                    path = \
            '../input/rsna-pneumonia-detection-challenge/stage_1_train_images/%s.dcm' % patientId

                    dcm_data = pydicom.read_file(path)

                    # get the image as a numpy array
                    image = dcm_data.pixel_array

                    # resize the image
                    small_image = resize(image,(num_rows,num_cols))

                    # add the image to the empty numpy array
                    image_array[j,:,:] = small_image

                # reshape the array and normalize
                X_val = image_array.reshape(batch_size,num_rows,num_cols,1)/255

                # [2] Create y_train

                # note: Here we use df_val instead of df_val_images
                # because we don't want the output to have the patientId column.

                # carve out 1000 rows
                y_val = df_val[k:(i+1)*batch_size]

                # convert to a numpy array
                y_val = y_val.values

             # to cater for the last batch i.e. the fractional part
            if i == num_batches-1: 

                batch_size_fractional = df_val.shape[0] - (batch_size*(num_batches-1)) 

                # create an empty numpy array matching the number of images
                image_array = np.zeros((batch_size_fractional,num_rows,num_cols))

                # select rows from the tail of df_test upwards
                batch1 = list(df_val_images['patientId'][-batch_size_fractional:]) 

                #for patientId in batch:
                for j in range(0,len(batch1)):
                    patientId = batch1[j]

                    path = \
            '../input/rsna-pneumonia-detection-challenge/stage_1_train_images/%s.dcm' % patientId

                    dcm_data = pydicom.read_file(path)

                    # get the image as a numpy array
                    image = dcm_data.pixel_array

                    # resize the image
                    small_image = resize(image,(num_rows,num_cols))

                    # add the image to the empty numpy array
                    image_array[j,:,:] = small_image

                # reshape the array and normalize
                X_val = image_array.reshape(batch_size_fractional,num_rows,num_cols,1)/255

                # [2] Create y_train

                # note: Here we use df_val instead of df_val_images
                # because we don't want the output to have the patientId column.

                # carve out a batch of rows
                y_val = df_val[-batch_size_fractional:]

                # convert to a numpy array
                y_val = y_val.values


            k = k + batch_size

            # For testing the generator so we can see how many batches it outputs
            # by calling next().
            #print(i)

            # Keras requires a tuple in the form (inputs,targets)
            yield (X_val.astype(np.float32), y_val)
            
           
    
df_test.head(1)
df_test.shape
# There are 1000 rows in df_test i.e. 1000 test images
def test_generator(df_test, batch_size, num_rows, num_cols):
    
    """
    Input: Dataframe df_test.
    
    Outputs one batch (X_test) on each iteration of the for loop.
    
    X_test:
    Reads images from a folder, converts the images to a numpy array 
    with shape: (batch_size, num_rows, num_cols, 1)
    
    """

    batch = []
    k = 0
    
    # note that we are rounding up.
    num_batches = math.ceil(df_test.shape[0]/batch_size)

    # create an empty numpy array matching the number of images
    image_array = np.zeros((batch_size,num_rows,num_cols))
    
    # this loop runs only once each time the next() function is called.
    for i in range(0,num_batches):
        
        if i < num_batches-1:
        
            # [1] Create X_test

            # carve out a batch of rows of the 'patientId' column
            batch = list(df_test['patientId'][k:(i+1)*batch_size]) #1000

            #for patientId in batch:
            for j in range(0,len(batch)):
                patientId = batch[j]

                path = \
        '../input/rsna-pneumonia-detection-challenge/stage_1_test_images/%s.dcm' % patientId

                dcm_data = pydicom.read_file(path)

                # get the image as a numpy array
                image = dcm_data.pixel_array

                # resize the image
                small_image = resize(image,(num_rows,num_cols))

                # add the image to the empty numpy array
                image_array[j,:,:] = small_image

            # reshape the array and normalize
            X_test = image_array.reshape(batch_size,num_rows,num_cols,1)/255
            
        # to cater for the last batch i.e. the fractional part
        if i == num_batches-1: 
            
            batch_size_fractional = df_test.shape[0] - (batch_size*(num_batches - 1))
            
            # create an empty numpy array matching the number of images
            image_array = np.zeros((batch_size_fractional,num_rows,num_cols))
            
            # select rows from the tail of df_test upwards
            batch = list(df_test['patientId'][-batch_size_fractional:]) #1000

            
            for j in range(0,len(batch)):
                patientId = batch[j]

                path = \
        '../input/rsna-pneumonia-detection-challenge/stage_1_test_images/%s.dcm' % patientId

                dcm_data = pydicom.read_file(path)

                # get the image as a numpy array
                image = dcm_data.pixel_array

                # resize the image
                small_image = resize(image,(num_rows,num_cols))

                # add the image to the empty numpy array
                image_array[j,:,:] = small_image

            # reshape the array and normalize
            X_test = image_array.reshape(batch_size_fractional,num_rows,num_cols,1)/255
            
        
        # For testing the generator so we can see how many batches it outputs
        # by calling next(). Uncomment the next line for testing.
        #print(i)
        
        k = k + batch_size
        
        # Keras requires a tuple in the form (inputs,targets)
        yield (X_test.astype(np.float32))
    
# train_generator

#train_gen = \
#train_generator(df_train_images, df_train, batch_size=50, num_rows=500, num_cols=500)

#val_gen = \
#val_generator(df_val_images, df_val, batch_size=10, num_rows=500, num_cols=500)

#test_gen = \
 #test_generator(df_test, batch_size=1000, num_rows=500, num_cols=500)
# Note: Each time this notebook cell is run, the generators will output only one batch.

# If the generators are working correctly, the following shapes should be output:
# X_train (10,500,500,1)
# y_train (10,10)
# X_val (10,500,500,1)
# y_val (10,10)
# X_test(10,500,500,1)

# tuple unpacking
#X_train, y_train = next(train_gen)
#X_val, y_val = next(val_gen)
#X_test = next(test_gen)

#print(X_train.shape)
#print(X_train.dtype)
#print(y_train.shape)
#print(X_val.shape)
#print(X_val.dtype)
#print(y_val.shape)
#print(X_test.shape)
#print(X_test.dtype)
# Check the train_generator()
#train_gen = \
#train_generator(df_train_images, df_train, batch_size=5000,, num_rows=500, num_cols=500)
#X_train, y_train = next(train_gen)

#print(X_train.shape)
#print(y_train.shape)
# Check the val_generator()
#val_gen = \
#val_generator(df_val_images, df_val, batch_size=2000, num_rows=500, num_cols=500)
#X_val, y_val = next(val_gen)

#print(X_val.shape)
#print(y_val.shape) 
# check test_generator
#test_gen = \
#test_generator(df_test, batch_size=300, num_rows=500, num_cols=500)
# Uncomment the print() function in test_generator() before running this cell. 
# Each time this cell is run the output should increment by 1. 
# The last number to be output should be 4.

# Remember to re-quote the print() function after this test.

# With 1000 test samples, a batch size of 300, and image size of 500x500
# these are the shapes we should get with each iteration:

# 0: (300,500,500,1)
# 1: (300,500,500,1)
# 2: (300,500,500,1)
# 3: (100,500,500,1)
# 4: Error

# With 1000 samples and a batch_size of 300 the generator should only run for 4 loops.
# Run this cell 5 times. On the fifth time you should get a "StopIteration" error.

#X_test = next(test_gen)

#X_test.shape
# get the number of train and val images

print(df_train.shape)
print(df_val.shape)
print(df_test.shape)
########################
# INPUTS

# Set the batch sizes:

train_batch_size = 10
val_batch_size = 10
test_batch_size = 1

# Set the image size:

num_rows = 1024
num_cols = 1024

#########################

# train_generator
train_gen = \
train_generator(df_train_images, df_train, train_batch_size, num_rows, num_cols)

num_train_samples = df_train.shape[0]

num_train_batches = math.ceil(num_train_samples/train_batch_size) # round down


# val_generator
val_gen = \
val_generator(df_val_images, df_val, val_batch_size, num_rows, num_cols)

num_val_samples = df_val.shape[0]

num_val_batches = math.ceil(num_val_samples/val_batch_size) # round down

# test_generator
test_gen = \
test_generator(df_test, test_batch_size, num_rows, num_cols)

num_test_samples = df_test.shape[0]

num_test_batches = math.ceil(num_test_samples/test_batch_size) # round up

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                        input_shape=(num_rows, num_cols, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='linear'))


model.summary()
# compile the model
Adam_opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=Adam_opt, loss='mse')
# Notes: 
# To reduce RAM use it's best to keep 'max_queue_size' small.
# The test and val generators run infinitely therefore we must set 
# steps_per_epoch=num_train_batches and validation_steps=num_val_batches so 
# that fit_generator() knows when to stop an epoch and to ensure that the model sees
# the same batch only once.

filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


history = model.fit_generator(generator=train_gen, 
                        steps_per_epoch=num_train_batches, 
                        epochs=3, 
                        verbose=1, 
                        callbacks=callbacks_list, 
                        validation_data=val_gen,
                        validation_steps=num_val_batches, 
                        class_weight=None, 
                        max_queue_size=2, 
                        workers=4,
                        use_multiprocessing=True, 
                        shuffle=False, 
                        initial_epoch=0)

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.legend()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# Initialize the test generator
# Note: Put the intilization in the same cell as the prediction step because the 
# test generator was not designed to run infinitely. We want the prediction process
# to always start at the first batch and run only once.

# I keep the test_batch_size=1 just to ensure that nothing strange happens.

test_gen = \
test_generator(df_test, test_batch_size, num_rows, num_cols)

model.load_weights(filepath = 'model.h5')
predictions = model.predict_generator(test_gen, 
                                      steps=num_test_batches, 
                                      max_queue_size=1, 
                                      workers=1, 
                                      use_multiprocessing=False, 
                                      verbose=1)
predictions.shape
predictions[1]
# put the predictions into a dataframe
df_preds = pd.DataFrame(predictions)

# add column names
new_names = ['conf_1', 'x_1', 'y_1', 'width_1', 'height_1',
       'conf_2', 'x_2', 'y_2', 'width_2', 'height_2']

df_preds.columns = new_names

# add the patientId column
df_preds['patientId'] = df_test['patientId']

# add the PredictionString column
df_preds['PredictionString'] = 0
df_preds.head()
# Version 2: Changes were made. See comments below.

def process_preds(df):
    
    limit = 0.5
    
    conf_1 = 0
    conf_2 = 0
    conf_3 = 0
    conf_4 = 0
    
    string_1 = ''
    string_2 = ''
    string_3 = ''
    string_4 = ''
    
    
    for i in range(0,len(df)):
        
        #get the conf scores
        conf_1 = df.loc[i,'conf_1'] # revised in Version 2
        conf_2 = df.loc[i,'conf_2'] # revised in Version 2
        
        if conf_1 >= limit:
            string_1 = \
            str(conf_1) + ' ' + str(round(df.loc[i,'x_1']))+ ' ' + \
            str(round(df.loc[i,'y_1']))+ ' ' + str(round(df.loc[i,'width_1']))+ ' ' + str(round(df.loc[i,'height_1']))

        if conf_2 >= limit:
            string_2 = \
            str(conf_2) + ' ' + str(round(df.loc[i,'x_2']))+ ' ' + \
            str(round(df.loc[i,'y_2']))+ ' ' + str(round(df.loc[i,'width_2']))+ ' ' + str(round(df.loc[i,'height_2']))

        df.loc[i,'PredictionString']  = \
        string_1 + ' ' + string_2 

    df_submission = df[['patientId', 'PredictionString']]
    
    return df_submission

# call the function
df_submission = process_preds(df_preds)
df_submission.head()

ID = df_preds['patientId']
preds = df_preds['PredictionString']

submission = pd.DataFrame({'patientId':ID, 
                           'PredictionString':preds, 
                          }).set_index('patientId')

submission.to_csv('pneu_keras_model.csv', columns=['PredictionString']) 
# This is a simple example of a generator.
def my_generator():
    for i in range(0,3):
        yield print(i)

my_gen = my_generator()

# If you run this cell 3 times  you will notice that the output increases by 1 each time.
# On the 4th iteration there will be a 'StopIteration'.

out_put = next(my_gen)
out_put
# source: @Liquid_Fire
#https://stackoverflow.com/questions/3704918/
    #python-way-to-restart-a-for-loop-similar-to-continue-for-while-loops

# To use fit_generator() in keras our generator needs to loop infinitely.
# This is how to do that:

def my_generator():

   
    while True: 
        
        for i in range(0,4):
            
            yield i
        
        

infinity_gen = my_generator()

# if you run this cell you will see that a 'StopIteration' never happens.
out_put = next(infinity_gen)
out_put
