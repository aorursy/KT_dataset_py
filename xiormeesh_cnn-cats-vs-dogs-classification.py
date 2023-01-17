import numpy as np

import pandas as pd

import os



from sklearn.model_selection import train_test_split



import tensorflow as tf

print(tf.__version__)
# preparing data



TRAIN_DIR = "../input/dogs-vs-cats/train/train"

TEST_DIR = "../input/dogs-vs-cats/test1/test1"



TRAIN_SIZE = len([name for name in os.listdir(TRAIN_DIR)])

TEST_SIZE = len([name for name in os.listdir(TEST_DIR)])

print("Number of training images:", TRAIN_SIZE)

print("Number of test images:", TEST_SIZE)



VALID_FRACTION = 0.2

BATCH_SIZE = 100

EPOCHS = 50



IMAGE_WIDTH = IMAGE_HEIGHT = 150



# creating df with train labels

train_filenames = os.listdir(TRAIN_DIR)

train_labels = []

for filename in train_filenames:

    label = filename.split('.')[0]

    train_labels.append(label)



train_df = pd.DataFrame({

    'id': train_filenames,

    'label': train_labels

})



# splitting to train & valid

train_df, valid_df = train_test_split(train_df, test_size=VALID_FRACTION)



# augmentation settings, for now just normalizing

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(    

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    rescale=1./255.,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

    )



# not doing any data augmentation on validation test set

valid_datagen  = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)



# creating train and valid generators (not using valid_split to avoid doing data augmentation on validation set)

train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    TRAIN_DIR, 

    x_col='id',

    y_col='label',

    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

    class_mode='binary',

    batch_size=BATCH_SIZE

)



valid_generator = valid_datagen.flow_from_dataframe(

    valid_df, 

    TRAIN_DIR, 

    x_col='id',

    y_col='label',

    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

    class_mode='binary',

    batch_size=BATCH_SIZE

)
model = tf.keras.models.Sequential([

    # the images were resized by ImageDataGenerator 150x150 with 3 bytes color

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2), 

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Flatten(), 

    # 512 neuron hidden layer

    tf.keras.layers.Dense(512, activation='relu'),

    # since we have only 2 classes to predict we can use 1 neuron and sigmoid

    tf.keras.layers.Dense(1, activation='sigmoid')  

])



model.summary()



model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),

    loss='binary_crossentropy',

    metrics = ['accuracy'])



es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',

    mode='min',

    restore_best_weights=True, 

    verbose=1,

    patience=5)
tf.keras.utils.plot_model(model)
%%time



# training

history = model.fit_generator(train_generator,

    validation_data=valid_generator,

    steps_per_epoch=round(TRAIN_SIZE*(1.-VALID_FRACTION)/BATCH_SIZE),

    validation_steps=round(TRAIN_SIZE*VALID_FRACTION/BATCH_SIZE),

    epochs=EPOCHS,

    callbacks=[es],

    verbose=1)
#plotting



import matplotlib.pyplot as plt



#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

acc = history.history['accuracy']

val_acc = history.history[ 'val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs   = range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot(epochs, acc)

plt.plot(epochs, val_acc)

plt.title('Training and validation accuracy')

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot(epochs, loss)

plt.plot(epochs, val_loss)

plt.title('Training and validation loss')
%%time



# preparing testing data

test_filenames = os.listdir(TEST_DIR)

test_df = pd.DataFrame({

    'id': test_filenames

})



test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.)



test_generator = test_datagen.flow_from_dataframe(

    test_df, 

    TEST_DIR, 

    x_col='id',

    y_col=None,

    class_mode=None,

    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

    batch_size=BATCH_SIZE,

    shuffle=False

)



yhat = model.predict_generator(test_generator, steps=np.ceil(TEST_SIZE/BATCH_SIZE))
# sigmoid returns probability between 0 and 1, need to convert it to an integer class

yhat = [1 if y > 0.5 else 0 for y in yhat]



test_df['label'] = yhat



# restoring back to class names (dog|cat)

label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['label'] = test_df['label'].replace(label_map)



# encoding according to submission format, 1 = dog, 0 = cat

test_df['label'] = test_df['label'].replace({ 'dog': 1, 'cat': 0 })



test_df.to_csv('submission.csv', index=False)
test_df.head()