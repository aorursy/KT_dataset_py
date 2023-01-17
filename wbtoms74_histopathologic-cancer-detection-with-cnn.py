# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def append_ext(fn):

    return fn+".tif"



df=pd.read_csv("../input/train_labels.csv",dtype=str)

df["id"]=df["id"].apply(append_ext)

df.describe()
# Display the head of the train data frame

df.head()
# Get the labels

Y = df["label"]

# Display the histogram of labels

import seaborn as sns

g = sns.countplot(Y)
# Count the number of labels

Y.value_counts()
df_train = df.sample(frac=1, random_state=1)

df_train.describe()
# Get the labels

Y_train = df_train["label"]

# Display the histogram of labels

import seaborn as sns

g_train = sns.countplot(Y_train)

# Count the number of labels

Y_train.value_counts()
crop_dim = (32,32) # (x,y)

img_dim = (96,96) # (x,y)
def crop_center(img, crop_dim, img_dim):

    # Note: image_data_format is 'channel_last'

    assert img.shape[2] == 3

    height, width = img.shape[0], img.shape[1]

    dx, dy = crop_dim

    x, y = (int((img_dim[0]-crop_dim[0])/2),int((img_dim[1]-crop_dim[1])/2))

    return img[y:(y+dy), x:(x+dx), :]





def crop_generator(batches):

    while True:

        batch_x, batch_y = next(batches)

        batch_crops = np.zeros((batch_x.shape[0], crop_dim[0], crop_dim[1], 3))

        for i in range(batch_x.shape[0]):

            batch_crops[i] = crop_center(batch_x[i],crop_dim,img_dim)

        yield (batch_crops, batch_y)





from keras_preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(validation_split=0.20,

                          rescale=1/255.0)



train_generator=datagen.flow_from_dataframe(

        dataframe=df_train,

        directory="../input/train/",

        x_col="id",

        y_col="label",

        subset="training",

        batch_size=160,

        seed=42,

        shuffle=True,

        class_mode="categorical",

        target_size=img_dim)



if 0:

    train_generator_crops = crop_generator(train_generator)



valid_generator=datagen.flow_from_dataframe(

        dataframe=df_train,

        directory="../input/train/",

        x_col="id",

        y_col="label",

        subset="validation",

        batch_size=160,

        seed=42,

        shuffle=True,

        class_mode="categorical",

        target_size=img_dim)



if 0:

    valid_generator_crops = crop_generator(valid_generator)
# Importing libraries

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout



#height, width = crop_dim

height, width = img_dim



# Initialising the CNN

classifier = Sequential()



classifier.add(Conv2D(32, (3, 3), input_shape = (height,width,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.20))



classifier.add(Conv2D(64, (3, 3),activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.20))



classifier.add(Conv2D(128, (3, 3),activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.20))



# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(units = 256, activation = 'relu'))

classifier.add(Dropout(0.20))

classifier.add(Dense(units = 64, activation = 'relu'))

classifier.add(Dropout(0.20))

classifier.add(Dense(units = 64, activation = 'relu'))

classifier.add(Dropout(0.20))

classifier.add(Dense(units = 64, activation = 'relu'))

classifier.add(Dropout(0.20))

classifier.add(Dense(units = 64, activation = 'relu'))

classifier.add(Dropout(0.20))

classifier.add(Dense(units = 2, activation = 'softmax')) #softmax for classification
# Compiling the CNN

from keras import optimizers

classifier.compile(optimizer = 'adam', 

                   loss = 'categorical_crossentropy', 

                   metrics = ['accuracy'])
# Not used at the moment

if 0:

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

    history = classifier.fit_generator(generator=train_generator_crops,

                             steps_per_epoch=STEP_SIZE_TRAIN,

                             validation_data=valid_generator_crops,

                             validation_steps=STEP_SIZE_VALID,

                             epochs=10,

                             verbose=2)
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

history = classifier.fit_generator(generator=train_generator,

                         steps_per_epoch=STEP_SIZE_TRAIN,

                         validation_data=valid_generator,

                         validation_steps=STEP_SIZE_VALID,

                         epochs=20,

                         verbose=2)
# Display model performance over epochs

from matplotlib import pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(15,5))

ax[0].set_title('loss')

ax[0].plot(history.epoch, history.history["loss"], label="Train loss")

ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")

ax[0].legend()

ax[1].set_title('acc')

ax[1].plot(history.epoch, history.history["acc"], label="Train acc")

ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")

ax[1].legend()
# Not used at the moment

if 0:

    score = classifier.evaluate_generator(generator=valid_generator_crops, steps=STEP_SIZE_VALID, workers=1)

    print('Score: ',score)

    print('Metrics: ',classifier.metrics_names)

    classifier.summary()
score = classifier.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID, workers=1)

print('Score: ',score)

print('Metrics: ',classifier.metrics_names)

classifier.summary()
df_test=pd.read_csv("../input/sample_submission.csv",dtype=str)

df_test["id"]=df_test["id"].apply(append_ext)

df_test.describe()
df_test.isnull().values.any()
test_datagen=ImageDataGenerator(rescale=1/255.0)



test_generator=test_datagen.flow_from_dataframe(

        dataframe=df_test,

        directory="../input/test/",

        x_col="id",

        y_col=None,

        batch_size=1,

        seed=42,

        shuffle=False,

        class_mode=None,

        target_size=img_dim)
def crop_generator_single(batches):

    while True:

        batch_x = next(batches)

        batch_crops = np.zeros((batch_x.shape[0], crop_dim[0], crop_dim[1], 3))

        for i in range(batch_x.shape[0]):

            batch_crops[i] = crop_center(batch_x[i],crop_dim,img_dim)

        yield (batch_crops)

        

if 0:

    test_generator_crops = crop_generator_single(test_generator)



STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

test_generator.reset()



if 0:

    pred=classifier.predict_generator(test_generator_crops,

                                      steps=STEP_SIZE_TEST,

                                      verbose=1)



    

pred=classifier.predict_generator(test_generator,

                                  steps=STEP_SIZE_TEST,

                                  verbose=1)
print(pred.shape)
print(pred[:5,:])
predicted_class_indices=np.argmax(pred,axis=1)





labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]



filenames=test_generator.filenames

print(labels)
print(predictions[:5])

print(len(predictions))

# Display the histogram of labels

import seaborn as sns

g = sns.countplot(predictions)
# Remove the file extension .tif

print(len(filenames))

import os

print(os.path.splitext(os.path.basename(filenames[0]))[0])

for i in range(0,len(filenames)):

    filenames[i] = os.path.splitext(os.path.basename(filenames[i]))[0]
# Create submission file

submission=pd.DataFrame({"id":filenames,"label":predictions})

submission.to_csv("Submission.csv",index=False)