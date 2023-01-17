import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random



import os

print(os.listdir("../input"))
# first thing is extracting the files

import os, shutil, zipfile



data = ['train', 'test1']



for el in data:

    with zipfile.ZipFile('../input/' + el + ".zip", "r") as z:

        z.extractall(".")  # extract zip files to current dir
print(os.listdir("."))  # List files in current dir
# Image files in training dir is either cat.x.jpg or dog.x.jpg

# Create a dataframe to label each image file accordingly

filenames = os.listdir("./train")

categories = []  # store the label for each image file

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})

df.head()
df['category'].value_counts().plot.bar()
sample = random.choice(filenames)

image = load_img("./train/"+sample)

plt.imshow(image)

print(sample)
from keras.layers import Dropout, Flatten, Dense

from keras import Model, optimizers

from keras.applications import VGG16



img_width, img_height = 224, 224

target_size = (img_width, img_height)



# Load a pre-trained convolutional neural network (CNN) model

model = VGG16(include_top=False, weights="imagenet",

             input_shape = (img_width, img_height, 3))



# Do not retrain feature extraction layers

for layer in model.layers:

    layer.trainable = False



x = model.output

# Extend the pre-trained model

# Flatten the output layer to 1 dimension

x = Flatten()(x)

x = Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5 to control overfitting

x = Dropout(0.5)(x)

#x = Dense(64, activation='relu')(x)

# Add a final sigmoid layer for classification

predictions = Dense(1, activation='sigmoid')(x)



model_final = Model(inputs = model.input,

                   outputs = predictions)



model_final.summary()
model_final.compile(loss='binary_crossentropy',

#              optimizer='adam',

#              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

              optimizer=optimizers.Adam(lr=1e-4),

              metrics=['accuracy'])
# Split training & validation datasets

train_df, validate_df = train_test_split(df, test_size=0.1)

train_df = train_df.reset_index()

validate_df = validate_df.reset_index()



total_train = train_df.shape[0]

total_validate = validate_df.shape[0]
train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest',

    width_shift_range=0.1,

    height_shift_range=0.1

)



batch_size = 1  #16  # No of images per batch

x_col, y_col = 'filename', 'category'

class_mode = 'binary'



# To avoid the error below:

# TypeError: If class_mode="binary", y_col="category" column values must be strings.

train_df['category'] = train_df['category'].astype(str)  #optional



train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    "./train/",

    x_col=x_col,

    y_col=y_col,

    class_mode=class_mode,

    target_size=target_size,

    batch_size=batch_size

)
validation_datagen = ImageDataGenerator(rescale=1./255)



# To avoid the error below:

# TypeError: If class_mode="binary", y_col="category" column values must be strings.

validate_df['category'] = validate_df['category'].astype(str)



validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    "./train/", 

    x_col=x_col,

    y_col=y_col,

    class_mode=class_mode,

    target_size=target_size,

    batch_size=batch_size

)
example_df = train_df.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(

    example_df, 

    "./train/", 

    x_col='filename',

    y_col='category',

    class_mode='binary'

)

plt.figure(figsize=(12, 12))

for i in range(0, 9):

    plt.subplot(3, 3, i+1)

    for X_batch, Y_batch in example_generator:

        image = X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
# Train the model

epochs = 2

history = model_final.fit_generator(

    train_generator,

    epochs=epochs,

    validation_data=validation_generator,

#    validation_steps=total_validate//batch_size,

#    steps_per_epoch=total_train//batch_size)

    validation_steps=100,

    steps_per_epoch=1000)
def plot_model_history(model_history, acc='acc', val_acc='val_acc'):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(range(1,len(model_history.history[acc])+1),model_history.history[acc])

    axs[0].plot(range(1,len(model_history.history[val_acc])+1),model_history.history[val_acc])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].set_xticks(np.arange(1,len(model_history.history[acc])+1),len(model_history.history[acc])/10)

    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(['train', 'val'], loc='best')

    plt.show()

    

plot_model_history(history)  # Plot the accuracy & loss during training 
# Save the trained model

model_final.save('my_model.h5')